"""LiveCodeBench normalization and its official execution grader in Docker.

Only public problem text is sent to the model. Test cases remain in the local
bank and enter a network-disabled, credential-free grading container.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import pickle
import re
import subprocess
import tempfile
import uuid
import zlib

from .api_scoring import final_answer_text


class _StringsOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("LiveCodeBench test payload must not contain executable pickle objects")


def _tests(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise ValueError("LiveCodeBench test payload must be a list or string")
    try:
        result = json.loads(value)
    except (ValueError, TypeError):
        try:
            compressed = base64.b64decode(value, validate=True)
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed, 32 * 1024 * 1024)
            if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
                raise ValueError("Invalid or oversized compressed test payload")
            stream = io.BytesIO(raw)
            decoded = _StringsOnly(stream).load()
            if stream.read(1):
                raise ValueError("Trailing data in compressed test payload")
        except (TypeError, ValueError, pickle.UnpicklingError, EOFError, zlib.error):
            raise ValueError("Invalid or unsafe compressed LiveCodeBench test payload") from None
        if not isinstance(decoded, str):
            raise ValueError("Compressed test payload must contain a JSON string")
        result = json.loads(decoded)
    if not isinstance(result, list):
        raise ValueError("Test cases must be a list")
    return result


def normalize_livecodebench(rows: list[dict]) -> list[dict]:
    result, seen = [], set()
    for row in rows:
        required = {"question_id", "question_content", "public_test_cases", "private_test_cases", "metadata"}
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError("LiveCodeBench source row is missing required fields")
        if (not isinstance(row["question_id"], (str, int)) or isinstance(row["question_id"], bool)
                or not str(row["question_id"]).strip()):
            raise ValueError("LiveCodeBench requires an original nonempty question_id")
        question_id = str(row["question_id"])
        if question_id in seen:
            raise ValueError(f"Duplicate LiveCodeBench question_id: {question_id}")
        seen.add(question_id)
        public, private = _tests(row["public_test_cases"]), _tests(row["private_test_cases"])
        tests = public + private
        if not tests or any(not isinstance(t, dict) or not {"input", "output"} <= set(t)
                            or not isinstance(t["input"], str) or not isinstance(t["output"], str)
                            for t in tests):
            raise ValueError(f"Missing or invalid execution tests for {question_id}")
        metadata = row["metadata"]
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        if not isinstance(metadata, dict):
            raise ValueError("LiveCodeBench metadata must be an object")
        function = metadata.get("func_name")
        if function is not None and (not isinstance(function, str) or not function.strip()):
            raise ValueError("LiveCodeBench func_name must be a nonempty string or null")
        expected_type = "functional" if function is not None else "stdin"
        if any(test.get("testtype", expected_type) != expected_type for test in tests):
            raise ValueError("LiveCodeBench test types do not match its function metadata")
        starter = row.get("starter_code", "")
        if not isinstance(starter, str) or not isinstance(row["question_content"], str) or not row["question_content"].strip():
            raise ValueError("LiveCodeBench requires problem text and a string starter_code")
        instruction = ("Complete the provided Python starter code.\n```python\n" + starter + "\n```"
                       if starter else "Write a Python program that reads stdin and writes stdout.")
        prompt = (row["question_content"] + "\n\n" + instruction
                  + "\nReturn the solution in a fenced Python code block.")
        sample = {"inputs": [t["input"] for t in tests], "outputs": [t["output"] for t in tests],
                  "fn_name": function}
        result.append({"benchmark": "livecodebench", "question_id": question_id,
                       "prompt": prompt, "metadata": {
                           "stratum": str(row.get("difficulty", "unknown")),
                           "platform": row.get("platform"), "contest_date": row.get("contest_date"),
                           "prompt_protocol": "qwenpt-lcb-python-v1",
                           "lcb_sample": {"input_output": json.dumps(sample, ensure_ascii=False)}}})
    if not result:
        raise ValueError("LiveCodeBench source contains no tasks")
    return result


def validate_checkout(checkout: str, revision: str) -> Path:
    if not isinstance(checkout, str) or not checkout.strip():
        raise ValueError("LiveCodeBench requires a local official checkout")
    path = Path(checkout).expanduser().resolve()
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("LiveCodeBench requires an exact 40-character grader Git revision")
    try:
        kwargs = {"text": True, "encoding": "utf-8", "stderr": subprocess.DEVNULL, "timeout": 30}
        root = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"], **kwargs).strip()
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], **kwargs).strip()
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], **kwargs)
    except (OSError, subprocess.SubprocessError):
        raise ValueError("LiveCodeBench checkout unavailable; clone and checkout the configured revision") from None
    if Path(root).resolve() != path:
        raise ValueError("LiveCodeBench checkout must be the root of its own Git clone")
    if head != revision or dirty:
        raise ValueError("LiveCodeBench grader checkout differs from the frozen revision")
    if not (path / "lcb_runner/evaluation/compute_code_generation_metrics.py").is_file():
        raise ValueError("Checkout does not contain the official LiveCodeBench evaluator")
    return path


def preflight_livecodebench(scorer: dict) -> dict:
    import os
    checkout = validate_checkout(scorer.get("checkout", ""), scorer.get("revision", ""))
    image = scorer.get("image") or os.environ.get(scorer.get("image_env", "LCB_SANDBOX_IMAGE"), "")
    if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Set LCB_SANDBOX_IMAGE to the immutable sha256 image ID built from Dockerfile.livecodebench")
    try:
        actual = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True,
                                         stderr=subprocess.DEVNULL, timeout=30).strip()
    except (OSError, subprocess.SubprocessError):
        raise ValueError("LiveCodeBench requires a running Docker engine and the local grader image") from None
    if actual != image:
        raise ValueError("LiveCodeBench grader image identity mismatch")
    return {"checkout": str(checkout), "revision": scorer["revision"], "image": image}


def score_livecodebench(question: dict, text: str, scorer: dict) -> dict:
    sample = question.get("metadata", {}).get("lcb_sample")
    if not isinstance(sample, dict) or set(sample) != {"input_output"} or not isinstance(sample.get("input_output"), str):
        raise ValueError("LiveCodeBench question is missing its frozen execution sample")
    timeout = scorer.get("timeout", 6)
    # Upstream passes this directly to signal.alarm(), which requires an int.
    if type(timeout) is not int or not 0 < timeout <= 120:
        raise ValueError("LiveCodeBench test timeout must be an integer in [1, 120]")
    tests = json.loads(sample["input_output"])
    if (not isinstance(tests, dict) or set(tests) != {"inputs", "outputs", "fn_name"}
            or not isinstance(tests["inputs"], list) or not tests["inputs"]
            or not isinstance(tests["outputs"], list) or len(tests["inputs"]) != len(tests["outputs"])
            or any(not isinstance(item, str) for item in tests["inputs"] + tests["outputs"])
            or (tests["fn_name"] is not None and (not isinstance(tests["fn_name"], str) or not tests["fn_name"].strip()))):
        raise ValueError("LiveCodeBench execution sample has invalid inputs, outputs, or function name")
    identity = preflight_livecodebench(scorer)
    budget = min(3600, 60 + (timeout + 1) * len(tests["inputs"]))
    name = "qwenpt-lcb-" + uuid.uuid4().hex
    worker = Path(__file__).with_name("lcb_worker.py").resolve()
    with tempfile.TemporaryDirectory(prefix="qwenpt-lcb-") as directory:
        payload = Path(directory) / "input.json"
        payload.write_text(json.dumps({"sample": sample, "text": final_answer_text(text), "timeout": timeout}, allow_nan=False), encoding="utf-8")
        command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
                   "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "256",
                   "--memory", "2g", "--cpus", "2", "--user", "65534:65534",
                   "--tmpfs", "/tmp:rw,exec,size=512m", "--tmpfs", "/dev/shm:rw,size=128m",
                   "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONPATH=/opt/lcb",
                   # Mount code only, excluding .git credential configuration.
                   "--mount", f"type=bind,src={Path(identity['checkout']) / 'lcb_runner'},dst=/opt/lcb/lcb_runner,readonly",
                   "--mount", f"type=bind,src={payload},dst=/input.json,readonly",
                   "--mount", f"type=bind,src={worker},dst=/worker.py,readonly",
                   identity["image"], "python", "/worker.py"]
        try:
            completed = subprocess.run(command, text=True, encoding="utf-8", capture_output=True, timeout=budget)
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
            raise RuntimeError("LiveCodeBench grading infrastructure exceeded its outer timeout") from None
        if completed.returncode:
            raise RuntimeError("LiveCodeBench grading container failed; verify grader dependencies/image")
        lines = [line.removeprefix("QWENPT_LCB_RESULT=") for line in completed.stdout.splitlines()
                 if line.startswith("QWENPT_LCB_RESULT=")]
        if len(lines) != 1:
            raise RuntimeError("LiveCodeBench container did not return exactly one verdict")
        try:
            result = json.loads(lines[0])
        except (ValueError, TypeError):
            raise RuntimeError("LiveCodeBench returned invalid verdict JSON") from None
        if not isinstance(result, dict) or type(result.get("infra_error")) is not bool:
            raise RuntimeError("LiveCodeBench returned an invalid verdict")
        if result.get("infra_error"):
            raise RuntimeError("Official LiveCodeBench evaluator reported a TestRunnerError")
        values = result.get("test_results")
        if (type(result.get("correct")) is not bool or not isinstance(result.get("code"), str)
                or not isinstance(result.get("metadata"), dict)
                or not isinstance(values, list) or not values or len(values) > len(tests["inputs"])
                or any(type(value) not in {bool, int} or value not in {-4, -3, -2, -1, 0, 1} for value in values)):
            raise RuntimeError("LiveCodeBench returned an invalid verdict")
        expected_correct = len(values) == len(tests["inputs"]) and all(value is True or value == 1 for value in values)
        if result["correct"] != expected_correct:
            raise RuntimeError("LiveCodeBench verdict does not match complete execution results")
        return {"score": float(result["correct"]), "correct": result["correct"], "metric": "pass@1",
                "predicted": result.get("code", ""), "execution": result, "grader": identity}
