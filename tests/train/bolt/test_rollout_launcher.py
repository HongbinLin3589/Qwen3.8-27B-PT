"""Verify the rollout submission graph without requesting cluster resources."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_submit_queues_eight_single_gpu_shards(tmp_path):
    project = tmp_path / "project with spaces"
    scripts = project / "scripts/train"
    scripts.mkdir(parents=True)
    script = scripts / "bolt_rollout.sh"
    shutil.copyfile(Path(__file__).resolve().parents[3] / "scripts/train/bolt_rollout.sh", script)
    (project / "pyproject.toml").touch()
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "submissions.jsonl"
    sbatch = binary / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "assert all(key not in os.environ for key in "
        "['http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'])\n"
        "path=pathlib.Path(os.environ['BOLT_TEST_CALLS'])\n"
        "count=len(path.read_text().splitlines()) if path.exists() else 0\n"
        "with path.open('a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print(100+count)\n"
    )
    sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update(PATH=f"{binary}:{env['PATH']}", BOLT_TEST_CALLS=str(calls), https_proxy="test-proxy")
    env.pop("SLURM_SUBMIT_DIR", None)
    subprocess.run(["bash", str(script), "submit", "--dependency=afterok:99"],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    submissions = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(submissions) == 8
    assert "#SBATCH --gres=gpu:1" in script.read_text()
    for shard_id, arguments in enumerate(submissions[:8]):
        assert "--dependency=afterok:99" in arguments
        assert arguments[-2:] == ["generate", str(shard_id)]
    assert "--finalize-output" in script.read_text()
