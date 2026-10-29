"""Entry point mounted only inside the isolated LiveCodeBench grader container."""
import contextlib
import io
import json


def main():
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from lcb_runner.utils.extraction_utils import extract_code
    from lcb_runner.lm_styles import LMStyle

    with open("/input.json", encoding="utf-8") as stream:
        payload = json.load(stream)
    code = extract_code(payload["text"], LMStyle.OpenAIChat)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        metrics = codegen_metrics([payload["sample"]], [[code]], k_list=[1], num_process_evaluate=1,
                                  timeout=payload["timeout"])
    values = metrics[1][0][0]
    metadata = json.loads(metrics[2][0][0])
    result = {"correct": bool(values) and all(value is True or value == 1 for value in values),
              "test_results": values, "metadata": metadata, "code": code,
              "infra_error": metadata.get("error_code") == -5}
    print("QWENPT_LCB_RESULT=" + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
