"""Generate tiny Parquet fixtures for SFT/BOLT tests and local debugging."""

import argparse
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


def write_synthetic_data(directory: str | Path, include_image: bool = False) -> dict[str, str]:
    """Write small train/validation Parquet files under directory, returning paths.

    Samples follow the delivered format with reasoning, multiple turns and tools;
    ``include_image`` appends one generated screenshot with embedded PNG bytes.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = [
        {
            "sample_id": "synthetic-math",
            "messages": [
                {"role": "system", "content": "你是数学助手。"},
                {"role": "user", "content": "3 + 2 等于多少？"},
                {"role": "assistant", "content": "<think>把两个整数相加，得到 5。</think>答案是 5。"},
            ],
            "reasoning_effort": None,
        },
        {
            "sample_id": "synthetic-multiturn",
            "messages": [
                {"role": "user", "content": "2 的平方是多少？"},
                {"role": "assistant", "reasoning_content": "2 乘以 2 等于 4。", "content": "4。"},
                {"role": "user", "content": "再加 1 呢？"},
                {"role": "assistant", "reasoning_content": "在 4 的基础上加 1。", "content": "5。"},
            ],
        },
        {
            "sample_id": "synthetic-tool",
            "messages": [
                {"role": "user", "content": "调用加法工具计算 3 + 2。"},
                {
                    "role": "assistant", "content": "", "reasoning_content": "先调用工具取得结果。",
                    "tool_calls": [{"id": "sum-1", "type": "function", "function": {
                        "name": "add", "arguments": '{"a": 3, "b": 2}',
                    }}],
                },
                {"role": "tool", "tool_call_id": "sum-1", "content": "TOOL_RESULT_5"},
                {"role": "assistant", "reasoning_content": "工具返回的结果是 5。", "content": "答案是 5。"},
            ],
            "tools": [{"type": "function", "function": {
                "name": "add", "description": "Add two integers.",
                "parameters": {"type": "object", "properties": {
                    "a": {"type": "integer"}, "b": {"type": "integer"},
                }, "required": ["a", "b"]},
            }}],
        },
    ]
    if include_image:
        stream = BytesIO()
        Image.new("RGB", (32, 32), "blue").save(stream, format="PNG")
        samples.append({
            "sample_id": "synthetic-image",
            "messages": [
                {"role": "user", "content": "<image>图片主要是什么颜色？"},
                {"role": "assistant", "reasoning_content": "图片中只有蓝色。", "content": "蓝色。"},
            ],
            "images": [{"bytes": stream.getvalue()}],
        })
    # Explicit nullable columns keep Parquet from inferring only the first row's fields.
    fields = ("sample_id", "messages", "tools", "images", "videos", "enable_thinking", "reasoning_effort")
    rows = [{field: sample.get(field, True if field == "enable_thinking" else None) for field in fields}
            for sample in samples]
    table = pa.Table.from_pylist(rows)
    paths = {split: str(directory / f"{split}.parquet") for split in ("train", "val")}
    for path in paths.values():
        if Path(path).exists():
            raise FileExistsError(f"Refusing to overwrite an existing dataset: {path}")
    for path in paths.values():
        pq.write_table(table, path)
    return paths


def main() -> None:
    """Write synthetic train/validation files to --output and print their paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, metavar="DIRECTORY")
    parser.add_argument("--include-image", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write_synthetic_data(args.output, args.include_image), ensure_ascii=False))


if __name__ == "__main__":
    main()
