"""Verify exported math rollouts through the native BOLT training dataset on CPU."""

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile

from qwenpt.train.common.files import sha256


def verify_bolt_splits(output: str, tokenizer_path: str, max_length: int = 12288) -> dict:
    """Check every exported row and atomically save a reusable native-input audit."""
    root = Path(output).resolve()
    manifest_path = root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = {name: sha256(root / name) for name in ("train.parquet", "val.parquet")}
    if manifest["output_sha256"] != hashes:
        raise ValueError("BOLT Parquet hashes differ from the split manifest")
    tokenizer_root = Path(tokenizer_path).resolve()
    bindings = {
        "format_version": 1, "output_sha256": hashes, "manifest_sha256": sha256(manifest_path),
        "verifier_sha256": sha256(Path(__file__).resolve()),
        "tokenizer": str(tokenizer_root), "max_length": max_length,
        "tokenizer_sha256": {name: sha256(tokenizer_root / name) for name in (
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        ) if (tokenizer_root / name).is_file()},
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "pyarrow")},
        "adapter_sha256": {
            name: sha256(Path(__file__).resolve().parents[1] / relative)
            for name, relative in (
                ("dataset.py", "common/dataset.py"),
                ("bolt_dataset.py", "bolt/dataset.py"),
                ("template.py", "common/template.py"),
            )
        },
    }
    audit_path = root / "native_input_check.json"
    if audit_path.exists():
        cached = json.loads(audit_path.read_text())
        if cached.get("bindings") == bindings and cached.get("status") == "passed":
            return cached
    import torch
    from transformers import AutoTokenizer
    from qwenpt.train.bolt.dataset import QwenBOLTDataset

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    report = {"bindings": bindings, "status": "passed", "splits": {}}
    seen = set()
    for split, filename, prefix in (("train", "train.parquet", "train"),
                                    ("validation", "val.parquet", "validation")):
        dataset = QwenBOLTDataset(str(root / filename), tokenizer, {
            "max_length": max_length, "loss_scope": "all_assistant", "bolt_beta": 0.1,
        })
        groups = Counter(row["prompt_id"] for row in dataset.rows)
        if any(count != 8 for count in groups.values()) or seen.intersection(groups):
            raise ValueError(f"{split}: expected disjoint complete groups of eight candidates")
        if set(groups) != set(manifest["split_prompt_ids"][split]):
            raise ValueError(f"{split}: prompt IDs differ from the split manifest")
        if len(dataset) != manifest["counts"][f"{prefix}_rows"] or len(groups) != manifest["counts"][f"{prefix}_prompts"]:
            raise ValueError(f"{split}: counts differ from the split manifest")
        seen.update(groups)
        maximum, minimum, weight_sum = 0, max_length, 0.0
        for index in range(len(dataset)):
            item = dataset[index]
            ids, mask, weight = item["input_ids"], item["loss_mask"], item["bolt_weight"]
            if ids.shape != mask.shape or not torch.all((mask == 0) | (mask == 1)) or mask.sum().item() <= 0:
                raise ValueError(f"{split} row {index}: invalid assistant supervision mask")
            if weight.shape != (1,) or not torch.isfinite(weight).all() or (weight < 0).any():
                raise ValueError(f"{split} row {index}: invalid BOLT weight")
            maximum = max(maximum, ids.numel())
            minimum = min(minimum, int(mask.sum().item()))
            weight_sum += weight.item()
        mean_weight = weight_sum / len(dataset)
        if not math.isclose(mean_weight, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{split}: mean BOLT weight differs from one")
        report["splits"][split] = {
            "rows": len(dataset), "prompt_groups": len(groups), "max_sequence_tokens": maximum,
            "min_supervised_tokens": minimum, "mean_bolt_weight": mean_weight,
            "reward_counts": dict(Counter(str(row["reward"]) for row in dataset.rows)),
            "finish_reason_counts": dict(Counter(row["finish_reason"] for row in dataset.rows)),
        }
        print(json.dumps({"verified_split": split, **report["splits"][split]}), flush=True)
    if (len(seen) != manifest["counts"]["total_prompts"]
            or sum(part["rows"] for part in report["splits"].values()) != manifest["counts"]["total_rows"]):
        raise ValueError("Total counts differ from the split manifest")
    with tempfile.NamedTemporaryFile("w", dir=root, prefix=".native-input-", delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, audit_path)
    return report


def main() -> None:
    """Run CPU validation of finalized BOLT train and validation Parquet files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=12288)
    args = parser.parse_args()
    verify_bolt_splits(args.output, args.tokenizer, args.max_length)


if __name__ == "__main__":
    main()
