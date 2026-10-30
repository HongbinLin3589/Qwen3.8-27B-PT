"""Check receipt-gated BOLT export with real Parquet finalization on CPU."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from qwenpt.train.bolt import complete as bolt_complete
from qwenpt.train.bolt.data import load_math_pool
from qwenpt.train.common.files import sha256


class TestCompleteShard(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pool_file = self.root / "pool.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"messages": [{"role": "user", "content": f"Problem {index}"}], "ground_truth": "1"}
            for index in range(10)
        ]), self.pool_file)
        self.pool = load_math_pool([str(self.pool_file)])
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.output = self.root / "export"
        for index in range(2):
            self.write_shard(index)
        self.args = argparse.Namespace(
            output=str(self.raw), finalize_output=str(self.output), data=[str(self.pool_file)],
            shard_id=0, num_shards=2, num_candidates=8, seed=20261020,
            model=str(self.root / "unused-tokenizer"), max_model_len=12288,
        )
        verifier_patch = patch.object(bolt_complete, "verify_bolt_splits", return_value={"status": "passed"})
        self.verifier = verifier_patch.start()
        self.addCleanup(verifier_patch.stop)
        print_patch = patch("builtins.print")
        print_patch.start()
        self.addCleanup(print_patch.stop)

    def shard_path(self, index):
        return self.raw / f"shard_{index:05d}.jsonl"

    def write_shard(self, index, fingerprint="test-run"):
        groups = [
            {**row, "enable_thinking": True, "run_fingerprint": fingerprint, "candidates": [
                {"candidate_index": candidate, "content": f"Answer {candidate}", "reasoning_content": "Reasoning",
                 "reward": float(candidate % 2), "finish_reason": "stop", "generated_tokens": candidate + 1,
                 "text": f"Reasoning</think>Answer {candidate}"}
                for candidate in range(8)
            ]}
            for row in self.pool[index::2]
        ]
        path = self.shard_path(index)
        path.write_text("".join(json.dumps(group) + "\n" for group in groups))
        path.with_suffix(".manifest.json").write_text(json.dumps({"run_fingerprint": fingerprint, "shard_id": index}))

    def complete(self, index):
        self.args.shard_id = index
        bolt_complete.complete_shard(self.args)

    def test_export_waits_for_all_receipts_and_keeps_whole_prompt_groups(self):
        self.complete(0)
        receipt = json.loads(self.shard_path(0).with_suffix(".done.json").read_text())
        self.assertEqual(receipt, {"run_fingerprint": "test-run", "sha256": sha256(self.shard_path(0)), "prompts": 5})
        self.assertFalse(self.output.exists())
        self.verifier.assert_not_called()
        self.complete(1)
        train = pq.read_table(self.output / "train.parquet", use_threads=False).to_pylist()
        validation = pq.read_table(self.output / "val.parquet", use_threads=False).to_pylist()
        self.assertEqual((len(train), len(validation)), (72, 8))
        train_ids = {row["prompt_id"] for row in train}
        val_ids = {row["prompt_id"] for row in validation}
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(train_ids | val_ids, {row["prompt_id"] for row in self.pool})
        for rows in (train, validation):
            for prompt_id in {row["prompt_id"] for row in rows}:
                self.assertEqual(sum(row["prompt_id"] == prompt_id for row in rows), 8)
        manifest = json.loads((self.output / "split_manifest.json").read_text())
        self.assertEqual(manifest["counts"], {
            "total_prompts": 10, "train_prompts": 9, "validation_prompts": 1,
            "total_rows": 80, "train_rows": 72, "validation_rows": 8,
        })
        self.assertEqual(manifest["run_fingerprint"], "test-run")
        self.verifier.assert_called_once_with(str(self.output), self.args.model, max_length=12288)

    def test_successful_completion_can_be_resumed_without_rewriting_exports(self):
        self.complete(0)
        self.complete(1)
        originals = {
            name: (self.output / name).read_bytes()
            for name in ("train.parquet", "val.parquet", "split_manifest.json")
        }
        self.complete(0)
        self.complete(1)
        for name, original in originals.items():
            self.assertEqual((self.output / name).read_bytes(), original)
        self.assertEqual(self.verifier.call_count, 3)

    def test_raw_changes_after_receipt_prevent_export(self):
        self.complete(0)
        with self.shard_path(0).open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "completion receipt does not match"):
            self.complete(1)
        self.verifier.assert_not_called()
        self.assertFalse((self.output / "train.parquet").exists())
        self.assertFalse((self.output / "val.parquet").exists())
        self.assertFalse((self.output / "split_manifest.json").exists())

    def test_mixed_run_fingerprints_prevent_export(self):
        self.complete(0)
        self.write_shard(1, fingerprint="other-run")
        with self.assertRaisesRegex(ValueError, "completion receipt does not match"):
            self.complete(1)
        self.verifier.assert_not_called()
        self.assertFalse((self.output / "train.parquet").exists())
        self.assertFalse((self.output / "val.parquet").exists())
        self.assertFalse((self.output / "split_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
