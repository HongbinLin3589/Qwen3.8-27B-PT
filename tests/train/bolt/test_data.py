"""Offline BOLT export tests cover completeness, provenance, and group isolation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from qwenpt.train.bolt.data import MATH_PROMPT_SUFFIX, finalize_bolt_data, load_math_pool


class TestBOLTData(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pool_file = self.root / "pool.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"messages": [{"role": "user", "content": f"Compute {index} + 1"}], "ground_truth": str(index + 1)}
            for index in range(10)
        ]), self.pool_file)
        self.pool = load_math_pool([str(self.pool_file)])
        self.groups = [
            {**row, "enable_thinking": True, "run_fingerprint": "test-run", "candidates": [
                {"candidate_index": index, "content": str(index), "reasoning_content": "Reasoning",
                 "reward": float(index % 2), "finish_reason": "length" if index == 7 else "stop",
                 "generated_tokens": index + 1, "text": f"Reasoning {index}"}
                for index in range(8)
            ]}
            for row in self.pool
        ]
        self.shard = self.root / "shard.jsonl"
        self.output = self.root / "output"
        self.write_groups()

    def write_groups(self, groups=None):
        self.shard.write_text("".join(json.dumps(group) + "\n" for group in (self.groups if groups is None else groups)))

    def finalize(self, **kwargs):
        return finalize_bolt_data([str(self.pool_file)], [str(self.shard)], str(self.output), **kwargs)

    def test_complete_group_split_preserves_all_candidates(self):
        self.groups[0]["candidates"][0]["content"] = ""
        self.write_groups()
        manifest = self.finalize()
        train = pq.read_table(self.output / "train.parquet", use_threads=False).to_pylist()
        validation = pq.read_table(self.output / "val.parquet", use_threads=False).to_pylist()
        self.assertEqual(manifest["counts"], {
            "total_prompts": 10, "train_prompts": 9, "validation_prompts": 1,
            "total_rows": 80, "train_rows": 72, "validation_rows": 8,
        })
        train_ids = {row["prompt_id"] for row in train}
        val_ids = {row["prompt_id"] for row in validation}
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(train_ids | val_ids, {row["prompt_id"] for row in self.pool})
        for rows in (train, validation):
            for prompt_id in {row["prompt_id"] for row in rows}:
                candidates = [row for row in rows if row["prompt_id"] == prompt_id]
                self.assertEqual([row["sample_id"] for row in candidates], [f"{prompt_id}-c{index}" for index in range(8)])
                self.assertEqual([row["reward"] for row in candidates], [float(index % 2) for index in range(8)])
                self.assertEqual(candidates[-1]["finish_reason"], "length")
        self.assertTrue(all(row["messages"][-1]["role"] == "assistant" for row in train + validation))
        self.assertEqual({row["messages"][-1]["reasoning_content"] for row in train + validation}, {"Reasoning"})

    def test_missing_group_rejected(self):
        self.write_groups(self.groups[:-1])
        with self.assertRaisesRegex(ValueError, "Missing prompt groups"):
            self.finalize()
        self.assertFalse(self.output.exists())

    def test_duplicate_group_rejected(self):
        self.write_groups(self.groups + [self.groups[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate prompt group"):
            self.finalize()

    def test_incomplete_or_duplicate_candidates_rejected(self):
        for mode in ("missing", "duplicate"):
            groups = copy.deepcopy(self.groups)
            if mode == "missing":
                groups[0]["candidates"].pop()
            else:
                groups[0]["candidates"][-1]["candidate_index"] = 0
            self.write_groups(groups)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "expected 8 candidates|duplicate candidate_index"):
                self.finalize()

    def test_pool_provenance_must_match(self):
        for key, value in (("messages", [{"role": "user", "content": "different"}]),
                           ("ground_truth", "different"), ("source_index", 99), ("prompt_id", "unknown")):
            groups = copy.deepcopy(self.groups)
            groups[0][key] = value
            self.write_groups(groups)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "does not match|Unknown prompt"):
                self.finalize()

    def test_invalid_supervision_or_reward_rejected(self):
        for update in ({"reward": float("nan")}, {"reward": float("inf")}, {"reward": True},
                       {"content": "", "reasoning_content": ""}, {"generated_tokens": 0}):
            groups = copy.deepcopy(self.groups)
            groups[0]["candidates"][0].update(update)
            self.write_groups(groups)
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.finalize()

    def test_candidate_order_normalized(self):
        for group in self.groups:
            group["candidates"].reverse()
        self.write_groups()
        self.finalize()
        rows = pq.read_table(self.output / "val.parquet", use_threads=False).to_pylist()
        self.assertEqual([row["generated_tokens"] for row in rows], list(range(1, 9)))

    def test_mixed_or_missing_run_fingerprint_rejected(self):
        for value in ("different-run", "", None):
            groups = copy.deepcopy(self.groups)
            groups[-1]["run_fingerprint"] = value
            self.write_groups(groups)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "run_fingerprint"):
                self.finalize()

    def test_reuse_requires_matching_input_parameters_and_output(self):
        manifest = self.finalize()
        with self.assertRaises(FileExistsError):
            self.finalize()
        self.assertEqual(self.finalize(reuse_existing=True), manifest)
        with self.assertRaisesRegex(ValueError, "inputs or parameters differ"):
            self.finalize(reuse_existing=True, seed=1)
        self.shard.write_text(self.shard.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "inputs or parameters differ"):
            self.finalize(reuse_existing=True)
        self.write_groups()
        with (self.output / "train.parquet").open("ab") as destination:
            destination.write(b"changed")
        with self.assertRaisesRegex(ValueError, "output hashes differ"):
            self.finalize(reuse_existing=True)

    def test_split_is_reproducible(self):
        first = self.finalize()
        self.output = self.root / "repeat"
        second = self.finalize()
        self.assertEqual(first, second)

    def test_pool_identity_and_validation(self):
        repeated = load_math_pool([str(self.pool_file), str(self.pool_file)])
        self.assertEqual(repeated, self.pool)
        self.assertEqual([row["source_index"] for row in self.pool], list(range(10)))
        self.assertEqual(len({row["prompt_id"] for row in self.pool}), 10)
        pq.write_table(pa.Table.from_pylist([self.pool[0], self.pool[0]]), self.pool_file)
        with self.assertRaisesRegex(ValueError, "Duplicate prompt"):
            load_math_pool([str(self.pool_file)])

    def test_apex_answer_instruction_is_preserved_without_modifying_source(self):
        before = self.pool_file.read_bytes()
        prepared = load_math_pool([str(self.pool_file)])
        self.assertTrue(prepared[0]["messages"][-1]["content"].endswith(MATH_PROMPT_SUFFIX))
        self.assertEqual(before, self.pool_file.read_bytes())
        prepared_file = self.root / "prepared.parquet"
        pq.write_table(pa.Table.from_pylist(prepared), prepared_file)
        self.assertEqual(load_math_pool([str(prepared_file)]), prepared)


if __name__ == "__main__":
    unittest.main()
