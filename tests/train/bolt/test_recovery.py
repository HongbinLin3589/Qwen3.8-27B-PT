"""Validate execution-route recovery without rewriting generation provenance."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from qwenpt.train.bolt import recovery as bolt_recovery
from qwenpt.train.bolt.data import load_math_pool
from qwenpt.train.common.files import sha256


class TestExecutionFamily(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pool_path = self.root / "pool.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"messages": [{"role": "user", "content": f"Problem {index}"}], "ground_truth": "1"}
            for index in range(10)
        ]), self.pool_path)
        self.pool = load_math_pool([str(self.pool_path)])
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"fake"}\n')
        source = Path(bolt_recovery.__file__).parent
        self.config = {
            "format_version": 1, "inputs": [{"path": str(self.pool_path), "sha256": sha256(self.pool_path)}],
            "model": str(self.model), "model_metadata_sha256": {"config.json": sha256(self.model / "config.json")},
            "code_sha256": {name: sha256(source / name.removeprefix("bolt_")) for name in ("bolt_rollout.py", "bolt_data.py", "bolt_math.py")},
            "runtime": {"vllm": "test", "torch": "test"}, "limit": -1, "num_candidates": 8,
            "seed": 20261020, "max_new_tokens": 8192, "max_model_len": 12288,
            "temperature": 1.0, "top_p": 0.95, "engine": {"batch_prompts": 4},
        }
        self.shards = [self.write_shard(2, 0), self.write_shard(4, 1), self.write_shard(4, 3)]
        self.output = self.root / "export"

    def write_shard(self, count, index, overrides=None):
        directory = self.root / f"route-{count}"
        directory.mkdir(exist_ok=True)
        path = directory / f"shard_{index:05d}.jsonl"
        config = {**copy.deepcopy(self.config), "num_shards": count, **(overrides or {})}
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        groups = [
            {**row, "run_fingerprint": fingerprint, "enable_thinking": True, "candidates": [
                {"candidate_index": candidate, "content": str(candidate), "reasoning_content": "Reasoning",
                 "reward": float(candidate % 2), "finish_reason": "stop", "generated_tokens": candidate + 1}
                for candidate in range(8)
            ]}
            for row in self.pool[index::count]
        ]
        path.write_text("".join(json.dumps(group) + "\n" for group in groups))
        path.with_suffix(".manifest.json").write_text(json.dumps({
            "config": config, "run_fingerprint": fingerprint, "shard_id": index,
        }))
        return path

    def finalize(self, **kwargs):
        return bolt_recovery.finalize_execution_family(
            [str(self.pool_path)], [str(path) for path in self.shards], str(self.output), **kwargs,
        )

    def test_same_protocol_different_routes_exports_complete_groups_and_preserves_provenance(self):
        original = {path: path.read_bytes() for path in self.shards}
        manifest = self.finalize()
        self.assertEqual(manifest["counts"]["train_rows"], 72)
        self.assertEqual(manifest["counts"]["validation_rows"], 8)
        self.assertEqual(len(manifest["generation_configs"]), 2)
        for path, raw in original.items():
            self.assertEqual(path.read_bytes(), raw)
            fingerprint = json.loads(path.with_suffix(".manifest.json").read_text())["run_fingerprint"]
            self.assertTrue(all(json.loads(line)["run_fingerprint"] == fingerprint for line in raw.splitlines()))
        train = pq.read_table(self.output / "train.parquet", use_threads=False).to_pylist()
        validation = pq.read_table(self.output / "val.parquet", use_threads=False).to_pylist()
        self.assertFalse({row["prompt_id"] for row in train} & {row["prompt_id"] for row in validation})
        for rows in (train, validation):
            for prompt_id in {row["prompt_id"] for row in rows}:
                self.assertEqual(sum(row["prompt_id"] == prompt_id for row in rows), 8)
        self.assertEqual(manifest["protocol_sha256"], hashlib.sha256(
            json.dumps(self.config, sort_keys=True).encode()).hexdigest())
        self.assertEqual({item["sha256"] for item in manifest["shards"]}, {sha256(path) for path in self.shards})

    def test_seed_sampling_runtime_or_code_changes_rejected(self):
        for overrides in ({"seed": 7}, {"temperature": 0.5}, {"runtime": {"vllm": "different"}},
                          {"code_sha256": {"bolt_rollout.py": "different"}}, {"temperature": True}):
            with self.subTest(overrides=overrides):
                self.write_shard(4, 3, overrides)
                with self.assertRaisesRegex(ValueError, "protocols differ|code hashes differ"):
                    self.finalize()
        self.write_shard(4, 3)

    def test_stale_current_code_pool_or_model_bindings_rejected(self):
        original_model = (self.model / "config.json").read_text()
        (self.model / "config.json").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "model metadata hashes differ"):
            self.finalize()
        (self.model / "config.json").write_text(original_model)
        pq.write_table(pa.Table.from_pylist([{**self.pool[0], "ground_truth": "2"}]), self.pool_path)
        with self.assertRaisesRegex(ValueError, "input or code hashes differ"):
            self.finalize()

    def test_uniform_stale_code_is_rejected_even_when_all_protocols_agree(self):
        changed = dict(self.config["code_sha256"])
        changed["bolt_rollout.py"] = "outdated"
        for count, index in ((2, 0), (4, 1), (4, 3)):
            self.write_shard(count, index, {"code_sha256": changed})
        with self.assertRaisesRegex(ValueError, "input or code hashes differ"):
            self.finalize()

    def test_changed_chat_template_is_rejected(self):
        template = self.model / "chat_template.jinja"
        template.write_text("original template")
        template_hash = sha256(template)
        for count, index in ((2, 0), (4, 1), (4, 3)):
            self.write_shard(count, index, {"chat_template_sha256": template_hash})
        template.write_text("changed template")
        with self.assertRaisesRegex(ValueError, "chat template hash differs"):
            self.finalize()

    def test_wrong_routing_duplicate_and_missing_groups_rejected(self):
        original = self.shards[0].read_text()
        foreign = self.shards[1].read_text().splitlines()[0]
        for raw, error in ((foreign + "\n" + original, "wrong shard routing"),
                           (original + original.splitlines()[0] + "\n", "Duplicate prompt group"),
                           ("\n".join(original.splitlines()[1:]) + "\n", "Incomplete assigned shard")):
            with self.subTest(error=error):
                self.shards[0].write_text(raw)
                with self.assertRaisesRegex(ValueError, error):
                    self.finalize()
        self.shards[0].write_text(original)
        self.shards.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete prompt coverage"):
            self.finalize()

    def test_invalid_group_or_original_fingerprint_rejected(self):
        path = self.shards[0]
        original = path.read_text()
        groups = [json.loads(line) for line in original.splitlines()]
        groups[0]["candidates"].pop()
        path.write_text("".join(json.dumps(group) + "\n" for group in groups))
        with self.assertRaisesRegex(ValueError, "expected 8 candidates"):
            self.finalize()
        path.write_text(original)
        groups[0]["run_fingerprint"] = "forged"
        path.write_text("".join(json.dumps(group) + "\n" for group in groups))
        with self.assertRaisesRegex(ValueError, "Group fingerprint differs"):
            self.finalize()
        path.write_text(original)
        metadata_path = path.with_suffix(".manifest.json")
        metadata = json.loads(metadata_path.read_text())
        metadata["config"]["seed"] = 9
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Generation fingerprint does not match"):
            self.finalize()

    def test_reuse_requires_matching_sources_parameters_and_output(self):
        manifest = self.finalize()
        with self.assertRaises(FileExistsError):
            self.finalize()
        self.assertEqual(self.finalize(reuse_existing=True), manifest)
        with self.assertRaisesRegex(ValueError, "inputs or parameters differ"):
            self.finalize(reuse_existing=True, seed=1)
        self.shards[0].write_text(self.shards[0].read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "inputs or parameters differ"):
            self.finalize(reuse_existing=True)
        self.shards[0].write_text(self.shards[0].read_text()[:-1])
        with (self.output / "train.parquet").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "output hashes differ"):
            self.finalize(reuse_existing=True)
        (self.output / "train.parquet").unlink()
        with self.assertRaises(FileExistsError):
            self.finalize(reuse_existing=True)

    def test_cli_verifies_with_project_training_python(self):
        with patch.object(sys, "argv", [
            "bolt_recovery", "--data", str(self.pool_path), "--shards", *(str(path) for path in self.shards),
            "--output", str(self.output),
        ]), patch.object(bolt_recovery.subprocess, "run") as run, patch("builtins.print"):
            bolt_recovery.main()
        command = run.call_args.args[0]
        self.assertEqual(command[0], str(Path(bolt_recovery.__file__).resolve().parents[4] / ".venv/bin/python"))
        self.assertEqual(command[1:3], ["-m", "qwenpt.train.bolt.verify"])
        self.assertIn(str(self.model), command)
        self.assertIn("12288", command)
        self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
