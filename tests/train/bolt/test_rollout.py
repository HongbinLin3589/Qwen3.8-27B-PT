"""Exercise durable rollout recovery and CPU-only generation orchestration."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from qwenpt.train.bolt import rollout as bolt_rollout


class TestCompletedGroups(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "shard.jsonl"

    def test_missing_file_returns_no_completed_groups(self):
        self.assertEqual(bolt_rollout.read_completed_groups(self.path), [])
        self.assertFalse(self.path.exists())

    def test_valid_records_are_not_rewritten(self):
        original = b'{ "prompt_id": "first", "candidates": [] }\n{"prompt_id":"second"}\n'
        self.path.write_bytes(original)
        self.assertEqual(bolt_rollout.read_completed_groups(self.path), [
            {"prompt_id": "first", "candidates": []}, {"prompt_id": "second"},
        ])
        self.assertEqual(self.path.read_bytes(), original)

    def test_partial_last_record_is_removed_without_losing_completed_groups(self):
        durable = b'{"prompt_id":"first"}\n{"prompt_id":"second"}\n'
        self.path.write_bytes(durable + b'{"prompt_id":"interrupted","candidates":[')
        self.assertEqual(bolt_rollout.read_completed_groups(self.path), [
            {"prompt_id": "first"}, {"prompt_id": "second"},
        ])
        self.assertEqual(self.path.read_bytes(), durable)
        self.assertEqual(bolt_rollout.read_completed_groups(self.path), [
            {"prompt_id": "first"}, {"prompt_id": "second"},
        ])

    def test_valid_last_record_without_newline_remains_appendable(self):
        self.path.write_bytes(b'{"prompt_id":"first"}')
        self.assertEqual(bolt_rollout.read_completed_groups(self.path), [{"prompt_id": "first"}])
        self.assertEqual(self.path.read_bytes(), b'{"prompt_id":"first"}\n')
        with self.path.open("ab") as stream:
            stream.write(b'{"prompt_id":"second"}\n')
        self.assertEqual(bolt_rollout.read_completed_groups(self.path), [
            {"prompt_id": "first"}, {"prompt_id": "second"},
        ])

    def test_partial_utf8_at_end_is_recovered(self):
        durable = b'{"prompt_id":"first"}\n'
        self.path.write_bytes(durable + b'{"text":"' + "答".encode()[:2])
        self.assertEqual(bolt_rollout.read_completed_groups(self.path), [{"prompt_id": "first"}])
        self.assertEqual(self.path.read_bytes(), durable)

    def test_corrupt_middle_record_is_rejected_without_truncation(self):
        original = b'{"prompt_id":"first"}\n{broken json}\n{"prompt_id":"last"}\n'
        self.path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "Corrupt group before end"):
            bolt_rollout.read_completed_groups(self.path)
        self.assertEqual(self.path.read_bytes(), original)


class TestRolloutOrchestration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pool = self.root / "pool.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"messages": [{"role": "user", "content": f"Problem {index}"}], "ground_truth": "1"}
            for index in range(10)
        ]), self.pool)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"fake"}\n')
        (self.model / "generation_config.json").write_text('{"eos_token_id":[3,4]}\n')
        self.output = self.root / "rollouts"
        self.tokenizer = SimpleNamespace(chat_template="fake template", apply_chat_template=Mock(return_value=[11, 12]))
        self.args = argparse.Namespace(
            model=str(self.model), data=[str(self.pool)], num_shards=2, shard_id=0,
            limit=-1, num_candidates=8, seed=20261020, max_new_tokens=8192,
            max_model_len=12288, temperature=1.0, top_p=0.95, top_k=20,
            max_num_seqs=16, max_num_batched_tokens=8192, batch_prompts=4,
            gpu_memory_utilization=0.92, enforce_eager=False,
        )

    @staticmethod
    def fingerprint(config):
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def test_fingerprint_is_shared_by_shards_and_changes_with_protocol(self):
        with patch.object(bolt_rollout.importlib.metadata, "version", return_value="test-version"):
            first = bolt_rollout.generation_config(self.args, self.tokenizer)
            second_args = copy.copy(self.args)
            second_args.shard_id = 1
            second = bolt_rollout.generation_config(second_args, self.tokenizer)
            self.assertEqual(self.fingerprint(first), self.fingerprint(second))
            for key, value in (("seed", 1), ("num_candidates", 4), ("max_new_tokens", 4096)):
                changed = copy.copy(self.args)
                setattr(changed, key, value)
                with self.subTest(key=key):
                    self.assertNotEqual(self.fingerprint(first), self.fingerprint(
                        bolt_rollout.generation_config(changed, self.tokenizer)))
            self.tokenizer.chat_template = "changed template"
            self.assertNotEqual(self.fingerprint(first), self.fingerprint(
                bolt_rollout.generation_config(self.args, self.tokenizer)))

    def test_cli_generates_eight_candidates_and_resume_does_not_generate_again(self):
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=self.tokenizer))
        fake_vllm = ModuleType("vllm")
        fake_vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        engine = Mock()

        def generate(prompts, sampling_params, use_tqdm):
            self.assertFalse(use_tqdm)
            self.assertEqual([item["prompt_token_ids"] for item in prompts], [[11, 12]] * len(prompts))
            for parameter in sampling_params:
                self.assertEqual(parameter.n, 8)
                self.assertEqual(parameter.max_tokens, 8192)
                self.assertEqual(parameter.stop_token_ids, [3, 4])
            return [SimpleNamespace(outputs=[
                SimpleNamespace(
                    index=index,
                    text=(f"Reasoning {index}</think>FINAL-{index % 2}" if index != 7 else "Unfinished reasoning"),
                    finish_reason="length" if index == 7 else "stop", token_ids=list(range(index + 1)),
                )
                for index in reversed(range(8))
            ]) for _ in prompts]

        engine.generate.side_effect = generate
        fake_vllm.LLM = Mock(return_value=engine)
        argv = [
            "bolt_rollout", "generate", "--data", str(self.pool), "--model", str(self.model),
            "--output", str(self.output), "--num-shards", "2", "--shard-id", "0", "--batch-prompts", "2",
        ]
        with (
            patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}),
            patch.object(bolt_rollout.importlib.metadata, "version", return_value="test-version"),
            patch.object(bolt_rollout, "parse_reference", return_value=["1"]),
            patch.object(bolt_rollout, "score_math", side_effect=lambda content, reference: float(content == "FINAL-1")) as grader,
            patch.object(sys, "argv", argv),
            patch("builtins.print"),
        ):
            bolt_rollout.main()
            shard = self.output / "shard_00000.jsonl"
            original = shard.read_bytes()
            groups = bolt_rollout.read_completed_groups(shard)
            self.assertEqual([group["source_index"] for group in groups], [0, 2, 4, 6, 8])
            for group in groups:
                self.assertEqual([item["candidate_index"] for item in group["candidates"]], list(range(8)))
                self.assertEqual([item["reward"] for item in group["candidates"]], [0., 1., 0., 1., 0., 1., 0., 0.])
                self.assertEqual(group["candidates"][-1]["content"], "")
                self.assertEqual(group["candidates"][-1]["reasoning_content"], "Unfinished reasoning")
                self.assertEqual(group["candidates"][-1]["finish_reason"], "length")
            self.assertEqual(grader.call_count, 40)
            self.assertTrue(all("Reasoning" not in call.args[0] for call in grader.call_args_list))
            self.assertEqual(self.tokenizer.apply_chat_template.call_count, 5)
            for call in self.tokenizer.apply_chat_template.call_args_list:
                self.assertIs(call.kwargs["return_dict"], False)
                self.assertIs(call.kwargs["enable_thinking"], True)
            self.assertEqual(engine.generate.call_count, 3)
            self.assertEqual([parameter.seed for call in engine.generate.call_args_list
                              for parameter in call.kwargs["sampling_params"]],
                             [20261020 + index for index in (0, 2, 4, 6, 8)])
            bolt_rollout.main()
            self.assertEqual(shard.read_bytes(), original)
            self.assertEqual(engine.generate.call_count, 3)
            self.assertEqual(fake_vllm.LLM.call_count, 1)
            engine.llm_engine.engine_core.shutdown.assert_called_once_with(timeout=30)
            self.assertEqual(self.tokenizer.apply_chat_template.call_count, 5)
            with shard.open("ab") as destination:
                destination.write(b'{"prompt_id":"interrupted"')
            bolt_rollout.main()
            self.assertEqual(shard.read_bytes(), original)
            self.assertEqual(engine.generate.call_count, 3)
            self.assertEqual(fake_vllm.LLM.call_count, 1)
            engine.llm_engine.engine_core.shutdown.assert_called_once_with(timeout=30)

    def test_generation_failure_closes_engine_and_preserves_original_exception(self):
        for shutdown_failure in (False, True):
            with self.subTest(shutdown_failure=shutdown_failure):
                fake_transformers = ModuleType("transformers")
                fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=self.tokenizer))
                fake_vllm = ModuleType("vllm")
                fake_vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
                engine = Mock()
                failure = RuntimeError("generation failed")
                engine.generate.side_effect = failure
                if shutdown_failure:
                    engine.llm_engine.engine_core.shutdown.side_effect = RuntimeError("shutdown failed")
                fake_vllm.LLM = Mock(return_value=engine)
                argv = [
                    "bolt_rollout", "generate", "--data", str(self.pool), "--model", str(self.model),
                    "--output", str(self.output / str(shutdown_failure)),
                    "--num-shards", "2", "--shard-id", "0",
                ]
                with (
                    patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}),
                    patch.object(bolt_rollout.importlib.metadata, "version", return_value="test-version"),
                    patch.object(bolt_rollout, "parse_reference", return_value=["1"]),
                    patch.object(sys, "argv", argv),
                    patch("builtins.print"),
                    self.assertRaises(RuntimeError) as raised,
                ):
                    bolt_rollout.main()
                self.assertIs(raised.exception, failure)
                engine.llm_engine.engine_core.shutdown.assert_called_once_with(timeout=30)
                if shutdown_failure:
                    self.assertIn("shutdown failed", " ".join(failure.__notes__))


if __name__ == "__main__":
    unittest.main()
