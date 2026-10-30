"""Verify source-index evidence, excluded padding, resume attempts and failures."""

import hashlib
import json
import struct
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from qwenpt.train.sft.progress import SFTProgressRecorder, audit_sft_progress
from qwenpt.train.sft.sampler import build_length_bucket_dataloader


class _Dataset:
    def __init__(self):
        self.rows = [{"sample_id": str(index), "split": "train"} for index in range(4)]
        self.token_lengths = [3, 4, 5, 6]
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        length = self.token_lengths[index]
        return {"input_ids": torch.full((length,), index + 1, dtype=torch.long),
                "position_ids": torch.arange(length), "loss_mask": torch.tensor([0] + [1] * (length - 1))}


def _trainer(tmp_path, *, rank=0, resume=0):
    from verl.utils.dataset.dataset_utils import SFTTensorCollator

    config = OmegaConf.create({
        "trainer": {"default_local_dir": str(tmp_path), "resume_from_path": None},
        "data": {"train_files": ["frozen-train.parquet"], "train_batch_size": 6,
                 "micro_batch_size_per_gpu": 1, "num_workers": 0, "use_dynamic_bsz": False,
                 "length_bucket_drop_last": False, "length_bucket_pad_tail": True},
    })
    dataset = _Dataset()
    sampler, loader = build_length_bucket_dataloader(
        dataset, config.data, collate_fn=SFTTensorCollator("no_padding"),
        num_replicas=6, rank=rank, seed=7, track_source_indices=True,
    )
    return SimpleNamespace(
        config=config, train_dataset=dataset, train_sampler=sampler, train_dataloader=loader,
        resume_global_step=resume, steps_per_epoch=10, total_training_steps=20,
        engine=SimpleNamespace(get_data_parallel_rank=lambda: rank, get_data_parallel_size=lambda: 6),
    )


def _batch(trainer):
    from verl.utils import tensordict_utils as tu

    return tu.get_tensordict(next(iter(trainer.train_dataloader)))


def _output(loss=0.25):
    from verl.utils import tensordict_utils as tu

    return tu.get_tensordict({}, non_tensor_dict={"metrics": {"loss": loss, "grad_norm": 0.5, "lr": 1e-5}})


def _records(recorder):
    return [json.loads(line) for line in recorder.path.read_text().splitlines()]


def test_all_rank_provenance_covers_real_rows_and_excludes_internal_padding(tmp_path):
    """Raw indices remain unambiguous even when text is repeated or a rank has only padding."""
    completed = []
    for rank in range(6):
        trainer = _trainer(tmp_path, rank=rank)
        recorder = SFTProgressRecorder(trainer, rank=rank, attempt_id="first")
        batch = _batch(trainer)
        recorder.before_batch(batch)
        assert "sft_source_index" not in batch.keys()
        recorder.after_batch(_output(), 2.5)
        recorder.finish("completed")
        records = _records(recorder)
        assert [item["event"] for item in records] == [
            "attempt_started", "batch_started", "step_completed", "attempt_finished",
        ]
        assert records[0]["resume_source"] is None
        completed.append(records[2])
        assert records[2]["global_step"] == 1
        assert records[2]["process_peak_rss_bytes"] > 0
    samples = [sample for record in completed for sample in record["samples"]]
    assert sorted(sample["source_index"] for sample in samples) == list(range(4))
    assert sum(record["real_samples"] for record in completed) == 4
    assert sum(record["real_input_tokens"] for record in completed) == 18
    assert sum(record["real_supervised_tokens"] for record in completed) == 14
    assert sum(record["padding_samples"] for record in completed) == 2
    assert sum(record["padding_input_tokens"] for record in completed) == 32
    for sample in samples:
        row = {"sample_id": str(sample["source_index"]), "split": "train"}
        expected = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert sample["source_fingerprint"] == expected
        length = sample["input_tokens"]
        values = [sample["source_index"] + 1] * length + [0] + [1] * (length - 1)
        assert sample["tokenized_fingerprint"] == hashlib.sha256(struct.pack("=" + "q" * len(values), *values)).hexdigest()


def test_resume_creates_a_new_attempt_and_continues_native_global_steps(tmp_path):
    """Existing evidence remains untouched when training resumes a saved step."""
    initial = SFTProgressRecorder(_trainer(tmp_path), rank=0, attempt_id="before")
    initial.finish("failed")
    saved = initial.path.read_bytes()
    trainer = _trainer(tmp_path, resume=7)
    resumed = SFTProgressRecorder(trainer, rank=0, attempt_id="after")
    resumed.before_batch(_batch(trainer))
    resumed.after_batch(_output(), 1.25)
    resumed.finish("completed")
    records = _records(resumed)
    assert records[0]["resume_source"] == str(tmp_path / "global_step_7")
    assert records[0]["resume_global_step"] == 7
    assert records[2]["global_step"] == 8
    assert records[-1]["last_completed_global_step"] == 8
    assert initial.path.read_bytes() == saved
    with pytest.raises(FileExistsError):
        SFTProgressRecorder(trainer, rank=0, attempt_id="after")


def test_failure_flushes_pending_source_and_traceback_without_cuda_calls(tmp_path, monkeypatch):
    """An OOM leaves a complete attempted batch even if CUDA communication is broken."""
    trainer = _trainer(tmp_path)
    recorder = SFTProgressRecorder(trainer, rank=0)
    recorder.before_batch(_batch(trainer))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: (_ for _ in ()).throw(AssertionError("CUDA touched")))
    try:
        raise RuntimeError("simulated rank failure")
    except RuntimeError as error:
        recorder.fail(error)
    recorder.finish("failed")
    failure = _records(recorder)[-2]
    assert failure["event"] == "step_failed"
    assert failure["global_step"] == 1
    assert failure["samples"] or failure["padding_samples"]
    assert "simulated rank failure" in failure["traceback"]
    assert recorder.completed_steps == 0


def test_nonfinite_step_is_failed_and_never_marked_complete(tmp_path):
    """Native skipped/invalid updates cannot be misreported as successful evidence."""
    trainer = _trainer(tmp_path)
    recorder = SFTProgressRecorder(trainer, rank=0)
    recorder.before_batch(_batch(trainer))
    with pytest.raises(FloatingPointError) as raised:
        recorder.after_batch(_output(float("nan")), 1.0)
    recorder.fail(raised.value)
    recorder.finish("failed")
    assert "step_completed" not in [record["event"] for record in _records(recorder)]
    assert _records(recorder)[-2]["event"] == "step_failed"


def test_duplicate_source_and_changed_audit_fingerprint_are_rejected(tmp_path):
    """Source indexing cannot silently disagree with cached real tokenizer evidence."""
    for rank in range(6):
        trainer = _trainer(tmp_path, rank=rank)
        batch = _batch(trainer)
        source = int(batch["sft_source_index"].values()[0])
        if source >= 0:
            break
    assert source >= 0
    recorder = SFTProgressRecorder(trainer, rank=rank)
    recorder.before_batch(batch)
    recorder.after_batch(_output(), 1)
    with pytest.raises(ValueError, match="repeated"):
        recorder.before_batch(_batch(trainer))
    recorder.finish("failed")
    trainer.train_dataset.rows[source]["tokenized_fingerprint"] = "bad"
    changed = SFTProgressRecorder(trainer, rank=rank)
    with pytest.raises(ValueError, match="token fingerprint"):
        changed.before_batch(_batch(trainer))
    changed.finish("failed")


def _manifest():
    return {
        "status": "passed", "max_samples": -1,
        "counts": {"validated_rows": 4, "eligible_train_rows": 4, "selected_rows": 4, "failures": 0},
        "actual_length_bins": {"short": {"rows": 4, "tokens": 18, "supervised_tokens": 14}},
    }


def _history(tmp_path, rank, attempt, *, resume=0, steps=None, status="completed", started=1):
    if steps is None:
        steps = [(1, rank), (2, rank + 2)]
    records = [{
        "event": "attempt_started", "attempt_id": attempt, "rank": rank, "unix_time": started,
        "source_rows": 4, "data_parallel_size": 2, "data_parallel_rank": rank,
        "sequence_parallel_size": 1, "resume_global_step": resume,
        "sampler_configuration": {"rank": rank, "seed": 7},
        "steps_per_epoch": 2, "total_training_steps": 2,
    }]
    for step, source in steps:
        sample = {
            "source_index": source, "input_tokens": source + 3, "supervised_tokens": source + 2,
            "source_fingerprint": hashlib.sha256(f"source{source}".encode()).hexdigest(),
            "tokenized_fingerprint": hashlib.sha256(f"tokens{source}".encode()).hexdigest(),
        }
        records.append({
            "event": "step_completed", "attempt_id": attempt, "rank": rank, "unix_time": started + step / 10,
            "global_step": step, "samples": [sample], "real_samples": 1,
            "real_input_tokens": sample["input_tokens"], "real_supervised_tokens": sample["supervised_tokens"],
            "padding_samples": 0, "padding_input_tokens": 0,
            "loss": 0.5, "grad_norm": 1.0, "lr": 1e-5, "elapsed_seconds": 2.0,
        })
    if status is not None:
        records.append({
            "event": "attempt_finished", "attempt_id": attempt, "rank": rank, "unix_time": started + 0.5,
            "status": status, "last_completed_global_step": resume + len(steps),
        })
    directory = tmp_path / "progress"
    directory.mkdir(exist_ok=True)
    path = directory / f"{attempt}.rank{rank}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _change_completed(path, mutate):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    record = next(item for item in records if item["event"] == "step_completed")
    mutate(record)
    path.write_text("".join(json.dumps(item) + "\n" for item in records))


def test_final_audit_requires_complete_epoch_exact_source_coverage_and_token_totals(tmp_path):
    """A tracker is insufficient; every effective rank/step and exact real token sum is required."""
    for rank in range(2):
        _history(tmp_path, rank, "only")
    result = audit_sft_progress(tmp_path, _manifest(), world_size=2, expected_rows=4)
    assert result["status"] == "passed"
    assert result["real_samples"] == 4
    assert result["real_input_tokens"] == 18
    assert result["real_supervised_tokens"] == 14
    assert result["completed_global_steps"] == 2
    assert result["checkpoint_archives_verified"] is False


def test_final_audit_discards_uncheckpointed_old_tail_before_correct_resume(tmp_path):
    """A crashed attempt's extra optimizer step must not duplicate the replayed source."""
    old_paths = []
    for rank in range(2):
        old = _history(tmp_path, rank, "crashed", steps=[(1, rank), (2, rank)], status=None)
        with old.open("a") as handle:
            handle.write('{"event":')
        old_paths.append(str(old))
        _history(tmp_path, rank, "resumed", resume=1, steps=[(2, rank + 2)], started=2)
    result = audit_sft_progress(tmp_path, _manifest(), world_size=2)
    assert result["real_samples"] == 4
    assert sorted(result["truncated_old_attempt_files"]) == sorted(old_paths)


@pytest.mark.parametrize("problem", ["duplicate", "missing_source", "missing_step", "missing_rank",
                                     "nonfinite_loss", "nonfinite_elapsed", "not_completed", "wrong_tokens"])
def test_final_audit_rejects_incomplete_or_inconsistent_effective_progress(tmp_path, problem):
    paths = [_history(tmp_path, rank, "only") for rank in range(2)]
    manifest = _manifest()
    if problem == "duplicate":
        _change_completed(paths[1], lambda record: record["samples"][0].update(source_index=0))
    elif problem == "missing_source":
        _change_completed(paths[1], lambda record: record.update(
            samples=[], real_samples=0, real_input_tokens=0, real_supervised_tokens=0,
        ))
    elif problem == "missing_step":
        records = paths[0].read_text().splitlines(keepends=True)
        paths[0].write_text("".join(records[:1] + records[2:]))
    elif problem == "missing_rank":
        paths[1].unlink()
    elif problem == "nonfinite_loss":
        _change_completed(paths[0], lambda record: record.update(loss=float("nan")))
    elif problem == "nonfinite_elapsed":
        _change_completed(paths[0], lambda record: record.update(elapsed_seconds=float("inf")))
    elif problem == "not_completed":
        paths[0].write_text(paths[0].read_text().replace('"status": "completed"', '"status": "failed"'))
    elif problem == "wrong_tokens":
        manifest["actual_length_bins"]["short"]["tokens"] += 1
    with pytest.raises(ValueError):
        audit_sft_progress(tmp_path, manifest, world_size=2)


def test_final_audit_rejects_resume_without_earlier_step_evidence(tmp_path):
    """A checkpoint marker cannot supply missing source-consumption history."""
    for rank in range(2):
        _history(tmp_path, rank, "resumed", resume=1, steps=[(2, rank + 2)])
    with pytest.raises(ValueError, match="prior checkpoint-step"):
        audit_sft_progress(tmp_path, _manifest(), world_size=2)


@pytest.mark.parametrize("expected_rows,status", [(4, "passed"), (100000, "failed")])
def test_progress_audit_cli_writes_a_report_and_returns_failure_for_wrong_full_size(
    tmp_path, monkeypatch, expected_rows, status,
):
    """CLI defaults cannot mistake a tiny smoke pool for the full production dataset."""
    from qwenpt.train.sft.progress import main

    for rank in range(2):
        _history(tmp_path, rank, "only")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()))
    monkeypatch.setattr(sys, "argv", [
        "sft_progress", "--run-dir", str(tmp_path), "--manifest", str(manifest),
        "--world-size", "2", "--expected-rows", str(expected_rows),
    ])
    if status == "failed":
        with pytest.raises(SystemExit, match="1"):
            main()
    else:
        main()
    report = json.loads((tmp_path / "training_progress_audit.json").read_text())
    assert report["status"] == status
    assert report["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
