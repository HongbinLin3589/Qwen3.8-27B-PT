"""Verify BOLT smoke scheduler and checkpoint-retention acceptance on CPU."""

import json

from omegaconf import OmegaConf
import pytest
import torch

from qwenpt.train.bolt.train_verify import verify_checkpoints, verify_metrics, verify_schedule


@pytest.fixture
def schedule_run(tmp_path):
    """Record LRs produced by the installed native scheduler after optimizer updates."""
    from verl.utils.torch_functional import get_cosine_schedule_with_warmup

    config = OmegaConf.create({
        "trainer": {"total_training_steps": 20},
        "optim": {"lr_scheduler_type": "cosine", "lr": 5e-6, "lr_warmup_steps": -1,
                  "lr_warmup_steps_ratio": 0.1, "min_lr_ratio": 0.0,
                  "num_cycles": 0.5, "zero_indexed_step": True},
    })
    config_path = tmp_path / "hydra/.hydra/config.yaml"
    config_path.parent.mkdir(parents=True)
    OmegaConf.save(config, config_path)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=5e-6)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=20)
    records = []
    for step in range(1, 21):
        optimizer.step()
        scheduler.step()
        records.append({"step": step, "data": {
            "train/loss": 2.0, "train/grad_norm": 0.5, "train/lr": scheduler.get_last_lr()[0],
        }})
        if step in (10, 20):
            records.append({"step": step, "data": {"val/loss": 1.5}})
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    return tmp_path


def test_schedule_matches_native_warmup_cosine_and_post_update_logging(schedule_run):
    """Check two-step warmup, subsequent decay and the zero final logged LR."""
    metrics = verify_metrics(schedule_run / "metrics.jsonl", 20)
    report = verify_schedule(schedule_run, metrics, 20)
    assert report["warmup_steps"] == 2
    assert report["first_update_lr"] == 0
    learning_rates = report["expected_logged_lrs"]
    assert learning_rates[1] == pytest.approx(2.5e-6)
    assert learning_rates[2] == pytest.approx(5e-6)
    assert learning_rates[3] < learning_rates[2]
    assert learning_rates[19] > 0
    assert learning_rates[20] == 0


@pytest.mark.parametrize("step,wrong_lr", [(1, 5e-6), (2, 2.5e-6), (3, 5e-6), (20, 1e-9)])
def test_schedule_rejects_incorrect_learning_rates(schedule_run, step, wrong_lr):
    metrics = verify_metrics(schedule_run / "metrics.jsonl", 20)
    metrics["training"][step]["train/lr"] = wrong_lr
    with pytest.raises(ValueError, match=f"Incorrect logged LR at step {step}"):
        verify_schedule(schedule_run, metrics, 20)


def test_schedule_rejects_wrong_configured_total(schedule_run):
    path = schedule_run / "hydra/.hydra/config.yaml"
    config = OmegaConf.load(path)
    config.trainer.total_training_steps = 675
    OmegaConf.save(config, path)
    with pytest.raises(ValueError, match="Configured training steps differ"):
        verify_schedule(schedule_run, verify_metrics(schedule_run / "metrics.jsonl", 20), 20)


@pytest.fixture
def retained_run(tmp_path):
    """Create six complete native Torch archive layouts without a distributed model."""
    for step in (10, 12, 14, 16, 18, 20):
        checkpoint = tmp_path / f"global_step_{step}"
        (checkpoint / "huggingface").mkdir(parents=True)
        (checkpoint / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 2, "world_size": 2}))
        (checkpoint / "lora_train_meta.json").write_text(json.dumps({"r": 8, "lora_alpha": 16}))
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (checkpoint / "huggingface" / name).write_text('{"present":true}')
        for rank in range(2):
            for kind in ("model", "optim", "extra_state"):
                torch.save({"tensor": torch.ones(1)}, checkpoint / f"{kind}_world_size_2_rank_{rank}.pt")
            torch.save({"_num_yielded": step}, checkpoint / f"data_{rank}.pt")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20")
    return tmp_path


def test_exact_retained_checkpoints_are_complete_and_crc_readable(retained_run):
    report = verify_checkpoints(retained_run, [10, 12, 14, 16, 18, 20], 20, 2)
    assert report["steps"] == [10, 12, 14, 16, 18, 20]
    assert report["tracker"] == 20
    assert all(len(checkpoint["file_bytes"]) == 8 and checkpoint["zip_crc"] == "passed"
               for checkpoint in report["checkpoints"].values())


@pytest.mark.parametrize("extra", [False, True])
def test_missing_or_extra_checkpoint_directory_is_rejected(retained_run, extra):
    if extra:
        (retained_run / "global_step_8").mkdir()
    else:
        (retained_run / "global_step_10").rename(retained_run / "removed_10")
    with pytest.raises(ValueError, match="Retained checkpoints differ"):
        verify_checkpoints(retained_run, [10, 12, 14, 16, 18, 20], 20, 2)


def test_missing_rank_artifact_is_rejected(retained_run):
    (retained_run / "global_step_14/optim_world_size_2_rank_1.pt").unlink()
    with pytest.raises(ValueError, match="Missing or empty checkpoint artifact"):
        verify_checkpoints(retained_run, [10, 12, 14, 16, 18, 20], 20, 2)


def test_incomplete_torch_archive_is_rejected(retained_run):
    path = retained_run / "global_step_16/model_world_size_2_rank_0.pt"
    path.write_bytes(path.read_bytes()[:64])
    import zipfile

    with pytest.raises(zipfile.BadZipFile):
        verify_checkpoints(retained_run, [10, 12, 14, 16, 18, 20], 20, 2)


def test_stale_tracker_is_rejected(retained_run):
    (retained_run / "latest_checkpointed_iteration.txt").write_text("18")
    with pytest.raises(ValueError, match="Checkpoint tracker is 18"):
        verify_checkpoints(retained_run, [10, 12, 14, 16, 18, 20], 20, 2)
