"""Check the Slurm launcher without loading model weights or launching workers."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import pyarrow as pa
import pyarrow.parquet as pq


def _launch_sft(
    tmp_path, *, slurm=False, wandb_mode=None, exit_code=0, model_source="project",
):
    """Run the real shell launcher against a recorder without starting workers."""
    project = tmp_path / "project with spaces"
    script_dir = project / "scripts/train"
    script_dir.mkdir(parents=True)
    script = script_dir / "sft.sh"
    source = Path(__file__).resolve().parents[3] / "scripts/train/sft.sh"
    shutil.copyfile(source, script)
    (project / "pyproject.toml").touch()
    activate = project / ".venv/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.touch()
    model = project / "models/Qwen3.8-27B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    hub_cache = tmp_path / "HF cache"
    cached_model = hub_cache / "models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    cached_model.mkdir(parents=True)
    (cached_model / "config.json").write_text("{}")
    if model_source == "cache":
        (model / "config.json").unlink()
        model = cached_model
    elif model_source == "explicit":
        model = tmp_path / "explicit model"
        model.mkdir()
        (model / "config.json").write_text("{}")
    data = project / "data/processed/sft/qwen27b-v2"
    (data / "train").mkdir(parents=True)
    (data / "validation").mkdir()
    pq.write_table(pa.Table.from_pylist([
        {"sample_id": f"sample-{index}", "split": "train", "messages": [
            {"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"},
        ]} for index in range(20)
    ]), data / "train/pool.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"sample_id": "validation-only", "messages": []},
    ]), data / "validation/pool.parquet")
    original_train = (data / "train/pool.parquet").read_bytes()
    original_val = (data / "validation/pool.parquet").read_bytes()

    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    captured = tmp_path / "arguments.json"
    torchrun = binary_dir / "torchrun"
    torchrun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "assert 'ROCR_VISIBLE_DEVICES' not in os.environ\n"
        "assert 'HIP_VISIBLE_DEVICES' not in os.environ\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3,4,5,6,7'\n"
        "with open(os.environ['QWENPT_CAPTURE_ARGUMENTS'], 'w') as stream:\n"
        "    json.dump({'arguments': sys.argv[1:], 'pid': os.getpid(), 'wandb_mode': os.environ['WANDB_MODE']}, stream)\n"
        "sys.exit(int(os.environ['QWENPT_CAPTURE_EXIT_CODE']))\n"
    )
    torchrun.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{binary_dir}:{environment['PATH']}"
    environment["QWENPT_CAPTURE_ARGUMENTS"] = str(captured)
    environment["QWENPT_CAPTURE_EXIT_CODE"] = str(exit_code)
    if wandb_mode is None:
        environment.pop("WANDB_MODE", None)
    else:
        environment["WANDB_MODE"] = wandb_mode
    environment["ROCR_VISIBLE_DEVICES"] = "0,1"
    environment["HIP_VISIBLE_DEVICES"] = "0,1"
    environment["CUDA_VISIBLE_DEVICES"] = "0,1"
    for name in ("QWENPT_SFT_MODEL_PATH", "QWENPT_SFT_TRAIN_FILE", "QWENPT_SFT_VAL_FILE", "QWENPT_SFT_MAX_TOKENS"):
        environment.pop(name, None)
    environment["HF_HUB_CACHE"] = str(hub_cache)
    if model_source == "explicit":
        environment["QWENPT_SFT_MODEL_PATH"] = str(model)
    if slurm:
        environment["SLURM_SUBMIT_DIR"] = str(script_dir)
    else:
        environment.pop("SLURM_SUBMIT_DIR", None)

    with subprocess.Popen(
        ["bash", str(script), "trainer.total_training_steps=2"],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        _, stderr = process.communicate()
    assert process.returncode == exit_code, stderr

    capture = json.loads(captured.read_text())
    assert capture["pid"] == process.pid
    assert (data / "train/pool.parquet").read_bytes() == original_train
    assert (data / "validation/pool.parquet").read_bytes() == original_val
    return capture, environment


@pytest.mark.parametrize("slurm", [False, True])
@pytest.mark.parametrize("wandb_mode,exit_code", [(None, 0), ("offline", 17)])
@pytest.mark.parametrize("model_source", ["project", "cache", "explicit"])
def test_launcher_preserves_paths_and_cli_overrides(
    tmp_path: Path, slurm: bool, wandb_mode: str | None, exit_code: int, model_source: str,
) -> None:
    """Preserve quoting, model lookup, environment, exit codes and CLI precedence."""
    capture, _ = _launch_sft(
        tmp_path, slurm=slurm, wandb_mode=wandb_mode, exit_code=exit_code,
        model_source=model_source,
    )
    project = tmp_path / "project with spaces"
    data = project / "data/processed/sft/qwen27b-v2"
    if model_source == "cache":
        model = tmp_path / "HF cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    elif model_source == "explicit":
        model = tmp_path / "explicit model"
    else:
        model = project / "models/Qwen3.8-27B"
    assert capture["wandb_mode"] == (wandb_mode or "online")
    arguments = capture["arguments"]
    assert arguments[arguments.index("--module") + 1] == "qwenpt.train.sft"
    assert f"model.path={model}" in arguments
    assert f"data.train_files={data / 'train/pool.parquet'}" in arguments
    assert f"data.val_files={data / 'validation/pool.parquet'}" in arguments
    assert "data.custom_cls.path=pkg://qwenpt.train.common.dataset" in arguments
    assert "data.max_length=262144" in arguments
    assert "model.lora_rank=0" in arguments
    assert "model.use_fused_kernels=false" in arguments
    assert "+checkpoint.save_lora_only=false" in arguments
    assert "engine.model_dtype=float32" in arguments
    assert "engine.offload_policy=false" in arguments
    assert "engine.param_offload=false" in arguments
    assert "engine.optimizer_offload=false" in arguments
    assert "model.enable_activation_offload=false" in arguments
    assert "model.use_liger=false" in arguments
    assert "data.train_batch_size=96" in arguments
    assert "data.micro_batch_size_per_gpu=1" in arguments
    assert "data.use_dynamic_bsz=true" in arguments
    assert "+data.length_bucketing=true" in arguments
    assert "+data.length_bucket_pad_tail=true" in arguments
    assert "+data.full_validation=true" in arguments
    assert "+model.override_config.attn_implementation=flash_attention_2" in arguments
    assert "optim.lr=1e-5" in arguments
    assert "optim.lr_scheduler_type=cosine" in arguments
    assert "optim.lr_warmup_steps_ratio=0.1" in arguments
    assert "trainer.total_epochs=1" in arguments
    assert "trainer.seed=20261028" in arguments
    assert "+trainer.progress_tracking=true" in arguments
    assert "trainer.total_training_steps=null" in arguments
    assert "trainer.save_freq=after_each_epoch" in arguments
    assert "trainer.max_ckpt_to_keep=2" in arguments
    assert "--nproc-per-node=6" in arguments
    assert not any("reasoning_effort" in argument for argument in arguments)
    assert arguments[-1] == "trainer.total_training_steps=2"


def test_launcher_configuration_defaults(tmp_path):
    """Compose the packaged config once, independently of the shell coverage matrix."""
    capture, environment = _launch_sft(tmp_path)
    arguments = capture["arguments"]
    # Compose the real packaged config: shell quoting alone does not verify Hydra keys.
    overrides = arguments[arguments.index("--module") + 2:]
    preview = subprocess.run(
        [sys.executable, "-m", "qwenpt.train.sft", *overrides, "--cfg", "job"],
        cwd=tmp_path, env=dict(environment, CUDA_VISIBLE_DEVICES=""), check=True, capture_output=True, text=True,
    )
    assert "name: QwenSFTDataset" in preview.stdout
    assert "loss_scope: all_assistant" in preview.stdout
    config = OmegaConf.create(preview.stdout)
    assert config.model.lora_rank == 0
    assert not config.model.use_fused_kernels
    assert not config.checkpoint.save_lora_only
    assert config.engine.model_dtype == "float32"
    assert not config.engine.offload_policy
    assert not config.engine.param_offload
    assert not config.engine.optimizer_offload
    assert not config.model.enable_activation_offload
    assert not config.model.use_liger
    assert config.model.enable_gradient_checkpointing
    assert config.model.override_config.attn_implementation == "flash_attention_2"
    assert config.data.train_batch_size == 96
    assert config.data.use_dynamic_bsz and config.data.length_bucketing
    assert config.data.length_bucket_pad_tail and not config.data.length_bucket_drop_last
    assert config.data.full_validation and config.trainer.progress_tracking
    assert config.optim.lr == 1e-5
    assert config.optim.lr_scheduler_type == "cosine"
    assert config.optim.lr_warmup_steps_ratio == 0.1
    assert config.trainer.total_epochs == 1
    assert config.trainer.save_freq == "after_each_epoch"
    assert config.trainer.max_ckpt_to_keep == 2


def test_launcher_rejects_missing_cached_model_before_torchrun(tmp_path: Path) -> None:
    """An absent offline snapshot must fail without spawning any training workers."""
    project = tmp_path / "project"
    script_dir = project / "scripts/train"
    script_dir.mkdir(parents=True)
    script = script_dir / "sft.sh"
    shutil.copyfile(Path(__file__).resolve().parents[3] / "scripts/train/sft.sh", script)
    (project / "pyproject.toml").touch()
    activate = project / ".venv/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.touch()
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    marker = tmp_path / "torchrun-started"
    torchrun = binary_dir / "torchrun"
    torchrun.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n")
    torchrun.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{binary_dir}:{environment['PATH']}"
    environment["HF_HUB_CACHE"] = str(tmp_path / "empty-cache")
    environment.pop("QWENPT_SFT_MODEL_PATH", None)
    environment.pop("SLURM_SUBMIT_DIR", None)

    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=environment, capture_output=True, text=True,
    )

    assert result.returncode != 0
    assert "LocalEntryNotFoundError" in result.stderr
    assert not marker.exists()


def test_full_epoch_cosine_schedule_uses_ten_percent_warmup() -> None:
    """Check the native scheduler over ceil(100000 / 96) optimizer steps."""
    from types import SimpleNamespace

    import torch
    from verl.workers.config import FSDPOptimizerConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    config = FSDPOptimizerConfig(
        lr=1e-5, total_training_steps=1042,
        lr_scheduler_type="cosine", lr_warmup_steps_ratio=0.1,
    )
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=config.lr)
    engine = SimpleNamespace(optimizer_config=config, rank=1)
    scheduler = FSDPEngine._build_lr_scheduler(engine, optimizer)
    scale = scheduler.lr_lambdas[0]
    assert scale(0) == 0
    assert scale(103) == pytest.approx(103 / 104)
    assert scale(104) == 1
    assert 0 < scale(521) < 1
    assert scale(1042) == 0
