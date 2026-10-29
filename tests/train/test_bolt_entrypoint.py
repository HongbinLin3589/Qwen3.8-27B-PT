"""Check the real BOLT launcher and trainer guards without launching workers."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest


@pytest.mark.parametrize("slurm", [False, True])
@pytest.mark.parametrize("with_validation", [False, True])
@pytest.mark.parametrize("wandb_mode,exit_code", [(None, 0), ("offline", 17)])
def test_launcher_preserves_paths_and_optional_validation(
    tmp_path, slurm, with_validation, wandb_mode, exit_code,
) -> None:
    """Capture torchrun arguments and compose the actual packaged Hydra config."""
    project = tmp_path / "project with spaces"
    script_dir = project / "scripts/train"
    script_dir.mkdir(parents=True)
    script = script_dir / "bolt.sh"
    shutil.copyfile(Path(__file__).resolve().parents[2] / "scripts/train/bolt.sh", script)
    (project / "pyproject.toml").touch()
    activate = project / ".venv/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.touch()
    model = project / "models/Qwen3.8-27B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    data = project / "data/processed/bolt/math-rl-n8-8k-v1"
    data.mkdir(parents=True)
    (data / "train.parquet").touch()
    if with_validation:
        (data / "val.parquet").touch()

    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    captured = tmp_path / "arguments.json"
    torchrun = binary_dir / "torchrun"
    torchrun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "assert 'ROCR_VISIBLE_DEVICES' not in os.environ\n"
        "assert 'HIP_VISIBLE_DEVICES' not in os.environ\n"
        "assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "with open(os.environ['QWENPT_CAPTURE_ARGUMENTS'], 'w') as stream:\n"
        "    json.dump({'arguments': sys.argv[1:], 'pid': os.getpid(), 'wandb_mode': os.environ['WANDB_MODE']}, stream)\n"
        "sys.exit(int(os.environ['QWENPT_CAPTURE_EXIT_CODE']))\n"
    )
    torchrun.chmod(0o755)
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{binary_dir}:{environment['PATH']}",
        "QWENPT_CAPTURE_ARGUMENTS": str(captured),
        "QWENPT_CAPTURE_EXIT_CODE": str(exit_code),
        "ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "0,1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
    })
    if wandb_mode is None:
        environment.pop("WANDB_MODE", None)
    else:
        environment["WANDB_MODE"] = wandb_mode
    if slurm:
        environment["SLURM_SUBMIT_DIR"] = str(script_dir)
    else:
        environment.pop("SLURM_SUBMIT_DIR", None)

    with subprocess.Popen(
        ["bash", str(script), "trainer.total_training_steps=2"],
        cwd=tmp_path, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as process:
        _, stderr = process.communicate()
    assert process.returncode == exit_code, stderr
    capture = json.loads(captured.read_text())
    assert capture["pid"] == process.pid
    assert capture["wandb_mode"] == (wandb_mode or "online")
    arguments = capture["arguments"]
    assert arguments[arguments.index("--module") + 1] == "qwenpt.train.bolt"
    assert f"model.path={model}" in arguments
    assert f"data.train_files={data / 'train.parquet'}" in arguments
    assert f"data.val_files={data / 'val.parquet' if with_validation else 'null'}" in arguments
    assert "data.custom_cls.path=pkg://qwenpt.train.verl.bolt_dataset" in arguments
    assert "data.custom_cls.name=QwenBOLTDataset" in arguments
    assert "+data.bolt_beta=0.1" in arguments
    assert "--nproc-per-node=2" in arguments
    assert "engine.fsdp_size=2" in arguments
    assert "trainer.n_gpus_per_node=2" in arguments
    assert "trainer.seed=20261020" in arguments
    assert "engine.seed=${trainer.seed}" in arguments
    assert arguments[-1] == "trainer.total_training_steps=2"
    if not slurm and with_validation:
        overrides = arguments[arguments.index("--module") + 2:]
        preview = subprocess.run(
            [sys.executable, "-m", "qwenpt.train.bolt", *overrides, "--cfg", "job", "--resolve"],
            cwd=tmp_path, env=environment, check=True, capture_output=True, text=True,
        )
        assert "name: QwenBOLTDataset" in preview.stdout
        assert "bolt_beta: 0.1" in preview.stdout
        assert "loss_scope: all_assistant" in preview.stdout
        settings = OmegaConf.create(preview.stdout)
        assert settings.data.train_batch_size == 32
        assert settings.data.micro_batch_size_per_gpu == 1
        assert settings.optim.lr == 5e-6
        assert settings.optim.lr_scheduler_type == "cosine"
        assert settings.optim.lr_warmup_steps_ratio == 0.1
        assert settings.trainer.save_freq == 100
        assert settings.trainer.max_ckpt_to_keep == 6
        assert settings.engine.seed == settings.trainer.seed == 20261020
        assert settings.engine.full_determinism is False
        changed = subprocess.run(
            [sys.executable, "-m", "qwenpt.train.bolt", *overrides, "trainer.seed=20261021", "--cfg", "job", "--resolve"],
            cwd=tmp_path, env=environment, check=True, capture_output=True, text=True,
        )
        settings = OmegaConf.create(changed.stdout)
        assert settings.engine.seed == settings.trainer.seed == 20261021


@pytest.mark.parametrize("validation_ids", [None, {"validation-prompt"}, {"train-prompt"}])
def test_trainer_rejects_prompt_overlap_across_splits(monkeypatch, validation_ids) -> None:
    """Enforce prompt isolation while retaining optional validation."""
    from verl.trainer.sft_trainer import SFTTrainer

    from qwenpt.train.verl.bolt import BOLTTrainer

    def build_dataset(trainer):
        trainer.train_dataset = SimpleNamespace(prompt_ids={"train-prompt"})
        trainer.val_dataset = None if validation_ids is None else SimpleNamespace(prompt_ids=validation_ids)

    monkeypatch.setattr(SFTTrainer, "_build_dataset", build_dataset)
    trainer = object.__new__(BOLTTrainer)
    if validation_ids == {"train-prompt"}:
        with pytest.raises(ValueError, match="prompt_id sets overlap"):
            trainer._build_dataset()
    else:
        trainer._build_dataset()


@pytest.mark.parametrize("batch_size", [0, -1, 3])
def test_trainer_rejects_invalid_global_batch_before_building_loader(monkeypatch, batch_size) -> None:
    """Prevent native batch truncation when global batches cannot divide across ranks."""
    from verl.trainer.sft_trainer import SFTTrainer

    from qwenpt.train.verl.bolt import BOLTTrainer

    def unexpected_loader_build(trainer):
        pytest.fail("An invalid global batch reached the native loader")

    monkeypatch.setattr(SFTTrainer, "_build_dataloader", unexpected_loader_build)
    trainer = object.__new__(BOLTTrainer)
    trainer.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=batch_size))
    trainer.engine = SimpleNamespace(get_data_parallel_size=lambda: 2)
    with pytest.raises(ValueError, match="positive and divisible"):
        trainer._build_dataloader()


@pytest.mark.parametrize("train_length,val_length", [(0, None), (1, 0)])
def test_trainer_rejects_empty_complete_batches(monkeypatch, train_length, val_length) -> None:
    """Expose dropped final batches before training or validation silently does nothing."""
    from verl.trainer.sft_trainer import SFTTrainer

    from qwenpt.train.verl.bolt import BOLTTrainer

    def build_dataloader(trainer):
        trainer.train_dataloader = [None] * train_length
        trainer.val_dataloader = None if val_length is None else [None] * val_length

    monkeypatch.setattr(SFTTrainer, "_build_dataloader", build_dataloader)
    trainer = object.__new__(BOLTTrainer)
    trainer.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=4))
    trainer.engine = SimpleNamespace(get_data_parallel_size=lambda: 2)
    with pytest.raises(ValueError, match="full global"):
        trainer._build_dataloader()
