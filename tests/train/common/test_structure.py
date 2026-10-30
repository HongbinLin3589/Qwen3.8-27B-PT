"""Check public imports, compatibility CLIs, and isolated data-tool dependencies."""

import importlib
import os
import subprocess
import sys

from omegaconf import OmegaConf
import pytest


@pytest.mark.parametrize("old,new", [
    ("verl.dataset", "common.dataset"),
    ("verl.template", "common.template"),
    ("verl.sampler", "sft.sampler"),
    ("verl.bolt", "bolt.trainer"),
    ("verl.bolt_dataset", "bolt.dataset"),
    ("verl.bolt_ce", "bolt.ce"),
    ("utils.numa", "common.numa"),
    ("utils.prepare_data", "sft.split_data"),
    ("utils.prepare_sft_data", "sft.prepare_data"),
    ("utils.sft_progress", "sft.progress"),
    ("utils.inspect_data", "sft.inspect_data"),
    ("utils.bolt_rollout", "bolt.rollout"),
    ("utils.bolt_math", "bolt.math"),
    ("utils.bolt_data", "bolt.data"),
    ("utils.bolt_complete", "bolt.complete"),
    ("utils.bolt_recovery", "bolt.recovery"),
    ("utils.bolt_verify", "bolt.verify"),
    ("utils.bolt_train_verify", "bolt.train_verify"),
])
def test_old_module_paths_resolve_to_the_same_implementation(old, new):
    """Hydra, private imports, and patches must all reach the canonical module."""
    assert importlib.import_module("qwenpt.train." + old) is importlib.import_module("qwenpt.train." + new)


@pytest.mark.parametrize("algorithm,names", [
    ("sft", ("main", "run_sft")),
    ("bolt", ("main", "seed_training")),
])
def test_existing_entrypoint_exports_are_preserved(algorithm, names):
    package = importlib.import_module("qwenpt.train." + algorithm)
    implementation = importlib.import_module("qwenpt.train." + algorithm + ".entrypoint")
    for name in names:
        assert getattr(package, name) is getattr(implementation, name)


@pytest.mark.parametrize("module", [
    "prepare_data", "prepare_sft_data", "sft_progress", "inspect_data",
    "bolt_rollout", "bolt_recovery", "bolt_verify", "bolt_train_verify",
])
def test_existing_tool_clis_still_parse_help_without_training(module):
    result = subprocess.run(
        [sys.executable, "-m", "qwenpt.train.utils." + module, "--help"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_data_tools_do_not_require_training_framework_imports():
    """BOLT rollout and SFT preparation must remain usable in separate environments."""
    code = """
import sys
for name in ('hydra', 'omegaconf', 'verl'):
    sys.modules[name] = None
import qwenpt.train.bolt.rollout
import qwenpt.train.sft.prepare_data
import qwenpt.train.utils.bolt_rollout
import qwenpt.train.utils.prepare_sft_data
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dataset_path", [
    "pkg://qwenpt.train.bolt.dataset", "pkg://qwenpt.train.verl.bolt_dataset",
])
def test_bolt_accepts_existing_and_canonical_dataset_config(monkeypatch, dataset_path):
    """Old saved configs must still reach the same trainer without starting workers."""
    from types import SimpleNamespace

    from verl.utils import device, distributed
    from qwenpt.train.bolt import entrypoint, trainer

    events = []
    monkeypatch.setattr(device, "auto_set_device", lambda config: None)
    monkeypatch.setattr(entrypoint, "seed_training", lambda config: None)
    monkeypatch.setattr(distributed, "initialize_global_process_group", lambda: events.append("initialize"))
    monkeypatch.setattr(distributed, "destroy_global_process_group", lambda: events.append("destroy"))
    monkeypatch.setattr(trainer, "BOLTTrainer", lambda config: SimpleNamespace(fit=lambda: events.append("fit")))
    config = OmegaConf.create({
        "data": {"custom_cls": {"path": dataset_path, "name": "QwenBOLTDataset"}},
        "trainer": {},
    })
    entrypoint.main.__wrapped__(config)
    assert events == ["initialize", "fit", "destroy"]


def test_training_modules_import_without_archive():
    """Keep the installed training package independent of the removable archive."""
    code = """
import importlib
import pkgutil
import sys

sys.modules['legacy'] = None
import qwenpt.train
for module in pkgutil.walk_packages(qwenpt.train.__path__, 'qwenpt.train.'):
    if not module.name.endswith('.__main__'):
        importlib.import_module(module.name)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
