"""Check global loss semantics and the fixed gates used by the GPU offload audit."""

import pytest
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.checkpoint import checkpoint
from types import SimpleNamespace

from legacy.sft.utils.verify_sft_offload import (
    ADAM_OPTIONS, CheckpointBackbone, batch_definition, compare_tensors, replay_cpu_adam,
    restore_sharded_checkpoint, write_sharded_checkpoint,
    OffloadAuditModel, _initial_state,
)
from legacy.sft.verl.sft_heads import prepare_sft_targets


def test_norm_checkpoint_cli_requires_sublayer_mode_before_cuda(monkeypatch, tmp_path):
    from legacy.sft.utils.verify_sft_offload import main

    monkeypatch.setattr("sys.argv", ["verify_sft_offload", "--output", str(tmp_path / "report.json"),
                                    "--candidate-checkpoint-layernorms"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_unequal_microbatches_use_one_global_denominator_and_exclude_boundaries():
    batches = [batch_definition(0, rank, micro, 8, 16) for rank in range(2) for micro in range(2)]
    total = sum(int(batch["mask"].sum()) for batch in batches)
    weight = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)
    expected_weight = weight.detach().clone().requires_grad_()
    accumulated = 0
    expected = 0
    for batch in batches:
        inputs = batch["inputs"].double().requires_grad_()
        targets = prepare_sft_targets(batch["ids"], batch["mask"], batch["offsets"],
                                      global_supervised_tokens=total, dp_size=2)
        logits = F.linear(inputs.squeeze(0), weight)
        accumulated = accumulated + (F.cross_entropy(logits, targets.labels, reduction="none") * targets.mask).sum() * targets.normalizer / 2
        reference_logits = F.linear(inputs.squeeze(0), expected_weight)
        for start, end in zip(batch["offsets"][:-1], batch["offsets"][1:]):
            expected = expected + F.cross_entropy(reference_logits[start:end - 1],
                batch["ids"][start + 1:end].masked_fill(batch["mask"][start + 1:end] == 0, -100),
                reduction="sum") / total
    accumulated.backward()
    expected.backward()
    torch.testing.assert_close(accumulated, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(weight.grad, expected_weight.grad, rtol=1e-12, atol=1e-12)
    assert batches[-1]["mask"].sum() == 0


def test_native_nonreentrant_checkpoint_saves_only_boundary_and_recomputes():
    layer = CheckpointBackbone(8).double()
    reference = CheckpointBackbone(8).double()
    reference.load_state_dict(layer.state_dict())
    reference.gradient_checkpointing = False
    inputs = torch.randn(1, 17, 8, dtype=torch.float64, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    saved = []

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor.detach().clone()

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = layer(inputs)
    reference_output = reference(reference_inputs)
    output.sum().backward()
    reference_output.sum().backward()
    assert saved == [(0,), (1, 17, 8)]
    assert layer.forward_calls == 2
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    for actual, expected in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def test_fixed_gradient_gate_rejects_large_drift_and_nonfinite_values():
    expected = torch.ones(12)
    assert compare_tensors(expected * 1.01, expected)["passed"]
    assert not compare_tensors(expected * 1.03, expected)["passed"]
    assert not compare_tensors(expected * float("nan"), expected)["passed"]
    assert not compare_tensors(expected + 1e-4, expected, strict=True)["passed"]


def test_strict_adam_replay_checks_second_moment_separately():
    initial = {"weight": torch.tensor([0.1, -0.2, 0.3])}
    parameter = torch.nn.Parameter(initial["weight"].clone())
    optimizer = torch.optim.AdamW([parameter], **ADAM_OPTIONS)
    snapshots = []
    for step in range(3):
        before = parameter.detach().clone()
        parameter.grad = torch.tensor([0.3, -0.4, 0.7]) * (step + 1)
        gradient = parameter.grad.clone()
        optimizer.step()
        snapshots.append({"step": step + 1, "gradients": {"weight": gradient},
                          "before": {"weight": before}, "parameters": {"weight": parameter.detach().clone()},
                          "moments": {"weight": {key: optimizer.state[parameter][key].clone()
                                                  for key in ("exp_avg", "exp_avg_sq")}}})
    assert all(row["passed"] for row in replay_cpu_adam(initial, snapshots))
    snapshots[1]["moments"]["weight"]["exp_avg_sq"].mul_(1.1)
    result = replay_cpu_adam(initial, snapshots)
    assert not result[1]["checks"]["weight/exp_avg_sq"]["passed"]
    assert result[1]["checks"]["weight/parameters"]["passed"]


def test_cpu_checkpoint_restores_fresh_model_adam_counter_and_next_update(tmp_path):
    model = torch.nn.Linear(8, 4, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), **ADAM_OPTIONS)
    inputs = torch.arange(24, dtype=torch.float32).view(3, 8) / 24

    def step(module, optim):
        optim.zero_grad(set_to_none=True)
        module(inputs).square().mean().backward()
        optim.step()

    step(model, optimizer)
    path = tmp_path / "rank0.pt"
    receipt = write_sharded_checkpoint(model, optimizer, path, completed_step=1)
    resumed = torch.nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        resumed.weight.zero_()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), **ADAM_OPTIONS)
    assert not resumed_optimizer.state
    assert restore_sharded_checkpoint(resumed, resumed_optimizer, path, expected_step=1) == 1
    assert receipt["scope"] == "torch_sharded_state_dict_not_native_trainer_checkpoint"
    assert receipt["bytes"] > 0
    for _ in range(2):
        step(model, optimizer)
        step(resumed, resumed_optimizer)
        torch.testing.assert_close(model.weight, resumed.weight, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(optimizer.state[model.weight][key],
                                       resumed_optimizer.state[resumed.weight][key], rtol=0, atol=0)


def test_cpu_checkpoint_rejects_wrong_resume_step(tmp_path):
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), **ADAM_OPTIONS)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    path = tmp_path / "rank0.pt"
    write_sharded_checkpoint(model, optimizer, path, completed_step=1)
    with pytest.raises(ValueError, match="counter"):
        restore_sharded_checkpoint(model, optimizer, path, expected_step=2)


def test_reentrant_recompute_reaches_outer_saved_tensor_hooks_and_preserves_gradients():
    layer = CheckpointBackbone(8).double()
    layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=True)
    reference = CheckpointBackbone(8).double()
    reference.load_state_dict(layer.state_dict())
    reference.gradient_checkpointing = False
    inputs = torch.randn(1, 17, 8, dtype=torch.float64, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    saved_recompute = []

    def pack(tensor):
        if layer.is_recompute:
            saved_recompute.append(tuple(tensor.shape))
        return tensor.detach().clone()

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        layer(inputs).sum().backward()
    reference(reference_inputs).sum().backward()
    assert saved_recompute and (8, 8) in saved_recompute
    assert layer.forward_calls == 2
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    for actual, expected in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def test_real_qwen_fixture_architecture_has_identical_seed_and_sublayer_shapes():
    from legacy.sft.verl.checkpointing import qwen3_5_sublayer_checkpointing

    args = SimpleNamespace(candidate_sublayer_checkpointing=True, hidden_size=32, vocab_size=64)
    initial = _initial_state(args)
    assert all(torch.equal(value, _initial_state(args)[name]) for name, value in initial.items())
    baseline = OffloadAuditModel(32, 64, "native", qwen_backbone=True)
    candidate = OffloadAuditModel(32, 64, "native", qwen_backbone=True)
    baseline.load_state_dict(initial)
    candidate.load_state_dict(initial)
    batch = batch_definition(0, 0, 0, 32, 64)
    targets = prepare_sft_targets(batch["ids"], batch["mask"], batch["offsets"],
                                  global_supervised_tokens=int(batch["mask"].sum()))
    inputs = batch["inputs"].requires_grad_()
    expected = baseline(inputs, targets)
    expected.backward()
    copied = batch["inputs"].detach().clone().requires_grad_()
    with qwen3_5_sublayer_checkpointing(candidate, True) as names:
        actual = candidate(copied, targets)
        actual.backward()
    assert len(names) == 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(copied.grad, inputs.grad, rtol=0, atol=0)
    for left, right in zip(baseline.parameters(), candidate.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
