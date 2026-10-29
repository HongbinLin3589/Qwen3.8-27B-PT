"""Verify the optional saved-tensor context preserves nonreentrant checkpoint math."""

from contextlib import nullcontext

from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.checkpoint import checkpoint

from legacy.sft.entrypoint import _checkpoint_activation_context, _configure_checkpoint_reentrancy, _model_checkpoint_context


def _config(enabled=False, checkpointing=True, native=False):
    return OmegaConf.create({
        "trainer": {"checkpoint_activation_offload": enabled},
        "model": {"enable_gradient_checkpointing": checkpointing, "enable_activation_offload": native},
    })


def test_disabled_context_does_not_install_saved_tensor_hooks(monkeypatch):
    """Default training must keep its original activation policy."""
    def unexpected(**kwargs):
        raise AssertionError("save_on_cpu must be opt-in")

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", unexpected)
    with _checkpoint_activation_context(_config()):
        pass


@pytest.mark.parametrize("checkpointing,native", [(False, False), (True, True)])
def test_rejects_conflicting_checkpoint_policies(checkpointing, native):
    """Avoid replacing HF checkpointing or nesting the broken native async handler."""
    with pytest.raises(ValueError, match="requires gradient checkpointing"):
        _checkpoint_activation_context(_config(True, checkpointing, native))


def test_reentrant_policy_is_opt_in_and_requires_cpu_hook():
    """Keep the native HF policy unless the complete candidate is explicitly selected."""
    from unittest.mock import Mock

    model = Mock()
    config = _config()
    _configure_checkpoint_reentrancy(config, model)
    model.gradient_checkpointing_enable.assert_not_called()
    config.trainer.checkpoint_use_reentrant = True
    with pytest.raises(ValueError, match="requires gradient checkpointing"):
        _configure_checkpoint_reentrancy(config, model)
    config.trainer.checkpoint_activation_offload = True
    _configure_checkpoint_reentrancy(config, model)
    model.gradient_checkpointing_enable.assert_called_once_with(gradient_checkpointing_kwargs={"use_reentrant": True})


@pytest.mark.parametrize("extra", [
    {"mlp_chunk_tokens": 8192, "checkpoint_sublayers": True},
    {"mlp_chunk_tokens": 8192, "checkpoint_use_reentrant": True},
    {"checkpoint_sublayers": True, "checkpoint_use_reentrant": True},
])
def test_nested_checkpoint_candidates_cannot_be_combined(extra):
    """Keep separately measured policies from silently creating an untested combination."""
    config = _config(True)
    config.trainer.update(extra)
    with pytest.raises(ValueError, match="independent candidates"):
        _model_checkpoint_context(config, None)


def test_nested_checkpoint_requires_outer_gc_and_default_is_unchanged():
    """Reject a partial memory policy before entering training."""
    config = _config()
    with _model_checkpoint_context(config, None):
        pass
    config.model.enable_gradient_checkpointing = False
    config.trainer.checkpoint_sublayers = True
    with pytest.raises(ValueError, match="require model.enable_gradient_checkpointing"):
        _model_checkpoint_context(config, None)
    config.model.enable_gradient_checkpointing = True
    config.model.enable_activation_offload = True
    with pytest.raises(ValueError, match="cannot use native activation offload"):
        _model_checkpoint_context(config, None)


def test_layernorm_checkpoint_requires_the_complete_sublayer_policy():
    """Avoid a standalone norm policy whose capacity has not been measured."""
    config = _config(True)
    config.trainer.checkpoint_layernorms = True
    with pytest.raises(ValueError, match="requires checkpoint_sublayers"):
        _model_checkpoint_context(config, None)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reentrant_cpu_hook_matches_nonreentrant_gradients(dtype):
    """Compare recomputation schedules before the separate distributed GPU gate."""
    generator = torch.Generator().manual_seed(29)
    initial = torch.randn(8, 8, generator=generator)
    inputs = torch.randn(13, 8, generator=generator).to(dtype)
    results = []
    for reentrant in (False, True):
        weight = torch.nn.Parameter(initial.clone())
        value = inputs.clone().requires_grad_()
        config = _config(reentrant)
        with _checkpoint_activation_context(config):
            output = checkpoint(lambda x: torch.nn.functional.linear(x, weight.to(dtype)).sin(),
                                value, use_reentrant=reentrant)
            loss = output.float().square().mean()
            loss.backward()
        results.append((loss.detach(), value.grad, weight.grad))
    for expected, actual in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_gradients_and_adam_states_remain_identical(dtype):
    """CPU execution checks context semantics; real transfer is covered by the GPU fixture."""
    generator = torch.Generator().manual_seed(17)
    initial = torch.randn(8, 8, generator=generator)
    inputs = torch.randn(13, 8, generator=generator).to(dtype)
    results = []
    for offload in (False, True):
        weight = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW([weight], lr=1e-4, foreach=False)
        history = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            calls = []

            def layer(value):
                calls.append(True)
                return torch.nn.functional.linear(value, weight.to(dtype)).sin()

            with _checkpoint_activation_context(_config(True)) if offload else nullcontext():
                output = checkpoint(layer, inputs, use_reentrant=False)
                loss = output.float().square().mean()
                loss.backward()
            assert len(calls) == 2
            history.append((loss.detach().clone(), weight.grad.clone()))
            optimizer.step()
        state = optimizer.state[weight]
        results.append((history, weight.detach().clone(), state["exp_avg"].clone(), state["exp_avg_sq"].clone()))
    reference, candidate = results
    for reference_step, candidate_step in zip(reference[0], candidate[0], strict=True):
        for expected, actual in zip(reference_step, candidate_step, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected, actual in zip(reference[1:], candidate[1:], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
