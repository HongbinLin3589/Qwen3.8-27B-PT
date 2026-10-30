"""Compare Qwen's CUDA gated-delta kernel and gradients against its FP32 reference."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path


def main() -> None:
    """Run two small GPU checks and write versions, routing, and errors to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "CUDA_HOME": os.environ.get("CUDA_HOME"),
              "CC": os.environ.get("CC"), "CXX": os.environ.get("CXX"), "cases": []}
    try:
        import torch
        assert torch.cuda.is_available(), "This check requires a CUDA GPU; it must not be skipped"
        from fla.ops.common.backends import common_registry
        from fla.ops.common.backends.tilelang import TileLangBackend
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        report.update({
            "versions": {"torch": torch.__version__, "CUDA": torch.version.cuda,
                         **{name: version(name) for name in ("triton", "fla-core", "tilelang")}},
            "device": torch.cuda.get_device_name(),
            "device_capability": list(torch.cuda.get_device_capability()),
            "tilelang_available": TileLangBackend.is_available(),
            "tilelang_enabled": TileLangBackend.is_enabled(),
            "seed": 20261020,
            "reference": "Transformers torch_chunk_gated_delta_rule, independent FP32 leaves",
            "tolerance_basis": "FLA v0.5.2 tests/ops/test_gdn.py:182-189 FP16 thresholds; "
                               "applied to these BF16 cases, not a general BF16 guarantee",
        })
        assert report["tilelang_available"] and report["tilelang_enabled"], "TileLang must be usable"
        for length in (128, 129):
            torch.manual_seed(20261020 + length)
            shape = (1, length, 48, 128)
            values = {name: torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                      for name in ("q", "k", "v")}
            values["g"] = -(0.01 + 0.1 * torch.rand(shape[:-1], device="cuda"))
            values["beta"] = (0.1 + 0.8 * torch.rand(shape[:-1], device="cuda")).to(torch.bfloat16)
            fast = {name: value.detach().clone().requires_grad_() for name, value in values.items()}
            reference = {name: value.detach().float().clone().requires_grad_()
                         for name, value in values.items()}
            assert all(fast[name].data_ptr() != reference[name].data_ptr() for name in values)
            actual, _ = chunk_gated_delta_rule(**fast, use_qk_l2norm_in_kernel=True)
            expected, _ = torch_chunk_gated_delta_rule(
                reference["q"], reference["k"], reference["v"], reference["g"], reference["beta"],
                use_qk_l2norm_in_kernel=True,
            )
            dout = torch.randn_like(actual)
            actual_grads = torch.autograd.grad(actual, tuple(fast.values()), dout)
            reference_grads = torch.autograd.grad(expected, tuple(reference.values()), dout.float())
            pairs = [("output", expected, actual, 0.005)] + [
                (name, ref, got, 0.02 if name in ("g", "beta") else 0.008)
                for name, ref, got in zip(values, reference_grads, actual_grads, strict=True)
            ]
            errors = {}
            for name, ref, got, tolerance in pairs:
                ref, got = ref.detach().float(), got.detach().float()
                finite = bool(torch.isfinite(ref).all() & torch.isfinite(got).all())
                difference = ref.float() - got.float()
                relative = float(difference.square().mean().sqrt() / (ref.float().square().mean().sqrt() + 1e-8))
                maximum = float(difference.abs().max())
                errors[name] = {"normalized_rmse": relative, "max_abs": maximum,
                                "tolerance": tolerance, "finite": finite,
                                "passed": finite and (relative < tolerance or maximum <= 1e-6)}
            report["cases"].append({"shape": list(shape), "errors": errors})
        active = common_registry.get_active()
        report["active_common_backend"] = active.backend_type if active is not None else None
        # The dispatch log records the backend actually used, rather than just installed.
        report["tilelang_backward_used"] = "common:chunk_bwd_dqkwg:tilelang" in common_registry._logged
        assert report["tilelang_backward_used"], "The guarded backward must actually route to TileLang"
        assert all(error["passed"] for case in report["cases"] for error in case["errors"].values()), report["cases"]
        report["status"] = "passed"
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
