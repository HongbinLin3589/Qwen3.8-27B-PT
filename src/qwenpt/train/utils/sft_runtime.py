"""Read SFT runtime flags and already-selected autotuner configurations."""

import importlib.metadata
import json
import os
import sys

import torch


def _safe(value, depth=0):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if depth >= 4:
        return {"type": f"{type(value).__module__}.{type(value).__name__}", "truncated": True}
    if isinstance(value, (list, tuple)):
        return [_safe(item, depth + 1) for item in value[:64]]
    if isinstance(value, dict):
        return {str(key): _safe(item, depth + 1) for key, item in list(value.items())[:64]
                if type(key) in (str, int, float, bool)}
    return {"type": f"{type(value).__module__}.{type(value).__name__}"}


def _config(value):
    if value is None:
        return None
    try:
        fields = vars(value)
    except TypeError:
        fields = {}
    selected = {key: _safe(fields[key]) for key in ("kwargs", "num_warps", "num_stages", "num_ctas", "maxnreg")
                if key in fields}
    return {"fields": selected, "fields_accessible": bool(selected),
            "config_repr": json.dumps(selected, sort_keys=True, default=str)[:2048]}


def _autotuners():
    found, seen, visited = [], set(), set()
    modules = [(name, module) for name, module in list(sys.modules.items())
               if name == "fla" or name.startswith("fla.") or name == "triton" or name.startswith("triton.")]
    failures = []
    for module_name, module in sorted(modules):
        if module is None:
            continue
        pending = [(f"{module_name}.{name}", value, 0) for name, value in list(vars(module).items())]
        while pending:
            path, value, depth = pending.pop()
            if id(value) in visited:
                continue
            visited.add(id(value))
            try:
                fields = vars(value)
            except TypeError:
                continue
            except Exception as error:
                if len(failures) < 16:
                    failures.append({"object_path": path, "error_type": type(error).__name__})
                continue
            is_autotuner = any(cls.__name__ == "Autotuner" for cls in type(value).__mro__)
            if is_autotuner and id(value) not in seen:
                seen.add(id(value))
                cache = fields.get("cache")
                entries = []
                if isinstance(cache, dict):
                    for key, config in list(cache.items())[:64]:
                        entries.append({"key_repr": json.dumps(_safe(key), sort_keys=True, default=str)[:2048],
                                        **(_config(config) or {"fields_accessible": False, "config_repr": "null"})})
                found.append({
                    "object_path": path, "class": f"{type(value).__module__}.{type(value).__name__}",
                    "cached_config_count": len(cache) if isinstance(cache, dict) else None,
                    "cache_accessible": isinstance(cache, dict), "cache_entries": entries,
                    "cache_entries_truncated": isinstance(cache, dict) and len(cache) > 64,
                    "last_best_config": _config(fields.get("best_config")),
                })
            if depth < 6:
                for name in ("fn", "func", "__wrapped__", "_fn", "base_fn"):
                    child = fields.get(name)
                    if child is not None:
                        pending.append((f"{path}.{name}", child, depth + 1))
    return sorted(found, key=lambda item: item["object_path"]), {
        "loaded_fla_triton_modules": len(modules), "objects_visited": len(visited),
        "wrapper_depth_limit": 6, "cache_entries_limit_per_autotuner": 64,
        "inspection_failures": failures,
        "limitations": [
            "Only imported fla/triton module globals and fn/func/__wrapped__/_fn/base_fn chains are inspected.",
            "Closure-only, container-held and unimported autotuners may be missed; this is not a complete kernel trace.",
            "Cache entries record selected configs for known keys; they do not establish why numerical differences occur.",
            "Keys/configs use bounded safe structural representations; tensor values and arbitrary repr methods are never read.",
        ],
    }


def collect_runtime_diagnostics() -> dict:
    """Return read-only flags, package versions and discovered autotuner cache entries.

    This does not tune kernels, import FLA/Triton modules, sample RNGs, query GPU
    hardware, synchronize CUDA or perform distributed operations.
    """
    versions = {}
    for package in ("torch", "verl", "transformers", "fla-core", "triton", "tilelang", "flash-attn", "causal-conv1d"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    versions["cuda_runtime_compiled"] = torch.version.cuda
    autotuners, coverage = _autotuners()
    return {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS",
            "CUBLAS_WORKSPACE_CONFIG", "FLA_TILELANG",
        )},
        "versions": versions, "autotuners": autotuners, "coverage": coverage,
    }
