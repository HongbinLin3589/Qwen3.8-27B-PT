"""Launch verl SFT with its packaged configuration and project data adapter."""

import json
import os
import traceback
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import hydra
from omegaconf import DictConfig


def _checkpoint_activation_context(config: DictConfig):
    """Optionally save checkpoint tensors on CPU without changing their dtype."""
    if not config.trainer.get("checkpoint_activation_offload", False):
        return nullcontext()
    if not config.model.enable_gradient_checkpointing or config.model.get("enable_activation_offload", False):
        raise ValueError("checkpoint_activation_offload requires gradient checkpointing and disables native activation offload")
    import torch

    # The native async handler counts inactive vision modules; retain HF's GC instead.
    return torch.autograd.graph.save_on_cpu(pin_memory=False, device_type="cuda")


def _configure_checkpoint_reentrancy(config: DictConfig, model) -> None:
    """Opt into HF reentrant checkpointing so CPU hooks also cover recomputation."""
    if not config.trainer.get("checkpoint_use_reentrant", False):
        return
    if not config.model.enable_gradient_checkpointing or not config.trainer.get("checkpoint_activation_offload", False):
        raise ValueError("checkpoint_use_reentrant requires gradient checkpointing and checkpoint_activation_offload")
    if config.model.get("enable_activation_offload", False):
        raise ValueError("checkpoint_use_reentrant cannot use native activation offload")
    # Nonreentrant recomputation installs its own hooks and retains MLP temporaries on GPU.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})


def _model_checkpoint_context(config: DictConfig, model):
    """Select one explicitly requested nonreentrant layer-memory experiment."""
    chunks = config.trainer.get("mlp_chunk_tokens", 0)
    sublayers = config.trainer.get("checkpoint_sublayers", False)
    layernorms = config.trainer.get("checkpoint_layernorms", False)
    if layernorms and not sublayers:
        raise ValueError("checkpoint_layernorms requires checkpoint_sublayers")
    if not chunks and not sublayers:
        return nullcontext()
    if (chunks and sublayers) or config.trainer.get("checkpoint_use_reentrant", False):
        raise ValueError("MLP token chunking, sublayer checkpoints and reentrant checkpoints are independent candidates")
    if not config.model.enable_gradient_checkpointing:
        raise ValueError("Nested checkpoint candidates require model.enable_gradient_checkpointing")
    if config.model.get("enable_activation_offload", False):
        raise ValueError("Nested checkpoint candidates cannot use native activation offload")
    if sublayers:
        from legacy.sft.verl.checkpointing import qwen3_5_sublayer_checkpointing

        return qwen3_5_sublayer_checkpointing(model, enabled=True, include_layernorms=layernorms)
    from legacy.sft.verl.mlp import qwen3_5_mlp_token_chunking

    return qwen3_5_mlp_token_chunking(model, chunks)


def _configure_fsdp2_backward_prefetch(config: DictConfig, model) -> list[str]:
    """Apply an experimental current-unit policy and return the affected module names."""
    policy = config.trainer.get("fsdp2_backward_prefetch", "default")
    if policy == "default":
        return []
    if policy != "current" or config.engine.strategy != "fsdp2":
        raise ValueError("fsdp2_backward_prefetch must be default, or current with engine.strategy=fsdp2")
    from torch.distributed.fsdp import FSDPModule

    named_modules = model.named_modules() if model is not None else ()
    modules = [(name, module) for name, module in named_modules if isinstance(module, FSDPModule)]
    if not modules:
        raise ValueError("Current-unit backward prefetch requires initialized FSDP2 modules")
    for _, module in modules:
        # TODO: Replace when Torch exposes an off switch: [] restores next-unit prefetch.
        # The current unit is already unsharded when this explicit prefetch executes.
        module.set_modules_to_backward_prefetch([module])
    return [name for name, _ in modules]


def run_sft(config: DictConfig) -> None:
    """Run seeded native SFT and record synchronized step and run wall times."""
    if config.get("engine", {}).get("ulysses_sequence_parallel_size", 1) > 1:
        model_config = config.get("model", {})
        attention = model_config.get("override_config", {}).get("attn_implementation", "flash_attention_2")
        # verl's FlashAttention hook exchanges sequence shards; SDPA has no such hook.
        if not model_config.get("use_remove_padding", False) or attention != "flash_attention_2":
            raise ValueError("Sequence parallel SFT requires use_remove_padding=true and flash_attention_2")
    run_started = perf_counter()
    import torch
    from transformers import set_seed
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils import tensordict_utils as tu
    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
    from legacy.sft.verl.ce import sft_logprobs_backend
    from legacy.sft.verl.sft_heads import sft_output_head_backend
    from legacy.sft.utils.sft_benchmark import BenchmarkFinished, BenchmarkRecorder

    seed = int(config.trainer.seed)
    step_seconds = []
    benchmark_steps = config.trainer.get("benchmark_steps")
    track_progress = config.trainer.get("progress_tracking", False)
    if track_progress and benchmark_steps is not None:
        raise ValueError("Production progress tracking cannot use benchmark_steps")
    progress = None
    benchmark = (
        BenchmarkRecorder(config.trainer.default_local_dir, int(benchmark_steps), int(os.environ.get("RANK", "0")))
        if benchmark_steps is not None else None
    )

    class SeededSFTTrainer(SFTTrainer):
        """Keep native training with seeded sampling and optimizer step timing."""

        def __init__(self, config) -> None:
            """Wrap the initialized worker without replacing native training logic."""
            super().__init__(config=config)
            native_train_batch = self.training_client.train_batch

            def timed_train_batch(*args, **kwargs):
                if progress is not None:
                    data = kwargs["data"] if "data" in kwargs else args[0]
                    progress.before_batch(data)
                if benchmark is not None:
                    benchmark.before_batch(*args, **kwargs)
                # Include asynchronous device work and CPU optimizer/offload in wall time.
                if config.trainer.device == "cuda":
                    torch.cuda.synchronize()
                started = perf_counter()
                with _checkpoint_activation_context(config):
                    output = native_train_batch(*args, **kwargs)
                if config.trainer.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = perf_counter() - started
                step_seconds.append(elapsed)
                if output is not None:
                    tu.get(output, "metrics")["timing/train_step_seconds"] = elapsed
                if progress is not None:
                    progress.after_batch(output, elapsed)
                if benchmark is not None:
                    benchmark.after_batch(output, elapsed, self)
                return output

            self.training_client.train_batch = timed_train_batch

        def _build_dataloader(self) -> None:
            """Seed native loaders and optionally replace training batches with length buckets."""
            super()._build_dataloader()
            # verl 0.9.1 omits trainer.seed when constructing these samplers.
            self.train_sampler.seed = seed
            if self.val_dataloader is not None:
                self.val_sampler.seed = seed
            if config.data.get("length_bucketing", False):
                from qwenpt.train.sft.sampler import build_length_bucket_dataloader

                self.train_sampler, self.train_dataloader = build_length_bucket_dataloader(
                    self.train_dataset, config.data, collate_fn=self.collate_fn,
                    num_replicas=self.engine.get_data_parallel_size(),
                    rank=self.engine.get_data_parallel_rank(), seed=seed,
                    track_source_indices=track_progress,
                )
            elif track_progress:
                raise ValueError("Production progress tracking requires length_bucketing")
            if config.data.get("full_validation", False) and self.val_dataset is not None:
                from qwenpt.train.sft.sampler import build_full_validation_dataloader

                self.val_sampler, self.val_dataloader = build_full_validation_dataloader(
                    self.val_dataset, config.data, collate_fn=self.collate_fn,
                    num_replicas=self.engine.get_data_parallel_size(),
                    rank=self.engine.get_data_parallel_rank(), seed=seed,
                )

    status = "failed"
    initialization_seconds = None
    fit_started = None
    trainer = None
    try:
        set_seed(seed)
        initialize_global_process_group()
        offload_context = nullcontext()
        if (
            config.trainer.get("cpu_offload_pin_memory", True) is False
            and config.engine.strategy == "fsdp2"
            and config.engine.offload_policy
        ):
            from unittest.mock import patch
            from verl.workers.engine.fsdp import transformer_impl

            native_cpu_offload_policy = transformer_impl.CPUOffloadPolicy

            def unpinned_cpu_offload_policy(*args, **kwargs):
                kwargs["pin_memory"] = False
                return native_cpu_offload_policy(*args, **kwargs)

            # TODO: Remove this override when verl exposes pinning; this host rejects pinned shards.
            offload_context = patch.object(transformer_impl, "CPUOffloadPolicy", unpinned_cpu_offload_policy)
        head_context = (
            sft_output_head_backend(config)
            if config.trainer.get("output_head_backend") is not None else nullcontext()
        )
        from legacy.sft.verl.fsdp_loading import fsdp2_cpu_load_blocking

        with head_context:
            with offload_context, fsdp2_cpu_load_blocking(
                config.engine.strategy == "fsdp2" and bool(config.engine.get("offload_policy", False))
            ):
                trainer = SeededSFTTrainer(config=config)
            _configure_checkpoint_reentrancy(config, getattr(trainer.engine, "module", None))
            if track_progress:
                from qwenpt.train.sft.progress import SFTProgressRecorder

                progress = SFTProgressRecorder(trainer)
            prefetch_modules = _configure_fsdp2_backward_prefetch(config, getattr(trainer.engine, "module", None))
            if prefetch_modules:
                prefetch_receipt = Path(config.trainer.default_local_dir)
                prefetch_receipt.mkdir(parents=True, exist_ok=True)
                (prefetch_receipt / f"fsdp2_prefetch_rank{os.environ.get('RANK', '0')}.json").write_text(
                    json.dumps({"policy": "current", "modules": prefetch_modules}, indent=2) + "\n"
                )
            if benchmark is not None and config.trainer.get("capture_head_input", False):
                benchmark.capture_head_input(trainer)
            initialization_seconds = perf_counter() - run_started
            fit_started = perf_counter()
            with _model_checkpoint_context(config, getattr(trainer.engine, "module", None)), sft_logprobs_backend(
                config.trainer.get("ce_backend", "native"),
                chunk_size=int(config.trainer.get("ce_chunk_size", 256)),
            ):
                try:
                    trainer.fit()
                except BenchmarkFinished:
                    if benchmark is None:
                        raise
                    status = "benchmark_completed"
                else:
                    status = "completed"
    except BaseException as error:
        if progress is not None:
            progress.fail(error)
        if benchmark is not None or track_progress:
            # Preserve the cause even if distributed teardown cannot complete.
            traceback.print_exc()
        raise
    finally:
        finished = perf_counter()
        try:
            try:
                if benchmark is not None:
                    benchmark.finish(trainer, status)
            finally:
                try:
                    if progress is not None:
                        progress.finish(status)
                finally:
                    destroy_global_process_group()
        finally:
            if int(os.environ.get("RANK", "0")) == 0:
                summary = {
                    "status": status,
                    "completed_train_steps": len(step_seconds),
                    "train_steps_seconds": sum(step_seconds),
                    "train_step_seconds_mean": sum(step_seconds) / len(step_seconds) if step_seconds else None,
                    "initialization_seconds": initialization_seconds,
                    "fit_seconds": finished - fit_started if fit_started is not None else None,
                    "run_seconds": finished - run_started,
                }
                # fit includes data loading, logging, validation and checkpoint saving.
                run_dir = Path(config.trainer.default_local_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


@hydra.main(
    config_path="pkg://verl.trainer.config",
    config_name="sft_trainer_engine",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Run native verl SFT using the composed CLI configuration."""
    if config.get("trainer", {}).get("numa_binding", False):
        from qwenpt.train.common.numa import bind_to_local_gpu

        # Bind before model/optimizer first-touch allocations, including existing helper threads.
        bind_to_local_gpu(receipt_dir=Path(config.trainer.default_local_dir) / "numa_binding")
    from tensordict import set_capture_non_tensor_stack
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    # Identical media dictionaries must retain one entry per sample for verl batching.
    with set_capture_non_tensor_stack(False):
        run_sft(config)


if __name__ == "__main__":
    main()
