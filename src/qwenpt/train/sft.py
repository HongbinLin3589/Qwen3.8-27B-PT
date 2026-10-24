"""Launch verl SFT with its packaged configuration and project data adapter."""

import json
import os
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import hydra
from omegaconf import DictConfig


def run_sft(config: DictConfig) -> None:
    """Run seeded native SFT and record synchronized step and run wall times."""
    run_started = perf_counter()
    import torch
    from transformers import set_seed
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils import tensordict_utils as tu
    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
    from qwenpt.train.verl.ce import sft_logprobs_backend
    from qwenpt.train.utils.sft_benchmark import BenchmarkFinished, BenchmarkRecorder

    seed = int(config.trainer.seed)
    step_seconds = []
    benchmark_steps = config.trainer.get("benchmark_steps")
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
                if benchmark is not None:
                    benchmark.before_batch(*args, **kwargs)
                # Include asynchronous device work and CPU optimizer/offload in wall time.
                if config.trainer.device == "cuda":
                    torch.cuda.synchronize()
                started = perf_counter()
                output = native_train_batch(*args, **kwargs)
                if config.trainer.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = perf_counter() - started
                step_seconds.append(elapsed)
                if output is not None:
                    tu.get(output, "metrics")["timing/train_step_seconds"] = elapsed
                if benchmark is not None:
                    benchmark.after_batch(output, elapsed, self)
                return output

            self.training_client.train_batch = timed_train_batch

        def _build_dataloader(self) -> None:
            """Set sampler seeds before either loader creates its iterator."""
            super()._build_dataloader()
            # verl 0.9.1 omits trainer.seed when constructing these samplers.
            self.train_sampler.seed = seed
            if self.val_dataloader is not None:
                self.val_sampler.seed = seed

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
        with offload_context:
            trainer = SeededSFTTrainer(config=config)
        initialization_seconds = perf_counter() - run_started
        fit_started = perf_counter()
        with sft_logprobs_backend(
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
    finally:
        finished = perf_counter()
        try:
            try:
                if benchmark is not None:
                    benchmark.finish(trainer, status)
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
    if config.trainer.get("numa_binding", False):
        from qwenpt.train.utils.numa import bind_to_local_gpu

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
