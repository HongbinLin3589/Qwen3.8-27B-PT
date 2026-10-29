"""Run native verl SFT with seeded data loading and production progress auditing."""

import json
import os
import traceback
from pathlib import Path
from time import perf_counter

import hydra
from omegaconf import DictConfig


def _validate_production_config(config: DictConfig) -> None:
    """Reject archived experiments before device or distributed initialization."""
    defaults = {
        "model": {
            "use_fused_kernels": (None, False),
            "use_liger": (None, False),
            "enable_activation_offload": (None, False),
        },
        "engine": {
            "offload_policy": (None, False),
            "param_offload": (None, False),
            "optimizer_offload": (None, False),
        },
        "trainer": {
            "output_head_backend": (None, "native"),
            "ce_backend": (None, "native"),
            "ce_chunk_size": (None, 256),
            "output_head_chunk_size": (None, 512),
            "benchmark_steps": (None,),
            "capture_head_input": (None, False),
            "cpu_offload_pin_memory": (None, True),
            "checkpoint_activation_offload": (None, False),
            "checkpoint_use_reentrant": (None, False),
            "checkpoint_sublayers": (None, False),
            "checkpoint_layernorms": (None, False),
            "mlp_chunk_tokens": (None, 0),
            "fsdp2_backward_prefetch": (None, "default"),
        },
    }
    unsupported = []
    for section, options in defaults.items():
        values = config.get(section) or {}
        for key, allowed in options.items():
            if values.get(key) not in allowed:
                unsupported.append(f"{section}.{key}={values[key]!r}")
    if unsupported:
        raise ValueError(
            "Production SFT requires the native output head and no CPU offload; "
            "use legacy.sft.entrypoint for experimental options: " + ", ".join(unsupported)
        )


def run_sft(config: DictConfig) -> None:
    """Run the native trainer with seed control and unsynchronized host wall timing."""
    _validate_production_config(config)
    if config.get("engine", {}).get("ulysses_sequence_parallel_size", 1) > 1:
        model_config = config.get("model", {})
        attention = model_config.get("override_config", {}).get("attn_implementation", "flash_attention_2")
        # verl's FlashAttention hook exchanges sequence shards; SDPA has no such hook.
        if not model_config.get("use_remove_padding", False) or attention != "flash_attention_2":
            raise ValueError("Sequence parallel SFT requires use_remove_padding=true and flash_attention_2")
    run_started = perf_counter()
    from transformers import set_seed
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils import tensordict_utils as tu
    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

    seed = int(config.trainer.seed)
    step_seconds = []
    track_progress = config.trainer.get("progress_tracking", False)
    progress = None

    class SeededSFTTrainer(SFTTrainer):
        """Preserve native training with seeded, optionally length-bucketed loaders."""

        def _build_dataloader(self) -> None:
            """Seed native loaders and optionally use length buckets and complete validation."""
            super()._build_dataloader()
            # verl 0.9.1 omits trainer.seed when constructing these samplers.
            self.train_sampler.seed = seed
            if self.val_dataloader is not None:
                self.val_sampler.seed = seed
            if config.data.get("length_bucketing", False):
                from qwenpt.train.verl.sampler import build_length_bucket_dataloader

                self.train_sampler, self.train_dataloader = build_length_bucket_dataloader(
                    self.train_dataset, config.data, collate_fn=self.collate_fn,
                    num_replicas=self.engine.get_data_parallel_size(),
                    rank=self.engine.get_data_parallel_rank(), seed=seed,
                    track_source_indices=track_progress,
                )
            elif track_progress:
                raise ValueError("Production progress tracking requires length_bucketing")
            if config.data.get("full_validation", False) and self.val_dataset is not None:
                from qwenpt.train.verl.sampler import build_full_validation_dataloader

                self.val_sampler, self.val_dataloader = build_full_validation_dataloader(
                    self.val_dataset, config.data, collate_fn=self.collate_fn,
                    num_replicas=self.engine.get_data_parallel_size(),
                    rank=self.engine.get_data_parallel_rank(), seed=seed,
                )

    status = "failed"
    initialization_seconds = None
    fit_started = None
    try:
        set_seed(seed)
        initialize_global_process_group()
        trainer = SeededSFTTrainer(config=config)
        native_train_batch = trainer.training_client.train_batch

        def timed_train_batch(*args, **kwargs):
            if progress is not None:
                data = kwargs["data"] if "data" in kwargs else args[0]
                progress.before_batch(data)
            started = perf_counter()
            output = native_train_batch(*args, **kwargs)
            elapsed = perf_counter() - started
            step_seconds.append(elapsed)
            if output is not None:
                tu.get(output, "metrics")["timing/train_step_seconds"] = elapsed
            if progress is not None:
                progress.after_batch(output, elapsed)
            return output

        trainer.training_client.train_batch = timed_train_batch
        if track_progress:
            from qwenpt.train.utils.sft_progress import SFTProgressRecorder

            progress = SFTProgressRecorder(trainer)
        initialization_seconds = perf_counter() - run_started
        fit_started = perf_counter()
        trainer.fit()
        status = "completed"
    except BaseException as error:
        if progress is not None:
            progress.fail(error)
            traceback.print_exc()
        raise
    finally:
        finished = perf_counter()
        # A recorder failure must not leave ranks alive or prevent the timing receipt.
        try:
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
                    "timing_scope": "Host wall time around native train_batch; no explicit per-step CUDA synchronization",
                }
                # Fit also includes data loading, logging, validation and checkpoint saving.
                run_dir = Path(config.trainer.default_local_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


@hydra.main(
    config_path="pkg://verl.trainer.config",
    config_name="sft_trainer_engine",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Run native SFT inside the media context and optional NUMA binding."""
    # Reject before NUMA/device setup; run_sft also accepts direct callers.
    _validate_production_config(config)
    if config.get("trainer", {}).get("numa_binding", False):
        from qwenpt.train.utils.numa import bind_to_local_gpu

        # Bind before model/optimizer first-touch allocations, including helper threads.
        bind_to_local_gpu(receipt_dir=Path(config.trainer.default_local_dir) / "numa_binding")
    from tensordict import set_capture_non_tensor_stack
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    # Identical media dictionaries must retain one entry per sample for verl batching.
    with set_capture_non_tensor_stack(False):
        run_sft(config)


if __name__ == "__main__":
    main()
