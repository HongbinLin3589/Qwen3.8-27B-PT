"""Launch offline BOLT training with verl's packaged SFT engine configuration."""

import hydra
from omegaconf import DictConfig


def seed_training(config: DictConfig) -> None:
    """Seed Python, NumPy, Torch and CUDA before constructing the model or LoRA."""
    from transformers import set_seed

    seed = int(config.trainer.seed)
    config.engine.seed = seed
    set_seed(seed)


@hydra.main(
    config_path="pkg://verl.trainer.config",
    config_name="sft_trainer_engine",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Train on fixed, reward-labeled Parquet trajectories using BOLT weights."""
    from tensordict import set_capture_non_tensor_stack
    from verl.utils.device import auto_set_device
    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

    from qwenpt.train.bolt.trainer import BOLTTrainer
    from qwenpt.train.bolt.ce import bolt_logprobs_backend

    if (
        config.data.custom_cls.path not in (
            "pkg://qwenpt.train.bolt.dataset",
            "pkg://qwenpt.train.verl.bolt_dataset",
        )
        or config.data.custom_cls.name != "QwenBOLTDataset"
    ):
        raise ValueError("BOLT requires data.custom_cls=qwenpt.train.bolt.dataset.QwenBOLTDataset")
    auto_set_device(config)
    seed_training(config)
    initialize_global_process_group()
    try:
        # Keep one multimodal dictionary per trajectory, even for identical images.
        with set_capture_non_tensor_stack(False), bolt_logprobs_backend(config):
            BOLTTrainer(config).fit()
    finally:
        destroy_global_process_group()


if __name__ == "__main__":
    main()
