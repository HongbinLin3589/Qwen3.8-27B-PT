"""Launch verl SFT with its packaged configuration and project data adapter."""

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="pkg://verl.trainer.config",
    config_name="sft_trainer_engine",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Run native verl SFT using the composed CLI configuration."""
    from verl.trainer.sft_trainer import run_sft
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    run_sft(config)


if __name__ == "__main__":
    main()
