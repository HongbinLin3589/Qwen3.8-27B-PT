"""BOLT's response-normalized weighted loss on the native verl SFT engine."""

import torch
from tensordict import TensorDict
from verl.trainer.sft_trainer import SFTTrainer


def bolt_loss(model_output: dict, data: TensorDict, dp_group=None) -> tuple[torch.Tensor, dict]:
    """Return weighted response-mean NLL, scaled for verl's DP accumulation.

    Inputs follow verl's no-padding contract; ``bolt_weight`` contains one
    complete-prompt-group coefficient per response, independent of microbatches.
    """
    log_probs = model_output["log_probs"]
    mask = data["loss_mask"]
    lengths = mask.offsets().diff()
    shifted_mask = torch.roll(mask.values(), shifts=-1, dims=0).float()
    # Each row's final prediction has no target, including at packed boundaries.
    shifted_mask[mask.offsets()[1:] - 1] = 0
    token_counts = torch.segment_reduce(shifted_mask, "sum", lengths=lengths)
    nll = -torch.segment_reduce(log_probs.values().float() * shifted_mask, "sum", lengths=lengths)
    weights = data["bolt_weight"].values().detach().float()
    response_losses = nll / token_counts.clamp_min(1)

    # The engine sums microbatch gradients and FSDP averages them across ranks.
    # Dividing by token count or sum(weights) would change the BOLT objective.
    dp_size = data["dp_size"]
    global_batch_size = data["global_batch_size"]
    loss = (response_losses * weights).sum() * dp_size / global_batch_size
    return loss, {}


class BOLTTrainer(SFTTrainer):
    """Reuse verl's optimizer, batching, validation and checkpointing for BOLT."""

    def _build_dataset(self) -> None:
        """Build reward-labeled datasets and reject prompt leakage across splits."""
        super()._build_dataset()
        if self.val_dataset is not None and self.train_dataset.prompt_ids & self.val_dataset.prompt_ids:
            raise ValueError("BOLT train/validation prompt_id sets overlap; split complete prompt groups")

    def _build_engine(self) -> None:
        """Build the native training worker and install BOLT's loss callback."""
        if self.config.engine.strategy not in ("fsdp", "fsdp2"):
            raise ValueError("BOLT currently requires engine.strategy=fsdp or fsdp2")
        super()._build_engine()
        self.loss_fn = bolt_loss
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)

    def _build_dataloader(self) -> None:
        """Reject batch settings that native verl would truncate or leave empty."""
        dp_size = self.engine.get_data_parallel_size()
        batch_size = self.config.data.train_batch_size
        if batch_size <= 0 or batch_size % dp_size:
            raise ValueError("data.train_batch_size must be positive and divisible by the data parallel size")
        super()._build_dataloader()
        if not len(self.train_dataloader):
            raise ValueError("BOLT needs at least one full global training batch")
        if self.val_dataloader is not None and not len(self.val_dataloader):
            raise ValueError("BOLT needs at least one full global validation batch, or data.val_files=null")
        # verl 0.9.1 does not forward trainer.seed to its DistributedSamplers.
        self.train_sampler.seed = int(self.config.trainer.seed)
        if self.val_dataloader is not None:
            self.val_sampler.seed = int(self.config.trainer.seed)
