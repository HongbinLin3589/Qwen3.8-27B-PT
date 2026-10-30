"""Build reproducible length buckets with matching micro-steps across DP ranks."""

from collections.abc import Iterator, Sequence
import hashlib
import json
from operator import index
import random
from typing import Any

from torch.utils.data import DistributedSampler, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader


class DistributedLengthBucketBatchSampler(Sampler[list[int]]):
    """Yield rank-local batches from shuffled global length buckets.

    ``global_batch_size`` includes all ``num_replicas`` DP ranks; adjacent
    lengths are assigned to corresponding ``micro_batch_size`` micro-steps.
    A bucket spans ``bucket_size_multiplier`` global batches before sorting.
    ``drop_last`` discards only the randomly selected final global-batch tail.
    Otherwise that tail must contain complete DP micro-steps without repetition,
    or explicitly use ``pad_tail`` to fill its last wave with an internal index
    equal to ``len(lengths)``. The loader maps that index to zero-loss tokens.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        global_batch_size: int,
        num_replicas: int,
        rank: int,
        *,
        micro_batch_size: int = 1,
        bucket_size_multiplier: int = 50,
        seed: int = 0,
        drop_last: bool = True,
        pad_tail: bool = False,
    ) -> None:
        self.lengths = tuple(index(length) for length in lengths)
        self.global_batch_size = index(global_batch_size)
        self.num_replicas = index(num_replicas)
        self.rank = index(rank)
        self.micro_batch_size = index(micro_batch_size)
        self.bucket_size_multiplier = index(bucket_size_multiplier)
        self.seed = index(seed)
        self.drop_last = drop_last
        self.pad_tail = pad_tail
        self.epoch = 0
        if any(length <= 0 for length in self.lengths):
            raise ValueError("Sequence lengths must be positive integers")
        if min(self.global_batch_size, self.num_replicas, self.micro_batch_size, self.bucket_size_multiplier) <= 0:
            raise ValueError("Batch sizes, replica count and bucket multiplier must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        parallel_micro_size = self.num_replicas * self.micro_batch_size
        if self.global_batch_size % parallel_micro_size:
            raise ValueError("global_batch_size must be divisible by num_replicas * micro_batch_size")
        if pad_tail and drop_last:
            raise ValueError("pad_tail requires explicit drop_last=False")
        if not drop_last and not pad_tail and len(self.lengths) % parallel_micro_size:
            raise ValueError("An incomplete DP micro-step requires explicit drop_last=True")
        self.num_padding_samples = (-len(self.lengths)) % parallel_micro_size if pad_tail else 0
        self._configuration = {
            "lengths_sha256": hashlib.sha256(json.dumps(self.lengths).encode()).hexdigest(),
            "global_batch_size": self.global_batch_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "micro_batch_size": self.micro_batch_size,
            "bucket_size_multiplier": self.bucket_size_multiplier,
            "seed": self.seed,
            "drop_last": self.drop_last,
        }
        if pad_tail:
            # Keep older unpadded checkpoints compatible with their original contract.
            self._configuration["pad_tail"] = True

    def __len__(self) -> int:
        """Return the equal number of optimizer batches yielded on every DP rank."""
        count, remainder = divmod(len(self.lengths), self.global_batch_size)
        return count + int(bool(remainder) and not self.drop_last)

    def set_epoch(self, epoch: int) -> None:
        """Select an epoch's shuffle without changing global Python or Torch RNGs."""
        self.epoch = index(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """Return an independent iterator whose cursor StatefulDataLoader can save."""
        return _LengthBucketIterator(self)

    def _batches(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        if self.drop_last:
            # Drop before sorting so every length has the same chance of omission.
            indices = indices[: len(self) * self.global_batch_size]
        bucket_size = self.global_batch_size * self.bucket_size_multiplier
        global_batches = []
        for start in range(0, len(indices), bucket_size):
            bucket = sorted(indices[start : start + bucket_size], key=self.lengths.__getitem__)
            global_batches.extend(
                bucket[offset : offset + self.global_batch_size]
                for offset in range(0, len(bucket), self.global_batch_size)
            )
        rng.shuffle(global_batches)
        local_batches = []
        parallel_micro_size = self.num_replicas * self.micro_batch_size
        for batch in global_batches:
            if self.pad_tail:
                batch = batch + [len(self.lengths)] * (-len(batch) % parallel_micro_size)
            waves = [batch[start : start + parallel_micro_size] for start in range(0, len(batch), parallel_micro_size)]
            rng.shuffle(waves)
            local = []
            for wave in waves:
                ranks = list(range(self.num_replicas))
                # Avoid always assigning the longest sample in a wave to the last rank.
                rng.shuffle(ranks)
                start = ranks.index(self.rank) * self.micro_batch_size
                local.extend(wave[start : start + self.micro_batch_size])
            local_batches.append(local)
        return local_batches


class _TailPaddingDataset:
    """Expose real rows unchanged and one internal, unsupervised collective filler."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        tokenizer = getattr(dataset, "tokenizer", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.pad_token_id is None:
            raise ValueError("Tail padding requires a tokenizer pad_token_id or eos_token_id")
        if getattr(dataset, "max_length", 16) < 16:
            raise ValueError("Tail padding requires max_length >= 16")

    def __len__(self) -> int:
        """Return only the original row count; the sampler owns the extra index."""
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Return an original sample, or 16 valid tokens contributing zero loss."""
        if item != len(self.dataset):
            return self.dataset[item]
        import torch

        # Every rank must execute FSDP collectives, including an incomplete tail.
        # Sixteen tokens also avoid relying on tiny causal-convolution edge cases.
        input_ids = torch.full((16,), self.pad_token_id, dtype=torch.long)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        position_ids = (
            self.dataset.position_ids(inputs)
            if hasattr(self.dataset, "position_ids") else torch.arange(16, dtype=torch.long)
        )
        sample = {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": torch.zeros(16, dtype=torch.long)}
        if getattr(self.dataset, "processor", None) is not None:
            sample["multi_modal_inputs"] = {}
        return sample


class _SourceTrackingDataset:
    """Attach CPU-only row identities for removal before the training engine call."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        """Keep the original row count, excluding the internal padding index."""
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Add the exact source row, using -1 solely for the internal padding sample."""
        import torch

        return {
            **self.dataset[item],
            "sft_source_index": torch.tensor([item if item < len(self) else -1], dtype=torch.long),
        }


class _LengthBucketIterator(Iterator[list[int]]):
    def __init__(self, sampler: DistributedLengthBucketBatchSampler) -> None:
        self.sampler = sampler
        self.epoch = sampler.epoch
        self.batches = sampler._batches(self.epoch)
        self.yielded = 0

    def __next__(self) -> list[int]:
        if self.yielded == len(self.batches):
            raise StopIteration
        batch = self.batches[self.yielded]
        self.yielded += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        """Save the deterministic order's identity and consumed batch count."""
        return {"configuration": self.sampler._configuration, "epoch": self.epoch, "yielded": self.yielded}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a cursor only when lengths and the DP batching contract agree."""
        if state["configuration"] != self.sampler._configuration:
            raise ValueError("Length-bucket checkpoint does not match the dataset or sampler configuration")
        self.epoch = index(state["epoch"])
        self.batches = self.sampler._batches(self.epoch)
        self.yielded = index(state["yielded"])
        if not 0 <= self.yielded <= len(self.batches):
            raise ValueError("Length-bucket checkpoint contains an invalid batch cursor")


class _LengthBucketDataLoader(StatefulDataLoader):
    def state_dict(self) -> dict[str, Any]:
        """Include consumed progress separately from the sampler's prefetch cursor."""
        native_state = super().state_dict()
        return {
            "loader": native_state,
            "configuration": self.batch_sampler._configuration,
            "epoch": self.batch_sampler.epoch,
            # The pinned torchdata counter advances on delivery, not worker prefetch.
            "batches_consumed": self._iterator._num_yielded,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore mid-epoch, or start the next epoch after a fully consumed snapshot."""
        if state["configuration"] != self.batch_sampler._configuration:
            raise ValueError("Length-bucket checkpoint does not match the dataset or sampler configuration")
        if not 0 <= state["batches_consumed"] <= len(self):
            raise ValueError("Length-bucket checkpoint contains an invalid consumed batch count")
        if state["batches_consumed"] == len(self):
            # Native SFT saves before StopIteration but already advances start_epoch.
            # Restoring that exhausted iterator would otherwise skip the next epoch.
            self.batch_sampler.set_epoch(state["epoch"] + 1)
            super().load_state_dict({})
        else:
            self.batch_sampler.set_epoch(state["epoch"])
            super().load_state_dict(state["loader"])


def build_length_bucket_dataloader(
    dataset: Any, data_config: Any, *, collate_fn: Any, num_replicas: int, rank: int, seed: int,
    track_source_indices: bool = False,
) -> tuple[DistributedLengthBucketBatchSampler, Any]:
    """Build a stateful training loader using actual lengths or explicit row estimates.

    ``length_source=tokenized`` uses the dataset's validated ``token_lengths``
    cache, or tokenizes each sample once when no cache is available.
    ``length_source=total_tokens`` explicitly opts into producer length estimates.
    Dynamic batching uses reference micro-steps of one; the engine subsequently
    chooses and synchronizes the actual dynamic micro-batches.
    Optional source indices are audit metadata and must be popped before training.
    """
    from verl.utils.device import get_device_name

    source = data_config.get("length_source", "tokenized")
    if source == "tokenized":
        lengths = getattr(dataset, "token_lengths", None)
        if lengths is None:
            lengths = [len(dataset[item]["input_ids"]) for item in range(len(dataset))]
    elif source == "total_tokens":
        lengths = [row["total_tokens"] for row in dataset.rows]
    else:
        raise ValueError("length_source must be tokenized or total_tokens")
    if len(lengths) != len(dataset):
        raise ValueError("The sequence-length cache must contain one length per dataset sample")
    micro_size = 1 if data_config.get("use_dynamic_bsz", False) else data_config.micro_batch_size_per_gpu
    sampler = DistributedLengthBucketBatchSampler(
        lengths, data_config.train_batch_size, num_replicas, rank,
        micro_batch_size=micro_size,
        bucket_size_multiplier=data_config.get("length_bucket_size_multiplier", 50),
        seed=seed, drop_last=data_config.get("length_bucket_drop_last", True),
        pad_tail=data_config.get("length_bucket_pad_tail", False),
    )
    if len(sampler) == 0:
        raise ValueError("Length bucketing requires at least one complete global training batch")
    loader_dataset = _TailPaddingDataset(dataset) if sampler.pad_tail else dataset
    if track_source_indices:
        loader_dataset = _SourceTrackingDataset(loader_dataset)
    loader = _LengthBucketDataLoader(
        dataset=loader_dataset,
        batch_sampler=sampler, collate_fn=collate_fn,
        num_workers=data_config.num_workers, pin_memory=False, pin_memory_device=get_device_name(),
    )
    return sampler, loader


def build_full_validation_dataloader(
    dataset: Any, data_config: Any, *, collate_fn: Any, num_replicas: int, rank: int, seed: int,
) -> tuple[DistributedSampler, StatefulDataLoader]:
    """Read every validation row once with one sample per rank and no dropped tail.

    Validation must divide evenly across DP ranks; otherwise DistributedSampler
    would repeat rows to equalize collectives. This helper refuses that ambiguity.
    """
    from verl.utils.device import get_device_name

    if not len(dataset) or len(dataset) % num_replicas:
        raise ValueError("Full validation requires a nonempty dataset divisible by the data parallel size")
    sampler = DistributedSampler(
        dataset, shuffle=False, num_replicas=num_replicas, rank=rank, seed=seed, drop_last=False,
    )
    loader = StatefulDataLoader(
        dataset=dataset, sampler=sampler, batch_size=1, collate_fn=collate_fn,
        num_workers=data_config.num_workers, pin_memory=False,
        pin_memory_device=get_device_name(), drop_last=False,
    )
    return sampler, loader
