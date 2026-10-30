"""Verify DP coverage, corresponding micro-step lengths and checkpoint replay."""

from copy import deepcopy
import random
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from qwenpt.train.sft.sampler import (
    DistributedLengthBucketBatchSampler, build_full_validation_dataloader, build_length_bucket_dataloader,
)


def _samplers(lengths, *, replicas=4, batch_size=16, micro_size=2, **kwargs):
    return [
        DistributedLengthBucketBatchSampler(
            lengths, batch_size, replicas, rank, micro_batch_size=micro_size, seed=20261020, **kwargs,
        )
        for rank in range(replicas)
    ]


def test_all_ranks_cover_each_sample_once_with_matching_micro_step_lengths():
    """Samples remain disjoint across ranks while each collective sees nearby lengths."""
    lengths = list(range(1, 129))
    orders = [list(sampler) for sampler in _samplers(lengths)]
    seen = []
    for rank_batches in zip(*orders, strict=True):
        global_indices = [item for batch in rank_batches for item in batch]
        assert len(global_indices) == len(set(global_indices)) == 16
        seen.extend(global_indices)
        for offset in range(0, len(rank_batches[0]), 2):
            wave = [lengths[item] for batch in rank_batches for item in batch[offset : offset + 2]]
            assert max(wave) - min(wave) == 7
    assert sorted(seen) == list(range(128))


def test_epoch_and_seed_repeat_without_changing_global_rngs():
    """Reproducibility includes shuffling bucket order and sample assignments to ranks."""
    before_python, before_torch = random.getstate(), torch.random.get_rng_state().clone()
    sampler = _samplers(list(range(1, 129)))[0]
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(3)
    changed = list(sampler)
    repeat = _samplers(list(range(1, 129)))[0]
    repeat.set_epoch(3)
    assert changed == list(repeat)
    assert changed != first
    different = DistributedLengthBucketBatchSampler(list(range(1, 129)), 16, 4, 0, micro_batch_size=2, seed=7)
    assert first != list(different)
    assert random.getstate() == before_python
    assert torch.equal(torch.random.get_rng_state(), before_torch)


def test_drop_last_omits_only_a_random_tail_without_repeating_samples():
    """Tail dropping must not systematically remove the longest examples."""
    samplers = _samplers(list(range(1, 72)))
    seen = [item for sampler in samplers for batch in sampler for item in batch]
    assert len(seen) == len(set(seen)) == 64
    missing = set(range(71)) - set(seen)
    assert missing != set(range(64, 71))
    for sampler in samplers:
        sampler.set_epoch(1)
    next_seen = {item for sampler in samplers for batch in sampler for item in batch}
    assert set(range(71)) - next_seen != missing


def test_non_dropped_tail_preserves_complete_dp_micro_steps():
    """An equal partial batch is allowed; uneven rank padding is never implicit."""
    samplers = _samplers(list(range(1, 41)), drop_last=False)
    orders = [list(sampler) for sampler in samplers]
    assert all(len(sampler) == 3 for sampler in samplers)
    assert sorted(len(batch) for batch in orders[0]) == [2, 4, 4]
    assert sorted(item for batches in orders for batch in batches for item in batch) == list(range(40))
    with pytest.raises(ValueError, match="incomplete DP micro-step"):
        _samplers(list(range(1, 42)), drop_last=False)


@pytest.mark.parametrize("batch_size,micro_size", [(6, 1), (24, 1), (24, 2)])
def test_explicit_padding_covers_all_100k_rows_once_on_six_ranks(batch_size, micro_size):
    """Only internal sentinels fill the 100,000-row dataset's incomplete DP wave."""
    lengths = [(item * 3571) % 231346 + 1 for item in range(100000)]
    samplers = _samplers(
        lengths, replicas=6, batch_size=batch_size, micro_size=micro_size,
        drop_last=False, pad_tail=True,
    )
    orders = [list(sampler) for sampler in samplers]
    real_seen, padding_count = [], 0
    for rank_batches in zip(*orders, strict=True):
        assert len({len(batch) for batch in rank_batches}) == 1
        assert len(rank_batches[0]) % micro_size == 0
        for batch in rank_batches:
            padding_count += batch.count(100000)
            real_seen.extend(item for item in batch if item != 100000)
    assert sorted(real_seen) == list(range(100000))
    assert padding_count == samplers[0].num_padding_samples == (-100000) % (6 * micro_size)
    assert len(orders[0]) == len(samplers[0]) == (100000 + batch_size - 1) // batch_size
    assert sum(lengths[item] for item in real_seen) == sum(lengths)


def test_tail_padding_requires_explicit_no_drop_and_distinct_checkpoint_contract():
    """Opting in cannot silently retain dropping or reuse an unpadded checkpoint."""
    with pytest.raises(ValueError, match="drop_last=False"):
        _samplers([1] * 16, pad_tail=True)
    ordinary = _samplers([1] * 16, drop_last=False)[0]
    padded = _samplers([1] * 16, drop_last=False, pad_tail=True)[0]
    with pytest.raises(ValueError, match="does not match"):
        iter(padded).load_state_dict(iter(ordinary).state_dict())


@pytest.mark.parametrize("epoch", [0, 3])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_loader_resumes_exactly_with_prefetched_batches(epoch, num_workers):
    """Resume the consumed cursor, including when workers have prefetched future batches."""
    sampler = _samplers(list(range(1, 129)))[0]
    sampler.set_epoch(epoch)
    reference = list(sampler)
    loader = StatefulDataLoader(list(range(128)), batch_sampler=sampler, num_workers=num_workers)
    iterator = iter(loader)
    consumed = [next(iterator).tolist() for _ in range(3)]
    state = deepcopy(loader.state_dict())
    restored_sampler = _samplers(list(range(1, 129)))[0]
    restored = StatefulDataLoader(list(range(128)), batch_sampler=restored_sampler, num_workers=num_workers)
    restored.load_state_dict(state)
    # Native SFT sets the sampler's epoch after loading the checkpoint and before iterating.
    restored_sampler.set_epoch(epoch)
    assert consumed + [batch.tolist() for batch in restored] == reference
    list(iterator)


def test_iterator_resume_restores_epoch_and_rejects_changed_contracts():
    """A new sampler cannot silently apply a checkpoint to other lengths or DP topology."""
    sampler = _samplers(list(range(1, 129)))[0]
    sampler.set_epoch(4)
    iterator = iter(sampler)
    next(iterator)
    state = deepcopy(iterator.state_dict())
    expected = list(iterator)
    restored = iter(_samplers(list(range(1, 129)))[0])
    restored.load_state_dict(state)
    assert list(restored) == expected
    for incompatible in (
        _samplers([2] + list(range(2, 129)))[0],
        _samplers(list(range(1, 129)))[1],
        _samplers(list(range(1, 129)), replicas=2)[0],
    ):
        with pytest.raises(ValueError, match="does not match"):
            iter(incompatible).load_state_dict(state)


@pytest.mark.parametrize("kwargs", [
    {"global_batch_size": 15}, {"micro_batch_size": 3}, {"num_replicas": 0},
    {"rank": 4}, {"bucket_size_multiplier": 0}, {"lengths": [1, 0]},
])
def test_invalid_batch_contract_is_rejected(kwargs):
    """Reject layouts that cannot produce complete equal micro-steps across all ranks."""
    args = dict(lengths=[1] * 64, global_batch_size=16, num_replicas=4, rank=0, micro_batch_size=2)
    with pytest.raises(ValueError):
        DistributedLengthBucketBatchSampler(**(args | kwargs))


class _Dataset:
    def __init__(self, cached=False):
        self.calls = []
        self.rows = [{"total_tokens": 100 + item} for item in range(16)]
        if cached:
            self.token_lengths = list(range(1, 17))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        self.calls.append(item)
        return {"input_ids": torch.arange(item + 1)}


class _TailDataset:
    def __init__(self, count=22):
        self.count = count
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.token_lengths = [item % 7 + 3 for item in range(count)]

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        length = self.token_lengths[item]
        return {
            "input_ids": torch.full((length,), item + 1, dtype=torch.long),
            "position_ids": torch.arange(length),
            "loss_mask": torch.tensor([0, 0] + [1] * (length - 2)),
        }


def _tail_indices(batch):
    return [int(item["input_ids"][0]) - 1 for item in batch]


def _config(**overrides):
    return OmegaConf.create({
        "train_batch_size": 8, "micro_batch_size_per_gpu": 2, "num_workers": 0,
        "use_dynamic_bsz": False, **overrides,
    })


@pytest.mark.parametrize("cached", [False, True])
def test_loader_prefers_validated_lengths_and_otherwise_tokenizes_each_sample(cached):
    """Raw producer total_tokens must not become the default bucketing source."""
    dataset = _Dataset(cached)
    sampler, loader = build_length_bucket_dataloader(
        dataset, _config(), collate_fn=None, num_replicas=2, rank=0, seed=7,
    )
    assert sampler.lengths == tuple(range(1, 17))
    assert dataset.calls == ([] if cached else list(range(16)))
    assert loader.batch_sampler is sampler
    assert loader.batch_size is None
    assert len(loader) == 2


def test_explicit_producer_lengths_and_dynamic_reference_micro_steps():
    """Dynamic grouping remains in the engine, using a compatible DP bucket schedule."""
    sampler, _ = build_length_bucket_dataloader(
        _Dataset(), _config(train_batch_size=6, use_dynamic_bsz=True, length_source="total_tokens"),
        collate_fn=None, num_replicas=2, rank=0, seed=7,
    )
    assert sampler.lengths == tuple(range(100, 116))
    assert sampler.micro_batch_size == 1
    assert len(sampler) == 2


def test_malformed_cache_or_unknown_source_is_rejected():
    """Fail before training when cache alignment or length provenance is invalid."""
    dataset = _Dataset(cached=True)
    dataset.token_lengths.pop()
    for source, pattern in (("tokenized", "one length per"), ("untrusted", "length_source")):
        with pytest.raises(ValueError, match=pattern):
            build_length_bucket_dataloader(
                dataset, _config(length_source=source), collate_fn=None, num_replicas=2, rank=0, seed=7,
            )


def test_padded_loader_preserves_every_real_tensor_and_supervision():
    """The wrapper changes no real row and adds only explicitly zero-loss tokens."""
    dataset = _TailDataset(22)
    config = _config(
        train_batch_size=6, micro_batch_size_per_gpu=1,
        length_bucket_drop_last=False, length_bucket_pad_tail=True,
    )
    seen, supervised, input_tokens, padding_count = [], 0, 0, 0
    for rank in range(6):
        sampler, loader = build_length_bucket_dataloader(
            dataset, config, collate_fn=list, num_replicas=6, rank=rank, seed=7,
        )
        assert len(dataset) == len(loader.dataset) == 22
        assert sampler.num_padding_samples == 2
        for batch in loader:
            for row in batch:
                if not row["loss_mask"].any():
                    padding_count += 1
                    assert row["input_ids"].tolist() == [0] * 16
                    assert row["position_ids"].tolist() == list(range(16))
                    continue
                index = int(row["input_ids"][0]) - 1
                seen.append(index)
                for key, value in dataset[index].items():
                    assert torch.equal(row[key], value)
                supervised += int(row["loss_mask"].sum())
                input_tokens += row["input_ids"].numel()
    assert sorted(seen) == list(range(22))
    assert padding_count == 2
    assert supervised == sum(length - 2 for length in dataset.token_lengths)
    assert input_tokens == sum(dataset.token_lengths)


def test_tail_padding_preserves_processor_positions_and_has_no_media():
    """Conditional-generation loaders retain their mRoPE dimensions for sentinels."""
    dataset = _TailDataset(4)
    dataset.processor = object()

    def positions(inputs):
        assert inputs["attention_mask"].tolist() == [1] * 16
        return torch.arange(16).repeat(4, 1)

    dataset.position_ids = positions
    config = _config(
        train_batch_size=6, micro_batch_size_per_gpu=1,
        length_bucket_drop_last=False, length_bucket_pad_tail=True,
    )
    _, loader = build_length_bucket_dataloader(
        dataset, config, collate_fn=list, num_replicas=6, rank=0, seed=7,
    )
    padding = loader.dataset[len(dataset)]
    assert padding["position_ids"].shape == (4, 16)
    assert padding["multi_modal_inputs"] == {}


@pytest.mark.parametrize("dynamic", [False, True])
def test_partial_global_batch_survives_native_microbatching(dynamic):
    """Native engine batching accepts a short optimizer batch with equal DP counts."""
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from verl.workers.engine.utils import prepare_micro_batches

    dataset = _TailDataset(28)
    config = _config(
        train_batch_size=24, micro_batch_size_per_gpu=1, use_dynamic_bsz=dynamic,
        length_bucket_drop_last=False, length_bucket_pad_tail=True,
    )
    for rank in range(6):
        _, loader = build_length_bucket_dataloader(
            dataset, config, collate_fn=SFTTensorCollator("no_padding"), num_replicas=6, rank=rank, seed=7,
        )
        batch_sizes = []
        for batch in loader:
            data = tu.get_tensordict(batch, non_tensor_dict={
                "use_dynamic_bsz": dynamic, "max_token_len_per_gpu": 64,
                "micro_batch_size_per_gpu": 1, "global_batch_size": 24, "sp_size": 1,
            })
            batch_sizes.append(len(data))
            micro_batches, _ = prepare_micro_batches(data)
            assert sum(len(micro) for micro in micro_batches) == len(data)
            assert sum(micro["loss_mask"].sum().item() for micro in micro_batches) == data["loss_mask"].sum().item()
        assert sorted(batch_sizes) == [1, 4]


@pytest.mark.parametrize("consumed", [2, 4])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_padded_loader_resume_preserves_tail_and_next_epoch(consumed, num_workers):
    """Resume exactly around a padded epoch, including worker prefetch cursors."""
    dataset = _TailDataset()
    config = _config(
        train_batch_size=6, micro_batch_size_per_gpu=1, num_workers=num_workers,
        length_bucket_drop_last=False, length_bucket_pad_tail=True,
    )
    sampler, loader = build_length_bucket_dataloader(
        dataset, config, collate_fn=_tail_indices, num_replicas=6, rank=3, seed=7,
    )
    sampler.set_epoch(2)
    expected = [[index if index < len(dataset) else -1 for index in batch] for batch in sampler]
    iterator = iter(loader)
    actual = [next(iterator) for _ in range(consumed)]
    assert actual == expected[:consumed]
    state = deepcopy(loader.state_dict())
    restored_sampler, restored = build_length_bucket_dataloader(
        dataset, config, collate_fn=_tail_indices, num_replicas=6, rank=3, seed=7,
    )
    restored.load_state_dict(state)
    next_epoch = 3 if consumed == len(loader) else 2
    restored_sampler.set_epoch(next_epoch)
    next_order = [[index if index < len(dataset) else -1 for index in batch] for batch in restored_sampler]
    assert list(restored) == (next_order if next_epoch == 3 else expected[consumed:])
    list(iterator)


def test_zero_loss_tail_matches_native_global_token_mean_and_all_gradients():
    """Simulated DP averaging retains the four real samples' complete objective."""
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorData
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from verl.workers.utils.losses import sft_loss

    torch.manual_seed(73)
    dataset = _TailDataset(4)
    config = _config(
        train_batch_size=6, micro_batch_size_per_gpu=1,
        length_bucket_drop_last=False, length_bucket_pad_tail=True,
    )
    weight = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    embedding = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    global_tokens = sum(length - 2 for length in dataset.token_lengths)
    local_losses = []
    all_real_terms = []
    for rank in range(6):
        _, loader = build_length_bucket_dataloader(
            dataset, config, collate_fn=SFTTensorCollator("no_padding"), num_replicas=6, rank=rank, seed=7,
        )
        batch = next(iter(loader))
        item_ids = batch["input_ids"].unbind()[0]
        logits = embedding[item_ids] @ weight.T
        log_probs = logits.log_softmax(-1).gather(-1, item_ids.roll(-1)[:, None]).squeeze(-1)
        data = TensorDict({
            "loss_mask": batch["loss_mask"], "pad_mode": NonTensorData("no_padding"),
            "dp_size": 6, "batch_num_tokens": global_tokens,
        }, batch_size=[])
        native_loss, _ = sft_loss(
            None, {"log_probs": torch.nested.as_nested_tensor([log_probs], layout=torch.jagged)}, data,
        )
        local_losses.append(native_loss)
        if batch["loss_mask"].values().any():
            all_real_terms.append(-(log_probs[:-1] * batch["loss_mask"].values()[1:]).sum())
        else:
            assert native_loss.item() == 0
    averaged = sum(local_losses) / 6
    reference = sum(all_real_terms) / global_tokens
    torch.testing.assert_close(averaged, reference, rtol=1e-12, atol=1e-12)
    actual_grads = torch.autograd.grad(averaged, (weight, embedding), retain_graph=True)
    reference_grads = torch.autograd.grad(reference, (weight, embedding))
    for actual, expected in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def _indices(batch):
    return [len(item["input_ids"]) - 1 for item in batch]


@pytest.mark.parametrize("train_batch_size", [96, 192])
def test_full_validation_uses_all_300_rows_once_independent_of_training_batch(train_batch_size):
    """All six ranks retain 50 validation rows even when 300 is not divisible by GBS."""
    dataset = _TailDataset(300)
    seen = []
    for rank in range(6):
        sampler, loader = build_full_validation_dataloader(
            dataset, _config(train_batch_size=train_batch_size), collate_fn=_tail_indices,
            num_replicas=6, rank=rank, seed=7,
        )
        assert not sampler.shuffle and not sampler.drop_last
        assert loader.batch_size == 1 and not loader.drop_last
        assert len(loader) == 50
        batches = list(loader)
        assert all(len(batch) == 1 for batch in batches)
        seen.extend(batch[0] for batch in batches)
    assert sorted(seen) == list(range(300))


def test_full_validation_refuses_implicit_repeated_rows():
    """Uneven validation partitions require a separate explicit policy."""
    with pytest.raises(ValueError, match="divisible"):
        build_full_validation_dataloader(
            _TailDataset(301), _config(), collate_fn=_tail_indices, num_replicas=6, rank=0, seed=7,
        )


@pytest.mark.parametrize("consumed", [1, 2])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_loader_checkpoint_handles_prefetch_and_native_epoch_boundary(consumed, num_workers):
    """A final-batch checkpoint starts the next epoch instead of skipping it."""
    config = _config(num_workers=num_workers)
    sampler, loader = build_length_bucket_dataloader(
        _Dataset(cached=True), config, collate_fn=_indices, num_replicas=2, rank=0, seed=7,
    )
    sampler.set_epoch(2)
    expected = list(sampler)
    iterator = iter(loader)
    for _ in range(consumed):
        next(iterator)
    state = deepcopy(loader.state_dict())
    restored_sampler, restored = build_length_bucket_dataloader(
        _Dataset(cached=True), config, collate_fn=_indices, num_replicas=2, rank=0, seed=7,
    )
    restored.load_state_dict(state)
    next_epoch = 3 if consumed == len(loader) else 2
    restored_sampler.set_epoch(next_epoch)
    assert list(restored) == (list(restored_sampler) if next_epoch == 3 else expected[consumed:])
    list(iterator)


def test_native_entrypoint_replaces_only_opted_in_training_loader(monkeypatch, tmp_path):
    """The native trainer retains its validation sampler and training lifecycle."""
    from torch.utils.data import DistributedSampler
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils import distributed
    from qwenpt.train.sft.entrypoint import run_sft

    captured = []

    def initialize(self, config):
        self.config = config
        self.engine = SimpleNamespace(get_data_parallel_rank=lambda: 0, get_data_parallel_size=lambda: 2)
        self.train_dataset = _Dataset(cached=True)
        self.val_dataset = _Dataset(cached=True)
        self._build_dataloader()
        self.training_client = SimpleNamespace(train_batch=lambda **kwargs: None)
        captured.append(self)

    monkeypatch.setattr(SFTTrainer, "__init__", initialize)
    monkeypatch.setattr(SFTTrainer, "fit", lambda self: None)
    monkeypatch.setattr(distributed, "initialize_global_process_group", lambda: None)
    monkeypatch.setattr(distributed, "destroy_global_process_group", lambda: None)
    monkeypatch.setenv("RANK", "0")
    config = OmegaConf.create({
        "trainer": {"seed": 7, "device": "cpu", "default_local_dir": str(tmp_path)},
        "data": dict(_config(pad_mode="no_padding", length_bucketing=True)),
    })
    run_sft(config)
    trainer = captured[0]
    assert isinstance(trainer.train_sampler, DistributedLengthBucketBatchSampler)
    assert trainer.train_dataloader.batch_sampler is trainer.train_sampler
    assert isinstance(trainer.val_sampler, DistributedSampler)
    assert trainer.train_sampler.seed == trainer.val_sampler.seed == 7
