import pytest
import torch

from pfns.experiments.model_benchmarks.evaluation import _set_data_generation_seed
from pfns.experiments.model_benchmarks.sampling import (
    AssociativeRecallBatchGenerator,
    ClassCoverageBatchGenerator,
)

NOTEBOOK_DATA_GENERATION_SEED = 42


def _resolve_test_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")
    return device


def _sample_one_batch(*, data_generation_seed: int, device: str):
    _set_data_generation_seed(data_generation_seed)
    get_batch = ClassCoverageBatchGenerator.create_prior_get_batch(
        num_classes=3,
        num_features=4,
        prior_type="mlp",
        device=_resolve_test_device(device),
        force_max_num_classes=True,
    )
    return get_batch(
        batch_size=1,
        seq_len=48,
        num_features=4,
        single_eval_pos=40,
        n_targets_per_input=1,
    )


def _sample_one_ar_batch(*, data_generation_seed: int, device: str):
    _set_data_generation_seed(data_generation_seed)
    generator = AssociativeRecallBatchGenerator(
        num_batches=1,
        smallest_seqlen=8,
        largest_seqlen=40,
        num_features=4,
        num_classes=3,
        number_of_test_samples=16,
        batch_device=_resolve_test_device(device),
    )
    batch, _ = generator.sample_one()
    return batch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prior_generation_is_deterministic_with_fixed_seed(device: str):
    batch_a = _sample_one_batch(
        data_generation_seed=NOTEBOOK_DATA_GENERATION_SEED,
        device=device,
    )
    batch_b = _sample_one_batch(
        data_generation_seed=NOTEBOOK_DATA_GENERATION_SEED,
        device=device,
    )

    assert torch.equal(batch_a.x, batch_b.x)
    assert torch.equal(batch_a.y, batch_b.y)
    assert torch.equal(batch_a.target_y, batch_b.target_y)
    assert torch.equal(batch_a.categorical_mask, batch_b.categorical_mask)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prior_generation_changes_when_seed_changes(device: str):
    batch_a = _sample_one_batch(
        data_generation_seed=NOTEBOOK_DATA_GENERATION_SEED,
        device=device,
    )
    batch_b = _sample_one_batch(
        data_generation_seed=NOTEBOOK_DATA_GENERATION_SEED + 1,
        device=device,
    )

    assert not torch.equal(batch_a.x, batch_b.x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_associative_recall_queries_come_from_smallest_prefix(device: str):
    batch = _sample_one_ar_batch(
        data_generation_seed=NOTEBOOK_DATA_GENERATION_SEED,
        device=device,
    )
    largest_seqlen = 40
    smallest_seqlen = 8

    train_x = batch.x[0, :largest_seqlen]
    test_x = batch.x[0, largest_seqlen:]

    prefix_x = train_x[:smallest_seqlen]
    prefix_matches = (test_x[:, None, :] == prefix_x[None, :, :]).all(dim=-1).any(dim=1)
    assert torch.all(prefix_matches)
