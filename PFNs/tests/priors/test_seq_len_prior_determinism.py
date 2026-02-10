import pytest
import torch

from pfns.experiments.model_benchmarks.evaluation import _set_data_generation_seed
from pfns.experiments.model_benchmarks.sampling import ClassCoverageBatchGenerator

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
