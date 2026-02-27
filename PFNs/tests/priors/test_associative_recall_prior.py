import pytest
import torch

from pfns.priors.associative_recall import generate_associative_recall_batch, get_batch


def _resolve_test_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")
    return device


def _oracle_accuracy_from_context(batch, single_eval_pos: int) -> float:
    train_x = batch.x[:, :single_eval_pos]
    train_y = batch.y[:, :single_eval_pos].long()
    test_x = batch.x[:, single_eval_pos:]
    test_y = batch.target_y[:, single_eval_pos:].long()

    # Exact lookup oracle: find identical key vectors in the context.
    matches = (test_x[:, :, None, :] == train_x[:, None, :, :]).all(dim=-1)
    assert torch.all(matches.any(dim=-1)), "Each query must match a context key."

    label_matches = train_y[:, None, :] == test_y[:, :, None]
    correct = (matches & label_matches).any(dim=-1)
    return float(correct.float().mean().item())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_generate_associative_recall_batch_shapes(device: str):
    resolved_device = _resolve_test_device(device)
    torch.manual_seed(0)

    batch = generate_associative_recall_batch(
        batch_size=4,
        largest_seqlen=11,
        smallest_seqlen=5,
        num_features=7,
        num_classes=6,
        number_of_test_samples=3,
        batch_device=resolved_device,
    )

    assert batch.x.shape == (4, 14, 7)
    assert batch.y.shape == (4, 14)
    assert batch.target_y.shape == (4, 14)
    assert batch.y.dtype == torch.float32
    assert batch.categorical_mask is not None
    assert batch.categorical_mask.shape == (7,)
    assert batch.categorical_mask.dtype == torch.bool


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_batch_oracle_has_perfect_accuracy(device: str):
    resolved_device = _resolve_test_device(device)
    torch.manual_seed(0)

    single_eval_pos = 32
    batch = get_batch(
        batch_size=8,
        seq_len=48,
        num_features=5,
        single_eval_pos=single_eval_pos,
        max_num_classes=10,
        batch_device=resolved_device,
    )

    assert batch.x.shape == (8, 48, 5)
    assert batch.y.shape == (8, 48)
    assert int(batch.single_eval_pos) == single_eval_pos

    oracle_acc = _oracle_accuracy_from_context(batch, single_eval_pos=single_eval_pos)
    assert oracle_acc == 1.0
