from __future__ import annotations

from typing import Any

import torch

from pfns.priors.prior import Batch


def generate_associative_recall_batch(
    *,
    batch_size: int,
    largest_seqlen: int,
    smallest_seqlen: int,
    num_features: int,
    num_classes: int,
    number_of_test_samples: int,
    batch_device: str = "cpu",
    **kwargs,
) -> Batch:
    """Generate associative-recall key-value/query data.

    This is the single source of truth for AR data generation and is reused by:
    - prior training (`get_batch` below)
    - benchmark sequence-length evaluation (`sampling.py`)
    """

    assert min(batch_size, smallest_seqlen, num_features, number_of_test_samples) >= 1, (
        "batch_size, smallest_seqlen, num_features, and "
        "number_of_test_samples must be >= 1."
    )
    assert num_classes >= 2, "num_classes must be >= 2."
    assert largest_seqlen >= smallest_seqlen, "largest_seqlen must be >= smallest_seqlen."

    key_vectors = torch.randn(
        batch_size,
        largest_seqlen,
        num_features,
        device=batch_device,
        dtype=torch.float32,
    )
    value_labels = torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size, largest_seqlen),
        device=batch_device,
        dtype=torch.int64,
    )
    query_indices = torch.randint(
        low=0,
        high=smallest_seqlen,
        size=(batch_size, number_of_test_samples),
        device=batch_device,
        dtype=torch.int64,
    )
    query_vectors = torch.gather(
        key_vectors,
        dim=1,
        index=query_indices.unsqueeze(-1).expand(-1, -1, num_features),
    )
    query_targets = torch.gather(value_labels, dim=1, index=query_indices)

    all_x = torch.cat([key_vectors, query_vectors], dim=1)
    all_y = torch.cat([value_labels, query_targets], dim=1).to(torch.float32)

    return Batch(
        x=all_x,
        y=all_y,
        target_y=all_y.clone(),
        categorical_mask=torch.zeros(
            num_features,
            dtype=torch.bool,
            device=batch_device,
        ),
    )


def get_batch(
    *,
    batch_size: int,
    seq_len: int,
    num_features: int,
    single_eval_pos: int | None,
    n_targets_per_input: int = 1,
    max_num_classes: int = 10,
    fixed_num_features: int | None = None,
    fixed_num_classes: int | None = None,
    batch_device: str = "cpu",
    task_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> Batch:
    """Generate AR batches with the same sampler used in benchmark evaluation.

    This makes AR pretraining use the existing PFNs training loop with minimal
    custom code by plugging into `AdhocPriorConfig(prior_names=['associative_recall'])`.
    """
    del kwargs  # Unused extra kwargs from dataloader pipeline.

    assert n_targets_per_input == 1, "associative_recall prior only supports n_targets_per_input=1"
    assert batch_size >= 1, "batch_size must be >= 1"
    assert seq_len >= 2, "seq_len must be >= 2"
    if fixed_num_features is not None:
        assert int(fixed_num_features) >= 1, "fixed_num_features must be >= 1."
        num_features = int(fixed_num_features)
    if fixed_num_classes is not None:
        assert int(fixed_num_classes) >= 2, "fixed_num_classes must be >= 2."
        max_num_classes = int(fixed_num_classes)

    if single_eval_pos is None:
        single_eval_pos = max(1, int(0.8 * seq_len))
    single_eval_pos = int(single_eval_pos)
    single_eval_pos = max(1, min(single_eval_pos, seq_len - 1))
    num_test_samples = int(seq_len - single_eval_pos)

    options = dict(task_kwargs or {})
    options.setdefault("batch_device", batch_device)
    batch = generate_associative_recall_batch(
        batch_size=batch_size,
        largest_seqlen=single_eval_pos,
        smallest_seqlen=single_eval_pos,
        num_features=int(num_features),
        num_classes=int(max_num_classes),
        number_of_test_samples=num_test_samples,
        **options,
    )
    batch.single_eval_pos = single_eval_pos
    return batch
