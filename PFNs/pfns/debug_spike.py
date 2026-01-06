from __future__ import annotations

import os
import typing as tp

import torch
from torch import nn


def _tensor_stats(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach()
    finite = torch.isfinite(tensor)
    nan_pct = (~finite & torch.isnan(tensor)).float().mean().item() * 100.0
    inf_pct = (~finite & torch.isinf(tensor)).float().mean().item() * 100.0
    if finite.any():
        finite_tensor = tensor[finite]
        return (
            f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"min={finite_tensor.min().item():.4g} max={finite_tensor.max().item():.4g} "
            f"mean={finite_tensor.mean().item():.4g} std={finite_tensor.std().item():.4g} "
            f"nan%={nan_pct:.2f} inf%={inf_pct:.2f}"
        )
    return (
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"all-nonfinite nan%={nan_pct:.2f} inf%={inf_pct:.2f}"
    )


def log_loss_spike(
    *,
    loss: torch.Tensor,
    nan_share: float,
    threshold: float,
    single_eval_pos: int | None,
    n_targets_per_input: int,
    categorical_inds: list[int] | None,
    batch: tp.Any,
    targets: torch.Tensor,
    output: torch.Tensor,
    criterion: nn.Module,
    training: bool,
    rank: int,
    train_state_dict_save_path: str | None,
    spike_save_count: int,
    spike_save_max: int,
    epoch: int,
    batch_index: int,
    device: str,
) -> int:
    loss_is_finite = torch.isfinite(loss).item()
    if loss_is_finite and loss.item() <= threshold:
        return spike_save_count

    print("Loss spike detected")
    print(f"loss={loss.item():.6g} nan_share={nan_share:.4g}")
    print(f"single_eval_pos={single_eval_pos} n_targets_per_input={n_targets_per_input}")
    print(f"categorical_inds={categorical_inds}")
    print(_tensor_stats("batch.x", batch.x))
    print(_tensor_stats("batch.y", batch.y))
    print(_tensor_stats("targets", targets))
    print(_tensor_stats("output", output))

    if isinstance(criterion, nn.CrossEntropyLoss):
        targets_flat = targets.detach().reshape(-1)
        valid_targets = targets_flat[targets_flat != -100]
        if valid_targets.numel() > 0:
            print(
                f"targets: min={valid_targets.min().item()} "
                f"max={valid_targets.max().item()} "
                f"ignore%={float((targets_flat == -100).float().mean().item() * 100):.2f}"
            )
            num_classes = output.shape[-1]
            class_counts = torch.bincount(
                valid_targets.long(),
                minlength=num_classes,
            )
            print(f"class_counts={class_counts.tolist()}")
            with torch.no_grad():
                probs = torch.softmax(
                    output.detach().reshape(-1, num_classes),
                    dim=-1,
                )
                max_prob = probs.max(dim=-1).values
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
            print(
                f"probs: max_prob mean={max_prob.mean().item():.4g} "
                f"max={max_prob.max().item():.4g} "
                f"entropy mean={entropy.mean().item():.4g}"
            )

    if (
        training
        and rank == 0
        and train_state_dict_save_path is not None
        and spike_save_count < spike_save_max
    ):
        checkpoint_dir = os.path.dirname(train_state_dict_save_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(
            checkpoint_dir,
            f"spike_epoch{epoch}_step{batch_index}.pt",
        )
        spike_payload = {
            "epoch": epoch,
            "batch_index": batch_index,
            "loss": loss.detach().cpu().item(),
            "nan_share": float(nan_share),
            "single_eval_pos": single_eval_pos,
            "n_targets_per_input": n_targets_per_input,
            "categorical_mask": getattr(batch, "categorical_mask", None),
            "info_used_with_gradient_magnitudes": getattr(
                batch, "info_used_with_gradient_magnitudes", None
            ),
            "x": batch.x.detach().cpu(),
            "y": batch.y.detach().cpu(),
            "target_y": batch.target_y.detach().cpu(),
            "style": None if batch.style is None else batch.style.detach().cpu(),
            "y_style": None if batch.y_style is None else batch.y_style.detach().cpu(),
            "rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if device.startswith("cuda")
            else None,
        }
        torch.save(spike_payload, save_path)
        spike_save_count += 1
        print(f"Saved spike batch to {save_path} count number {spike_save_count}")

    return spike_save_count
