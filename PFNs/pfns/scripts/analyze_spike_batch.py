from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from pfns.train import compute_losses, load_config
from pfns.training_utils import move_style_and_check_shape, move_y_style_and_check_shape
from pfns.utils import strip_compiled_state_dict_prefix


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


def _per_class_loss(
    output: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> dict[int, float]:
    # output: (b, s, c), targets: (b, s)
    targets_flat = targets.reshape(-1)
    valid_mask = targets_flat != -100
    if valid_mask.sum() == 0:
        return {}
    logits = output.view(-1, num_classes)[valid_mask]
    t = targets_flat[valid_mask].long()
    log_probs = torch.log_softmax(logits, dim=-1)
    losses = -log_probs[torch.arange(t.numel(), device=t.device), t]
    sums = defaultdict(float)
    counts = defaultdict(int)
    for cls, loss_val in zip(t.tolist(), losses.detach().tolist()):
        sums[int(cls)] += float(loss_val)
        counts[int(cls)] += 1
    return {k: sums[k] / counts[k] for k in sorted(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a spike batch.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--spike", required=True, help="Path to spike batch .pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K grad norm layers")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(args.checkpoint)
    model = config.model.create_model()
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError:
        stripped_state_dict, prefix = strip_compiled_state_dict_prefix(
            checkpoint["model_state_dict"]
        )
        if prefix is None:
            raise
        print(f"Detected compiled model weights. Stripped '{prefix}' from state_dict keys.")
        model.load_state_dict(stripped_state_dict, strict=True)
    model.to(args.device)
    model.train()

    spike = torch.load(args.spike, map_location="cpu")
    x = spike["x"].to(args.device)
    y = spike["y"].to(args.device)
    target_y = spike["target_y"].to(args.device)
    single_eval_pos = spike.get("single_eval_pos")
    categorical_mask = spike.get("categorical_mask")
    style = spike.get("style")
    y_style = spike.get("y_style")

    categorical_inds = None
    if categorical_mask is not None:
        mask = categorical_mask
        if mask.ndim > 1:
            if not torch.all(mask == mask[0]):
                raise ValueError("Per-sample categorical masks not supported.")
            mask = mask[0]
        categorical_inds = torch.nonzero(mask, as_tuple=True)[0].tolist()

    print(_tensor_stats("x", x))
    print(_tensor_stats("y", y))
    print(_tensor_stats("target_y", target_y))
    print(f"single_eval_pos={single_eval_pos}")
    print(f"categorical_inds={categorical_inds}")
    
    if single_eval_pos is not None:
        targets = target_y[:, single_eval_pos:]
    else:
        targets = target_y
    
    output = model(
        x=x,
        y=y[:, :single_eval_pos].to(args.device) if single_eval_pos is not None else y,
        style=move_style_and_check_shape(style, x, args.device),
        y_style=move_y_style_and_check_shape(y_style, y, args.device),
        categorical_inds=categorical_inds,
        only_return_standard_out=True,
    )

    print(_tensor_stats("output", output))

    losses = compute_losses(
        output, targets, model.criterion, config.n_targets_per_input
    )
    loss = losses.mean()
    print(f"mean_loss={loss.item():.6g}")

    if isinstance(model.criterion, torch.nn.CrossEntropyLoss):
        num_classes = output.shape[-1]
        class_counts = torch.bincount(
            targets.reshape(-1).clamp_min(0).long(), minlength=num_classes
        )
        print(f"class_counts={class_counts.tolist()}")
        per_class = _per_class_loss(output, targets, num_classes)
        if per_class:
            print("per_class_loss:")
            for cls, val in per_class.items():
                print(f"  class={cls}: {val:.6g}")

    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))
    grad_norms.sort(key=lambda x: x[1], reverse=True)
    print(f"top_{args.top_k}_grad_norms:")
    for name, val in grad_norms[: args.top_k]:
        print(f"  {name}: {val:.6g}")


if __name__ == "__main__":
    main()
