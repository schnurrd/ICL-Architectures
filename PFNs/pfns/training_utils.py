import typing as tp
import math
from dataclasses import dataclass

import einops
import torch
from torch import nn


class EpochResult(tp.NamedTuple):
    loss: float  # total loss for the epoch
    data_time: float  # time spent getting batch data
    forward_time: float  # time spent in forward pass
    step_time: float  # total time per step
    nan_share: float  # share of NaN values
    ignore_share: float  # share of ignored values (-100)
    grad_norm_ema_mean: float  # mean of grad norms for the epoch
    grad_norm_infinite_steps_fraction: float  # fraction of non-finite grad norm steps
    grad_norm_ema_exceeded_fraction: float  # fraction of steps where grad norm EMA exceeded the threshold
    importance_sampling_infos: list  # gradient magnitude info


@dataclass
class Metrics:
    steps_per_epoch: int
    total_loss: float = 0.0
    nan_steps: float = 0.0
    ignore_steps: float = 0.0
    forward_time: float = 0.0
    step_time: float = 0.0
    time_to_get_batch: float = 0.0
    grad_norm_ema: float = 0.0
    grad_norm_infinite_steps: int = 0
    grad_norm_ema_exceeded: int = 0

    @torch.no_grad()
    def update(
        self,
        loss: torch.Tensor,
        nan_share: float,
        targets: torch.Tensor,
        forward_time: float,
        step_time: float,
        time_to_get_batch: float,
        grad_norm_ema: float,
        grad_norm_infinite_steps: int,
        grad_norm_ema_exceeded: int,
    ):
        self.total_loss += loss.cpu().detach().item()

        self.nan_steps += nan_share
        self.ignore_steps += (targets == -100).float().mean()
        self.forward_time += forward_time
        self.step_time += step_time
        self.time_to_get_batch += time_to_get_batch
        self.grad_norm_ema += grad_norm_ema
        self.grad_norm_infinite_steps += grad_norm_infinite_steps
        self.grad_norm_ema_exceeded += grad_norm_ema_exceeded
        
    def get_epoch_result(self, importance_sampling_infos: list[tuple]):
        return EpochResult(
            loss=self.total_loss / self.steps_per_epoch,
            data_time=self.time_to_get_batch / self.steps_per_epoch,
            forward_time=self.forward_time / self.steps_per_epoch,
            step_time=self.step_time / self.steps_per_epoch,
            nan_share=self.nan_steps.cpu().item() / self.steps_per_epoch,
            ignore_share=self.ignore_steps.cpu().item() / self.steps_per_epoch,
            grad_norm_ema_mean=self.grad_norm_ema / self.steps_per_epoch,
            grad_norm_infinite_steps_fraction=self.grad_norm_infinite_steps / self.steps_per_epoch,
            grad_norm_ema_exceeded_fraction=self.grad_norm_ema_exceeded / self.steps_per_epoch,
            importance_sampling_infos=importance_sampling_infos,
        )


@torch.no_grad()
def compute_update_ratio(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    current_grad_norm: float,
    eps: float = 1e-5,
) -> float:
    lr = optimizer.param_groups[0]["lr"]

    update_norm = lr * current_grad_norm
    param_norm_sq = torch.tensor(0.0, device=next(model.parameters()).device)

    for param in model.parameters():
        if param.grad is not None:
            param_norm_sq += param.pow(2).sum()

    param_norm = param_norm_sq.sqrt().item()
    return update_norm / (param_norm + eps)

@torch.no_grad()
def update_importance_sampling_infos(
    importance_sampling_infos: list,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    info_used_with_gradient_magnitudes: list,
):
    squared_grad_magnitudes = {
        name: (w.grad**2).sum().cpu().item()
        for name, w in model.named_parameters()
        if w.grad is not None
    }
    total_grad_magnitude = sum(squared_grad_magnitudes.values())

    normalized_squared_grad_magnitudes = {}
    total_normalized_grad_magnitude = None
    # Compute grad magnitude normalized by Adam's beta2 parameter if Adam optimizer is used
    if squared_grad_magnitudes and isinstance(
        optimizer, (torch.optim.Adam, torch.optim.AdamW)
    ):
        beta2 = optimizer.param_groups[0]["betas"][1]
        # Get the current state of Adam's running average of squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                state = optimizer.state.get(param, {})
                if "exp_avg_sq" in state:
                    # Normalize the squared gradient by the running average
                    normalized_grad_magnitude = (
                        (
                            (param.grad**2)
                            / (
                                state["exp_avg_sq"]
                                * (1 - beta2 ** state.get("step", 1))
                                + 1e-8
                            )
                        )
                        .sum()
                        .cpu()
                        .item()
                    )
                    normalized_squared_grad_magnitudes[name] = normalized_grad_magnitude
        total_normalized_grad_magnitude = sum(
            v for k, v in normalized_squared_grad_magnitudes.items()
        )

    importance_sampling_infos.append(
        (
            total_grad_magnitude,
            info_used_with_gradient_magnitudes,
            loss,
            total_normalized_grad_magnitude,
        )
    )


def set_model_to(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mode: tp.Literal["train", "eval"],
):
    assert mode in [
        "train",
        "eval",
    ], f"mode must be 'train' or 'eval', got {mode}"
    if mode == "train":
        model.train()
        if hasattr(optimizer, "train"):
            optimizer.train()
    else:
        model.eval()
        if hasattr(optimizer, "eval"):
            optimizer.eval()


def move_y_style_and_check_shape(
    y_style: torch.Tensor | None, y: torch.Tensor, device: torch.device
) -> torch.Tensor | None:
    y_style = y_style.to(device) if y_style is not None else None
    if y_style is not None:
        if y_style.dim() == 2:
            broken = y_style.shape[0] != y.shape[0]
        else:
            raise ValueError(f"y_style must have 2 dimensions, got {y_style.shape}")
        if broken:
            raise ValueError(
                f"y_style must have the same batch size as y, got {y_style.shape=} "
                f"and {y.shape=}"
            )
    return y_style


def move_style_and_check_shape(
    style: torch.Tensor | None, x: torch.Tensor, device: torch.device
) -> torch.Tensor | None:
    style = style.to(device) if style is not None else None
    if style is not None:
        if style.dim() == 2:
            broken = style.shape[0] != x.shape[0]
        elif style.dim() == 3:
            broken = style.shape[0] != x.shape[0] or style.shape[1] != x.shape[2]
        else:
            raise ValueError(f"style must have 2 or 3 dimensions, got {style.shape}")
        if broken:
            raise ValueError(
                f"style must have the same batch size as x and if it has 3 dimensions, "
                f"the middle dimension must match the number of features, got {style.shape=} "
                f"and {x.shape=}"
            )
    return style


def categorical_mask_to_inds(
    categorical_mask: torch.Tensor | None
) -> list[int] | None:
    if categorical_mask is None:
        return None

    mask = categorical_mask
    if mask.ndim > 1:
        if not torch.all(mask == mask[0]):
            raise NotImplementedError("Per-sample categorical masks are not supported.")
        mask = mask[0]
    return torch.nonzero(mask, as_tuple=True)[0].tolist()


def compute_losses(
    output: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
    n_targets_per_input: int,
) -> torch.Tensor:
    """Compute per-sequence losses with support for multiple targets per input."""
    output = output.unsqueeze(2).expand(
        *output.shape[:2],
        n_targets_per_input,
        output.shape[-1],
    )

    if len(targets.shape) == 2:
        targets = targets.unsqueeze(2)

    assert targets.shape == output.shape[:-1], (
        f"Target shape {targets.shape} "
        f"does not match output shape {output.shape}."
        f"This might be because you are missing trailing "
        "1 dimension in the target."
    )

    output = einops.rearrange(output, "b s t l -> (b t) s l")
    targets = einops.rearrange(targets, "b s t -> (b t) s")

    if isinstance(criterion, nn.GaussianNLLLoss):
        assert (
            output.shape[-1] == 2
        ), "need to write a little bit of code to handle multiple regression targets at once"

        mean_pred = output[..., 0]
        var_pred = output[..., 1].abs()
        losses = criterion(
            mean_pred.flatten(),
            targets.flatten(),
            var=var_pred.flatten(),
        )
    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
        targets[torch.isnan(targets)] = -100
        losses = criterion(output.flatten(), targets.flatten())
        losses = losses.view(*targets.shape)
    elif isinstance(criterion, nn.CrossEntropyLoss):
        targets[torch.isnan(targets)] = -100
        losses = criterion(
            output.reshape(-1, len(criterion.weight)),
            targets.long().flatten(),
        )
        losses = losses.view(*targets.shape)
    else:
        losses = criterion(output, targets.unsqueeze(-1))
    losses = einops.rearrange(losses, "(b t) s -> b s t", t=n_targets_per_input)
    losses = losses.mean(-1)
    return losses



def resolve_autocast_dtype(device: str, dtype_spec: str | None) -> torch.dtype:
    dtype_spec = (dtype_spec or "auto").lower()
    if dtype_spec in ("fp16", "float16"):
        return torch.float16
    if dtype_spec in ("bf16", "bfloat16"):
        if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise ValueError(
                "Requested bf16 autocast but CUDA device does not support bf16."
            )
        return torch.bfloat16
    if dtype_spec in ("fp32", "float32"):
        return torch.float32
    if dtype_spec == "auto":
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    raise ValueError(
        f"Unsupported train_mixed_precision_dtype '{dtype_spec}'. "
        "Use 'auto', 'bf16', 'fp16', or 'fp32'."
    )


def is_autocast_dtype_enabled(dtype: torch.dtype | None) -> bool:
    return dtype in (torch.float16, torch.bfloat16)
