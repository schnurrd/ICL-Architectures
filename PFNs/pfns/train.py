from __future__ import annotations

import importlib
import math

import os
import time
import typing as tp
from contextlib import nullcontext
from dataclasses import dataclass

import einops
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from . import base_config, utils
from .batch_shape_sampler import BatchShapeSamplerConfig
from .model.transformer_config import ModelConfig
from .optimizer import OptimizerConfig
from .run_logger import NullRunManager, RunManager, WandbConfig

from .priors import data_loading, prior, utils as priors_utils

from .training_utils import (
    Metrics,
    move_style_and_check_shape,
    move_y_style_and_check_shape,
    set_model_to,
    update_importance_sampling_infos,
    compute_update_ratio
)
from .utils import get_cosine_schedule_with_warmup, init_dist


@dataclass(frozen=True)
class MainConfig(base_config.BaseConfig):
    # Training configuration
    priors: tp.List[prior.PriorConfig]
    optimizer: OptimizerConfig

    # Model (includes criterion)
    model: ModelConfig

    # Training
    batch_shape_sampler: BatchShapeSamplerConfig # samples num_features and single_eval_pos per batch
    epochs: int = 10
    steps_per_epoch: int = 100 # number of steps that make up one epoch
    aggregate_k_gradients: int = 1 # for gradient accumulation
    n_targets_per_input: int = 1 # how many targets to sample per input during training
    train_mixed_precision: bool = True
    train_mixed_precision_dtype: str | None = "fp16"
    skip_grad_norm_spike_factor: float = 5.0  # skip step if grad norm > factor * grad_norm_ema

    # LR Scheduler
    scheduler: str = "cosine_decay"
    warmup_epochs: int = 10

    # Checkpointing
    train_state_dict_save_path: tp.Optional[str] = None
    train_state_dict_load_path: tp.Optional[str] = None

    # Validation
    test_priors: tp.List[prior.PriorConfig] | None = None
    validation_period: int | None = None

    # Logging
    verbose: bool = True
    progress_bar: bool = False
    wandb: WandbConfig | None = None
    wandb_run_id: str | None = None

    # Data loading
    dataloader_class: str | None = None
    num_workers: tp.Optional[int] = None

    # Debugging
    debug_spike_save_path: str | None = None
    debug_spike_threshold: float = 3.0
    debug_spike_max_saves: int = 10


def train(
    c: MainConfig,
    device: str | None = None,
    reusable_config: bool = True,
    compile: bool = False,
    overwrite: bool = False,
    logger: RunManager | None = None,
    log_every_n_steps: int | None = None,
    finish_logger: bool = True,
    # Handy functions to override when not working with a standard file system
    save_object_function: tp.Callable | None = None,  # defaults to torch.save
    load_object_function: tp.Callable | None = None,  # defaults to torch.load
    check_path_exists_function: tp.Callable | None = None,  # defaults to os.path.exists
):
    if reusable_config:
        assert c.from_yaml(c.to_yaml()) == c, (
            "Config is not safe to use, got different config: "
            f"{c.from_yaml(c.to_yaml())=} vs {c=}"
        )

    # Arguments from original signature not in MainConfig are set to defaults here
    load_weights_from_this_state_dict = None

    total_start_time = time.time()

    if device is None:
        device = utils.get_default_device()
    using_dist, rank, device = init_dist(device)
    print(f"ALL: Using device {device}.")

    if logger is None:
        logger = NullRunManager()
    if rank != 0:
        logger = NullRunManager()
        finish_logger = False
    if log_every_n_steps is None:
        log_every_n_steps = c.wandb.log_every_n_steps if c.wandb is not None else 10
    if getattr(logger, "run_id", None) is not None and c.wandb_run_id != logger.run_id:
        c = c.__class__(**{**c.__dict__, "wandb_run_id": logger.run_id})

    # Resolve dataloader_class string to actual class
    if c.dataloader_class is None:
        actual_dataloader_class = data_loading.StandardDataLoader
    else:
        parts = c.dataloader_class.split(".")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        try:
            module = importlib.import_module(module_path)
            actual_dataloader_class = getattr(module, class_name)
        except Exception as e:
            raise ImportError(
                f"Could not import dataloader_class '{c.dataloader_class}': {e}"
            ) from e

    def create_get_batch_method(priors: tp.List[prior.PriorConfig] | None):
        if not priors:
            raise ValueError("main_config.priors cannot be empty.")

        if len(priors) != 1:
            raise ValueError(
                "Currently only supporting a single prior. Later this should be a seqeunce that is called in order by wrapping."
            )

        if len(priors) == 1 and callable(priors[0]):  # Simplistic assumption
            get_batch_method_instance = priors[0]
        elif isinstance(priors[0], prior.PriorConfig):
            get_batch_method_instance = priors[0].create_get_batch_method()
        else:
            raise ValueError(
                "main_config.priors and main_config.test_priors must be a list of PriorConfig objects or a single callable."
            )

        return get_batch_method_instance

    get_batch_method_instance = create_get_batch_method(c.priors)
    if c.test_priors is not None:
        test_get_batch_method_instance = create_get_batch_method(c.test_priors)
    else:
        test_get_batch_method_instance = get_batch_method_instance

    current_extra_prior_kwargs_dict = {}
    if c.num_workers is not None:
        current_extra_prior_kwargs_dict["num_workers"] = c.num_workers

    data_loader = actual_dataloader_class(
        get_batch_method=get_batch_method_instance,  # Use the constructed/resolved instance
        batch_shape_sampler_function=c.batch_shape_sampler.sample_batch_shape,
        num_steps=c.steps_per_epoch,
        device=device,  # Pass the torch device object
        n_targets_per_input=c.n_targets_per_input,
        persistent_workers=True,  # can have persistent workers, as the dataset is counting the epochs itself here
        **current_extra_prior_kwargs_dict,
    )

    test_data_loader = actual_dataloader_class(
        get_batch_method=test_get_batch_method_instance,  # Use the constructed/resolved instance
        batch_shape_sampler_function=c.batch_shape_sampler.sample_batch_shape,
        num_steps=c.steps_per_epoch,
        device=device,  # Pass the torch device object
        n_targets_per_input=c.n_targets_per_input,
        persistent_workers=False,  # can't have persistent workers, otherwise the epoch count is not updated
        **current_extra_prior_kwargs_dict,
    )

    assert (
        c.model.features_per_group > 0 or c.model.features_per_group == -1
    ), "features_per_group must be > 0 or -1"

    model = c.model.create_model()
    criterion = model.criterion

    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)

    print(
        f"Using a model with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters"
    )

    model.to(device)

    if compile:
        model = torch.compile(model)

    if hasattr(c.optimizer, "create_optimizer"):
        optimizer = c.optimizer.create_optimizer(model.parameters())
    else:
        raise ValueError("main_config.optimizer must have a 'create_optimizer' method")

    # Resolve scheduler string to function
    if c.scheduler == "cosine_decay":
        scheduler_fn = get_cosine_schedule_with_warmup
    else:
        assert c.scheduler == "constant", f"Scheduler {c.scheduler} not supported"
        scheduler_fn = None

    if scheduler_fn is None:
        scheduler = None
    else:
        scheduler = scheduler_fn(  # todo move warmup epochs into scheduler args, ideally as steps instead!?
            optimizer,
            c.warmup_epochs,
            c.epochs if c.epochs is not None else 100,
        )

    start_epoch = 1  # Default start epoch

    if not overwrite and should_load_checkpoint(c, check_path_exists_function=check_path_exists_function):
        # load_checkpoint needs the scheduler instance, not the factory
        start_epoch = load_checkpoint(  # load_checkpoint might return start_epoch
            model,
            optimizer,
            scheduler,
            c.train_state_dict_load_path,
            device,
            load_function=load_object_function,
        )
    else:
        print(
            f"Checkpoint file {c.train_state_dict_load_path} not found or load/save paths are identical and file doesn't exist. Starting from scratch."
        )

    # set this before DDP
    data_loader.model = model
    test_data_loader.model = model

    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
        )  # each GPU gets a copy of the model, gradients are synced during backward()

    scaler = GradScaler() if c.train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(data_loader)
    utils.check_compatibility(test_data_loader)

    total_loss = float("inf")
    try:
        for epoch in range(start_epoch, c.epochs + 1):
            epoch_start_time = time.time()
            try:
                epoch_result = train_or_evaluate_epoch(
                    c=c,
                    model=model,
                    optimizer=optimizer,
                    dl=data_loader,
                    device=device,
                    scaler=scaler,
                    criterion=criterion,
                    rank=rank,
                    using_dist=using_dist,
                    training=True,
                    logger=logger,
                    epoch=epoch,
                    log_every_n_steps=log_every_n_steps,
                    last_epoch_result=epoch_result if epoch > start_epoch else None,
                )
                total_loss = epoch_result.loss
                data_loader.importance_sampling_infos = (
                    epoch_result.importance_sampling_infos
                )

            except Exception as e:
                print("Invalid epoch encountered, skipping...")
                print(e)
                raise  # Re-raises the original exception with trace
            if c.validation_period is not None and (
                (epoch % c.validation_period == 0)
                or (epoch == c.epochs)
                or (epoch == 1)
            ):
                with torch.no_grad():
                    test_data_loader.epoch_count = (
                        epoch - 1
                    )  # -1 because the data_loader.__iter__ increases before the epoch
                    val_epoch_result = train_or_evaluate_epoch(
                        c=c,
                        model=model,
                        optimizer=optimizer,
                        dl=test_data_loader,
                        device=device,
                        scaler=None,
                        criterion=criterion,
                        rank=rank,
                        using_dist=using_dist,
                        training=False,
                        logger=None,  # Don't log step-level info for validation
                        epoch=epoch,
                        log_every_n_steps=log_every_n_steps,
                    )
                    val_score_str = f"| eval mean loss {val_epoch_result.loss:5.2f} "

            else:
                val_score_str = ""

            epoch_time = time.time() - epoch_start_time
            if device.startswith("cuda"):
                max_gpu_mem_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
                gpu_utilization = torch.cuda.utilization()
            else:
                max_gpu_mem_gb = None
                gpu_utilization = None
            current_lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )

            if c.verbose:
                print("-" * 89)
                print(
                    f"| end of epoch {epoch:3d} | time: {epoch_time:5.2f}s | mean loss {epoch_result.loss:5.2f} "
                    + f"{val_score_str}"
                    + f"| lr {current_lr} "
                    + f"| data time {epoch_result.data_time:5.2f} step time {epoch_result.step_time:5.2f} "
                    + f"forward time {epoch_result.forward_time:5.2f} "
                    + f"| max gpu mem {f'{max_gpu_mem_gb:.1f}' if max_gpu_mem_gb is not None else 'N/A'} GiB "
                    + f"| gpu utilization {f'{gpu_utilization:.1f}' if gpu_utilization is not None else 'N/A'} %"
                    + f"| nan share {epoch_result.nan_share:5.2f} ignore share (for classification tasks) {epoch_result.ignore_share:5.4f} "
                    + f"| grad norm ema mean {epoch_result.grad_norm_ema_mean:5.2f}"
                    + f"| grad norm infinite steps fraction {epoch_result.grad_norm_infinite_steps_fraction:5.2f}"
                    + f"| grad norm ema exceeded fraction {epoch_result.grad_norm_ema_exceeded_fraction:5.2f}"
                )
                print("-" * 89)

            global_step_end = epoch * len(data_loader)
            logger_payload: dict[str, tp.Any] = {
                "trainer/epoch": epoch,
                "trainer/global_step": global_step_end,
                "epoch/train_loss": epoch_result.loss,
                "epoch/epoch_time": epoch_time,
                "epoch/data_time": epoch_result.data_time,
                "epoch/step_time": epoch_result.step_time,
                "epoch/forward_time": epoch_result.forward_time,
                "epoch/nan_share": epoch_result.nan_share,
                "epoch/ignore_share": epoch_result.ignore_share,
                "epoch/learning_rate": current_lr,
                "epoch/grad_norm_ema_mean": epoch_result.grad_norm_ema_mean,
                "epoch/grad_norm_infinite_steps_fraction": epoch_result.grad_norm_infinite_steps_fraction,
                "epoch/grad_norm_ema_exceeded_fraction": epoch_result.grad_norm_ema_exceeded_fraction,
            }
            if device.startswith("cuda"):
                logger_payload["epoch/max_gpu_memory_gb"] = max_gpu_mem_gb
                logger_payload["epoch/gpu_utilization"] = gpu_utilization

            if c.validation_period is not None and (
                (epoch % c.validation_period == 0)
                or (epoch == c.epochs)
                or (epoch == 1)
            ):
                logger_payload["epoch/val_loss"] = val_epoch_result.loss

            logger.log(logger_payload, step=global_step_end)

            if scheduler is not None:
                scheduler.step()

            if epoch_result.loss > .8 and epoch > 15:
                raise ValueError(f"Aborting training due to high loss: {epoch_result.loss}")
            # Save model state dict after each epoch if path is provided (on rank 0)
            if c.train_state_dict_save_path is not None and rank == 0 and epoch_result.loss < .8:
                save_checkpoint(
                    model,
                    optimizer,
                    c.train_state_dict_save_path,
                    epoch,
                    config=c,
                    save_function=save_object_function,
                )

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        pass
    finally:
        if rank == 0:  # trivially true for non-parallel training
            set_model_to(model, optimizer, "eval")
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
                data_loader = None
            if finish_logger:
                logger.finish()
    return {
        "total_loss": total_loss,
        "model": model.to("cpu"),
        "total_time": time.time() - total_start_time,
    }


def _resolve_autocast_dtype(device: str, dtype_spec: str | None) -> torch.dtype:
    dtype_spec = (dtype_spec or "fp16").lower()
    if dtype_spec == "auto":
        return torch.float16 if device.startswith("cuda") else torch.bfloat16
    if dtype_spec in ("fp16", "float16"):
        return torch.float16
    if dtype_spec in ("bf16", "bfloat16"):
        if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise ValueError(
                "Requested bf16 autocast but CUDA device does not support bf16."
            )
        return torch.bfloat16
    raise ValueError(
        f"Unsupported train_mixed_precision_dtype '{dtype_spec}'. "
        "Use 'auto', 'bf16', or 'fp16'."
    )


# we could think about removing c as arg here to make the dep's clearer
def train_or_evaluate_epoch(
    c: MainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dl: priors_utils.DataLoader,
    device: str,
    scaler: GradScaler | None,
    criterion: torch.nn.Module,
    rank: int,
    using_dist: bool,
    training: bool = True,
    logger: RunManager | None = None,
    epoch: int = 1,
    log_every_n_steps: int = 10,
    last_epoch_result: Metrics | None = None,
):
    """
    Train or evaluate one epoch.
    """
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
    if training:
        assert optimizer is not None, "Optimizer must be provided for training"
    else:
        assert scaler is None, "Scaler must be None for evaluation"

    set_model_to(model, optimizer, "train" if training else "eval")

    metrics = Metrics(steps_per_epoch=len(dl))

    importance_sampling_infos = []
    grad_norm_ema = 0.0 if last_epoch_result is None else last_epoch_result.grad_norm_ema_mean
    spike_save_count = 0
    autocast_dtype = (
        _resolve_autocast_dtype(device, c.train_mixed_precision_dtype)
        if scaler is not None
        else torch.float32
    )
    
    before_get_batch = time.time()
    assert (
        len(dl) % c.aggregate_k_gradients == 0
    ), "Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it."

    tqdm_iter = (
        tqdm(range(len(dl)), desc="Training Epoch")
        if rank == 0 and c.progress_bar
        else None
    )

    for batch_index, batch in enumerate(dl):
        batch: prior.Batch = batch  # for IDE support
        # batch.x.shape == (batch_size, seq_len, num_features)
        grad_norm_infinite_steps_batch = 0
        grad_norm_ema_exceeded = 0
        if not c.model.attention_between_features:
            num_features = batch.x.shape[2]
            assert (
                num_features <= c.model.features_per_group
            ), (
                "When attention_between_features is False, the model requires a single "
                "feature group. Set features_per_group to be >= the batch's number of "
                f"features (typically max_num_features). Got {num_features=} and "
                f"{c.model.features_per_group=}."
            )
        targets = batch.target_y.to(device)
        single_eval_pos = batch.single_eval_pos

        if tqdm_iter is not None:
            tqdm_iter.update()

        # only synch gradients once every aggregate_k_gradients steps
        if using_dist and not (
            batch_index % c.aggregate_k_gradients == c.aggregate_k_gradients - 1
        ):
            potentially_no_sync_context = model.no_sync()
        else:
            potentially_no_sync_context = nullcontext()

        if training:
            potentially_no_grad_context = nullcontext()
        else:
            potentially_no_grad_context = torch.no_grad()

        with potentially_no_sync_context, potentially_no_grad_context:
            time_to_get_batch = time.time() - before_get_batch
            before_forward = time.time()
            try:
                with autocast(
                    device.split(":")[0],
                    enabled=scaler is not None,
                    dtype=autocast_dtype,
                ):
                    categorical_inds = None
                    if hasattr(batch, 'categorical_mask') and batch.categorical_mask is not None:
                        mask = batch.categorical_mask
                        if mask.ndim > 1:
                            if not torch.all(mask == mask[0]):
                                raise NotImplementedError(
                                    "Per-sample categorical masks are not yet supported. "
                                    "All samples in a batch must have the same categorical features. "
                                    "This should not happen with TabPFN prior (flexible=True)."
                                )
                            mask = mask[0]
                        categorical_inds = torch.nonzero(mask, as_tuple=True)[0].tolist()
                    
                    output = model(
                        x=batch.x.to(device),
                        y=batch.y[:, :single_eval_pos].to(device),
                        style=move_style_and_check_shape(batch.style, batch.x, device),
                        y_style=move_y_style_and_check_shape(
                            batch.y_style, batch.y, device
                        ),
                        categorical_inds=categorical_inds,
                        only_return_standard_out=True,
                    )  # shape: (batch_size, test_len)

                    forward_time = time.time() - before_forward

                    if single_eval_pos is not None:
                        targets = targets[
                            :, single_eval_pos:
                        ]  # shape: (batch_size, test_len)

                    losses = compute_losses(
                        output, targets, criterion, c.n_targets_per_input
                    )  # shape: (batch_size, test_len)

                    loss, nan_share = utils.torch_nanmean(
                        losses.mean(
                            1
                        ),  # loss per sequence without nanmean, if any loss in a sequence is nan, the whole sequence is ignored
                        return_nanshare=True,
                    )  # loss and nan_share are both scalar tensors
                    loss_is_finite = torch.isfinite(loss).item()
                    if (not loss_is_finite) or (loss.item() > c.debug_spike_threshold):
                        print("Loss spike detected")
                        print(f"loss={loss.item():.6g} nan_share={nan_share:.4g}")
                        print(f"single_eval_pos={single_eval_pos} n_targets_per_input={c.n_targets_per_input}")
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
                            and c.train_state_dict_save_path is not None
                            and spike_save_count < c.debug_spike_max_saves
                        ):
                            checkpoint_dir = os.path.dirname(c.train_state_dict_save_path)
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
                                "n_targets_per_input": c.n_targets_per_input,
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
                            print(f"Saved spike batch to {save_path}")
                    loss_scaled = loss / c.aggregate_k_gradients

                if scaler:
                    loss_scaled = scaler.scale(loss_scaled)

                if training:
                    loss_scaled.backward()

                if batch_index % c.aggregate_k_gradients == c.aggregate_k_gradients - 1:
                    if scaler:
                        # we unscale s.t. we can clip grads right
                        scaler.unscale_(optimizer)
                    
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                    grad_norm = grad_norm_tensor.item()
                    update_ratio = compute_update_ratio(model, optimizer, grad_norm)
                    
                    is_spike = grad_norm > c.skip_grad_norm_spike_factor * grad_norm_ema
                    in_warmup = epoch <= c.warmup_epochs
                    
                    if grad_norm_ema == 0.0:
                        grad_norm_ema = grad_norm  # initialize ema
                    
                    if math.isfinite(grad_norm):
                        GRAD_NORM_DECAY = 0.95
                        
                        updated_grad_norm = grad_norm if in_warmup else min(grad_norm, c.skip_grad_norm_spike_factor * grad_norm_ema)
                        grad_norm_ema = (
                            GRAD_NORM_DECAY * grad_norm_ema
                            + (1.0 - GRAD_NORM_DECAY) * updated_grad_norm
                        )
                        
                        update_importance_sampling_infos(
                            importance_sampling_infos=importance_sampling_infos,
                            model=model,
                            optimizer=optimizer,
                            loss=loss.cpu().item(),
                            info_used_with_gradient_magnitudes=batch.info_used_with_gradient_magnitudes,
                        )

                        grad_norm_clip_value = 1.0
                        if is_spike and not in_warmup:
                            grad_norm_ema_exceeded += 1
                            grad_norm_clip_value = min(grad_norm_clip_value, c.skip_grad_norm_spike_factor * grad_norm_ema)
                            print(
                                f"Grad norm spike detected: grad_norm={grad_norm:.6g} "
                                f"ema={grad_norm_ema:.6g} lr={optimizer.param_groups[0]['lr']} "
                                f"update_ratio={update_ratio:.6g}"
                            )
                            if scaler is not None:
                                print(f"amp_scale={scaler.get_scale()}")
                        torch.nn.utils.clip_grads_with_norm_(model.parameters(), grad_norm_clip_value, total_norm=grad_norm_tensor)

                        if batch.gradient_multipliers is not None:  # this None by default
                            assert (
                                training
                            ), "Gradient multipliers are only supported for training"
                            assert (
                                c.aggregate_k_gradients == 1
                            ), "Scaling grads is only supported if you don't do grad acc."
                            assert all(
                                batch.gradient_multipliers.view(-1)[0]
                                == batch.gradient_multipliers.view(-1)[i]
                                for i in range(batch.gradient_multipliers.numel())
                            ), "we don't scale losses for now to be able to try the interaction with gradient clipping, and thus we can only support the same scaler"
                            # todo make print to see that this is actually running
                            with torch.no_grad():
                                for w in model.parameters():
                                    w.grad = w.grad * batch.gradient_multipliers.view(-1)[0]

                        if training:
                            if scaler:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            optimizer.zero_grad()
                    else:
                        grad_norm_infinite_steps_batch += 1
                        if training:
                            if scaler:
                                scaler.update()
                            optimizer.zero_grad()

                step_time = time.time() - before_forward

                metrics.update(
                    loss=loss,
                    nan_share=nan_share,
                    targets=targets,
                    forward_time=forward_time,
                    step_time=step_time,
                    time_to_get_batch=time_to_get_batch,
                    grad_norm_ema=grad_norm_ema,
                    grad_norm_infinite_steps=grad_norm_infinite_steps_batch,
                    grad_norm_ema_exceeded=grad_norm_ema_exceeded,
                )

            except Exception as e:
                print("Invalid step encountered, skipping...")
                print(e)
                raise (e)

        mean_loss = metrics.total_loss / (batch_index + 1)
        mean_infinite_steps = (
            metrics.grad_norm_infinite_steps / (batch_index + 1)
        )
        if tqdm_iter:
            tqdm_iter.set_postfix(
                {
                    "data_time": time_to_get_batch,
                    "step_time": step_time,
                    "mean_loss": mean_loss,
                }
            )

        if logger and training and (batch_index % log_every_n_steps == 0):
            global_step = (epoch - 1) * len(dl) + batch_index
            logger.log(
                {
                    "trainer/epoch": epoch,
                    "trainer/global_step": global_step,
                    "step/data_time": time_to_get_batch,
                    "step/step_time": step_time,
                    "step/mean_loss": mean_loss,
                    "step/grad_norm_ema": grad_norm_ema,
                    "step/grad_norm_infinite_steps": mean_infinite_steps,
                },
                step=global_step,
            )

        before_get_batch = time.time()

    return metrics.get_epoch_result(importance_sampling_infos)


def compute_losses(
    output: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
    n_targets_per_input: int,
):
    """
    Compute the losses for the given output and targets.

    Args:
        output: The output of the model, shape (batch_size, num_eval_positions, n_out)
        targets: The targets, shape (batch_size, num_eval_positions[, n_targets_per_input])
        criterion: The criterion to use.
        n_targets_per_input: The number of targets per input.

    Returns:
        The losses, shape (batch_size, num_eval_positions)
    """
    # Repeat output in the semi-last dimension n_targets_per_input times
    output = output.unsqueeze(2).expand(
        *output.shape[:2],
        n_targets_per_input,
        output.shape[-1],
    )

    if len(targets.shape) == 2:
        # This implies we only have a single target per input
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


def should_load_checkpoint(
    config: MainConfig, check_path_exists_function: tp.Callable | None = None
):
    if config.train_state_dict_load_path is None:
        return False
    if check_path_exists_function is None:
        check_path_exists_function = os.path.exists
    return (config.train_state_dict_save_path != config.train_state_dict_load_path) or (
        (config.train_state_dict_save_path == config.train_state_dict_load_path)
        and check_path_exists_function(config.train_state_dict_load_path)
    )


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    train_state_dict_load_path,
    device,
    load_function: tp.Callable | None = None,
):
    print(f"Loading checkpoint from {train_state_dict_load_path}")
    if load_function is None:
        load_function = torch.load
    try:
        checkpoint = load_function(train_state_dict_load_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # New format with model, optimizer state and epoch
            state_dict = checkpoint["model_state_dict"]
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                stripped_state_dict, prefix = utils.strip_compiled_state_dict_prefix(
                    state_dict
                )
                if prefix is None:
                    raise
                print(
                    "Detected compiled model weights. "
                    f"Stripping '{prefix}' from state_dict keys."
                )
                model.load_state_dict(stripped_state_dict, strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")
            # Fast-forward the scheduler to the correct epoch
            if scheduler is not None:
                for _ in range(start_epoch - 1):
                    scheduler.step()
            return start_epoch
        else:
            raise ValueError(
                f"Checkpoint does not contain 'model_state_dict' or 'optimizer_state_dict'. Checkpoint: {checkpoint}"
            )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise e


def load_config(train_state_dict_load_path, load_function: tp.Callable | None = None):
    if load_function is None:
        load_function = torch.load
    checkpoint = load_function(train_state_dict_load_path, map_location="cpu")
    return MainConfig.from_dict(checkpoint["config"])


def save_checkpoint(
    model,
    optimizer,
    train_state_dict_save_path,
    epoch,
    config: MainConfig,
    save_function: tp.Callable | None = None,
):
    set_model_to(model, optimizer, "eval")
    save_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    print(f"Saving checkpoint to {train_state_dict_save_path} (epoch {epoch})")
    try:
        # Save model state dict, optimizer state dict, and current epoch
        checkpoint = {
            "model_state_dict": save_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config.to_dict(),
        }
        if save_function is None:
            os.makedirs(os.path.dirname(train_state_dict_save_path), exist_ok=True)
            save_function = torch.save
        save_function(checkpoint, train_state_dict_save_path)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
