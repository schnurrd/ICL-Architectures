from __future__ import annotations

from contextlib import nullcontext
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from tqdm.auto import tqdm

from pfns.scripts.tabular_metrics import auc_metric
from pfns.training_utils import (
    categorical_mask_to_inds,
    compute_losses,
    is_autocast_dtype_enabled,
    move_style_and_check_shape,
    move_y_style_and_check_shape,
    resolve_autocast_dtype
)
from pfns.utils import get_default_device, torch_nanmean

from .constants import MEMORY_NAMES, METRIC_NAMES, SCHEMA_VERSION, TIMING_NAMES
from .benchmark_batch_generators import _set_data_generation_seed, create_seq_len_batch_generator

EVAL_MODES = ["fit_predict", "forward"]


@dataclass
class BenchmarkTables:
    metric_table: dict[str, dict[str, dict[int, list[float]]]]
    timing_table: dict[str, dict[str, dict[int, list[float]]]]
    memory_table: dict[str, dict[str, dict[int, list[float]]]]
    oom_errors: dict[str, set[int]]
    
    @staticmethod
    def _empty_table(
        model_names: list[str],
        metric_names: tuple[str, ...],
    ) -> dict[str, dict[str, dict[int, list[float]]]]:
        return {name: {metric: {} for metric in metric_names} for name in model_names}

    @classmethod
    def create(cls, model_names: list[str]) -> "BenchmarkTables":
        return cls(
            metric_table=cls._empty_table(model_names, METRIC_NAMES),
            timing_table=cls._empty_table(model_names, TIMING_NAMES),
            memory_table=cls._empty_table(model_names, MEMORY_NAMES),
            oom_errors={name: set() for name in model_names},
        )

    @staticmethod
    def _append(
        table: dict[str, dict[str, dict[int, list[float]]]],
        model: str,
        metric: str,
        seqlen: int,
        value: float,
    ) -> None:
        table[model][metric].setdefault(int(seqlen), []).append(float(value))

    def append_metric(self, model: str, metric: str, seqlen: int, value: float) -> None:
        self._append(self.metric_table, model, metric, seqlen, value)

    def append_timing(self, model: str, metric: str, seqlen: int, value: float) -> None:
        self._append(self.timing_table, model, metric, seqlen, value)

    def append_memory(self, model: str, metric: str, seqlen: int, value: float) -> None:
        self._append(self.memory_table, model, metric, seqlen, value)

    def should_skip_seqlen(self, model: str, seqlen: int) -> bool:
        return any(oom_len <= seqlen for oom_len in self.oom_errors[model])

    def mark_oom(self, model: str, seqlen: int) -> None:
        self.oom_errors[model].add(seqlen)


def _timed(fn: Any, *, is_cuda: bool) -> tuple[Any, float]:
    if is_cuda:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()
        out = fn()
        end_evt.record()
        torch.cuda.synchronize()
        return out, float(start_evt.elapsed_time(end_evt))

    start_time = time.perf_counter()
    out = fn()
    return out, float((time.perf_counter() - start_time) * 1000)


class BenchmarkOOMError(RuntimeError):
    """Raised when a model evaluation step runs out of CUDA memory."""


def _evaluate_model_at_seqlen(
    *,
    model: Any,
    config: Any,
    base_batch: Any,
    seqlen: int,
    largest_seqlen: int,
    number_of_test_samples: int,
    resolved_device: str,
    is_cuda: bool,
    warmup_iters: int,
    num_classes: int,
    categorical_inds: list[int] | None,
    eval_mode: Literal["fit_predict", "forward"],
    subsample_dataset_size: int | None = None,
) -> dict[str, float | None]:
    train_x = base_batch.x[:, :seqlen]
    train_y = base_batch.y[:, :seqlen]
    test_x = base_batch.x[:, largest_seqlen : largest_seqlen + number_of_test_samples]
    test_target_y = base_batch.target_y[
        :,
        largest_seqlen : largest_seqlen + number_of_test_samples,
    ]
    x_test = test_x.to(resolved_device)
    selected_rows_list = [None]
    if subsample_dataset_size is not None and seqlen > subsample_dataset_size:
        shuffled_rows = np.random.default_rng(seqlen).permutation(seqlen)
        num_slices = (seqlen + subsample_dataset_size - 1) // subsample_dataset_size
        selected_rows_list = [
            sorted(rows.tolist())
            for rows in np.array_split(shuffled_rows, num_slices)
        ]

    def run_eval(selected_rows: list[int] | None, warmup_only: bool = False):
        member_train_x = train_x
        member_train_y = train_y
        if selected_rows is not None:
            member_train_x = member_train_x[:, selected_rows]
            member_train_y = member_train_y[:, selected_rows]

        x_train = member_train_x.to(resolved_device)
        y_train = member_train_y.to(resolved_device)
        style = move_style_and_check_shape(base_batch.style, member_train_x, resolved_device)
        y_style = move_y_style_and_check_shape(base_batch.y_style, member_train_y, resolved_device)
        fit_kwargs = {
            "x": x_train,
            "y": y_train,
            "style": style,
            "y_style": y_style,
            "categorical_inds": categorical_inds,
        }
        pred_kwargs = {
            "test_x": x_test,
            "style": style,
            "y_style": y_style,
            "categorical_inds": categorical_inds,
            "only_return_standard_out": True,
        }
        forward_kwargs = {
            "x": x_train,
            "y": y_train,
            "test_x": x_test,
            "style": style,
            "y_style": y_style,
            "categorical_inds": categorical_inds,
            "only_return_standard_out": True,
        }
        if eval_mode == "fit_predict":
            if warmup_only:
                for _ in range(warmup_iters):
                    warm_state = model.incontext_fit(**fit_kwargs)
                    _ = model.incontext_predict(warm_state, **pred_kwargs)
                return None
            state, fit_ms = _timed(lambda: model.incontext_fit(**fit_kwargs), is_cuda=is_cuda)
            output, pred_ms = _timed(
                lambda: model.incontext_predict(state, **pred_kwargs),
                is_cuda=is_cuda,
            )
            return output, fit_ms, pred_ms, state.size_bytes() / (1024**2)
        if eval_mode == "forward":
            if warmup_only:
                for _ in range(warmup_iters):
                    _ = model(**forward_kwargs)
                return None
            output, pred_ms = _timed(lambda: model(**forward_kwargs), is_cuda=is_cuda)
            return output, 0.0, pred_ms, 0.0
        raise ValueError(f"Unknown eval_mode {eval_mode!r}. Expected one of: {', '.join(EVAL_MODES)}.")

    try:
        for selected_rows in selected_rows_list:
            run_eval(selected_rows, warmup_only=True)

        peak_allocated_mb = None
        peak_reserved_mb = None
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated() / (1024**2)
            base_reserved = torch.cuda.memory_reserved() / (1024**2)

        outputs = []
        fit_ms = 0.0
        pred_ms = 0.0
        context_size_mb_values = []
        for selected_rows in selected_rows_list:
            output_i, fit_ms_i, pred_ms_i, context_size_mb_i = run_eval(selected_rows)
            outputs.append(output_i)
            fit_ms += fit_ms_i
            pred_ms += pred_ms_i
            context_size_mb_values.append(float(context_size_mb_i))
        context_size_mb = max(context_size_mb_values) if context_size_mb_values else 0.0
        output = outputs[0] if len(outputs) == 1 else torch.stack(outputs).mean(dim=0)

        forward_ms = fit_ms + pred_ms

        if is_cuda:
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
            peak_allocated_mb = max(0.0, peak_alloc - base_alloc)
            peak_reserved_mb = max(0.0, peak_reserved - base_reserved)

        targets = test_target_y.to(resolved_device)
        losses = compute_losses(
            output,
            targets,
            model.criterion,
            config.n_targets_per_input,
        )
        loss, _ = torch_nanmean(losses.mean(1), return_nanshare=True)

        valid = targets != -100
        if valid.any():
            pred = output.argmax(dim=-1)
            acc = (pred[valid] == targets[valid]).float().mean().item()
            probs = torch.softmax(output[..., :num_classes], dim=-1)[valid]
            try:
                auc = auc_metric(
                    targets[valid].cpu(),
                    probs.detach().cpu(),
                    multi_class="ovr",
                )
                auc = float(auc.item() if torch.is_tensor(auc) else auc)
            except Exception as err:
                print(f"Error computing AUC: {err}")
                auc = float("nan")
        else:
            acc = float("nan")
            auc = float("nan")

        return {
            "fit_time_ms": float(fit_ms),
            "predict_time_ms": float(pred_ms),
            "forward_time_ms": float(forward_ms),
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "context_size_mb": float(context_size_mb),
            "acc": float(acc),
            "ce": float(loss.item()),
            "roc_auc": float(auc),
        }
    except torch.cuda.OutOfMemoryError as err:
        raise BenchmarkOOMError from err
    except RuntimeError as err:
        if is_cuda and "out of memory" in str(err).lower():
            raise BenchmarkOOMError from err
        raise


def evaluate_models_over_seqlens(
    *,
    models: dict[str, Any],
    configs: dict[str, Any],
    seqlen_list: list[int],
    num_features: int,
    num_classes: int,
    number_of_test_samples: int = 100,
    number_of_repetitions: int = 100,
    use_warmup_iters: bool = False,
    print_timing: bool = False,
    autocast_models: dict[str, torch.dtype] | None | Literal["auto"] = "auto", # Dict of (model_name, dtype) pairs to apply autocast to, or "auto" to infer from configs
    forward_models: list[str] | None = None,
    device: str | None = None,
    progress_desc: str = "Overall progress",
    data_generation_seed: int | None = None,
    subsample_dataset_size: int | None = None,
    task_variant: str = "tabular_prior",
    task_kwargs: dict[str, Any] | None = None,
    precomputed_batches: list[tuple[Any, float]] | None = None,
) -> dict[str, Any]:
    """Run sequence-length evaluation for all models and return nested result tables."""
    if not models:
        raise ValueError("No models provided.")
    if not seqlen_list:
        raise ValueError("seqlen_list cannot be empty.")
    forward_models_set = set(forward_models or [])
    
    resolved_device = device or get_default_device()
    device_type = torch.device(resolved_device).type
    is_cuda = device_type == "cuda"
    if autocast_models == "auto":
        print("Inferring autocast settings from configs...")
        resolved_eval_dtype = resolve_autocast_dtype(resolved_device, "auto")
        autocast_models = { 
            name: resolved_eval_dtype
            for name in configs.keys() 
            if is_autocast_dtype_enabled(resolved_eval_dtype)
        }
    elif autocast_models is None:
        autocast_models = {}
    warmup_iters = 3 if use_warmup_iters else 0
    resolved_task_kwargs = dict(task_kwargs or {})

    if data_generation_seed is not None and precomputed_batches is None:
        _set_data_generation_seed(int(data_generation_seed))
        if task_variant == "associative_recall":
            resolved_task_kwargs.setdefault(
                "data_generation_seed",
                int(data_generation_seed),
            )

    tables = BenchmarkTables.create(list(models))
    smallest_seqlen, largest_seqlen = min(seqlen_list), max(seqlen_list)

    def tprint(*args, **kwargs):
        if print_timing:
            print(*args, **kwargs)

    if precomputed_batches is not None:
        batch_generator = precomputed_batches
    else:
        batch_generator = create_seq_len_batch_generator(
            task_variant=task_variant,
            num_batches=number_of_repetitions,
            smallest_seqlen=smallest_seqlen,
            largest_seqlen=largest_seqlen,
            num_features=num_features,
            num_classes=num_classes,
            number_of_test_samples=number_of_test_samples,
            default_device=resolved_device,
            task_kwargs=resolved_task_kwargs,
        )

    for rep, (base_batch, data_gen_ms) in enumerate(
        tqdm(batch_generator, total=number_of_repetitions, desc=progress_desc),
        start=1,
    ):
        tprint(f"Data generation rep {rep}: {data_gen_ms:.2f} ms")
        categorical_inds = categorical_mask_to_inds(base_batch.categorical_mask)

        for model_name, model in models.items():
            config = configs[model_name]
            model_eval_mode = "forward" if model_name in forward_models_set else "fit_predict"
            optimization_config = getattr(model, "optimization_config", None)
            evaluate_only_max_seqlen = bool(
                getattr(optimization_config, "evaluate_only_max_seqlen", False)
            )
            model_seqlens = [largest_seqlen] if evaluate_only_max_seqlen else seqlen_list
            grad_context = (
                nullcontext()
                if getattr(model, "requires_grad_during_eval", False)
                else torch.inference_mode()
            )
            with grad_context:
                with torch.autocast(
                    device_type=device_type,
                    enabled=is_cuda and model_name in autocast_models,
                    dtype=autocast_models.get(model_name, torch.float32),
                ):
                    for seqlen in model_seqlens:
                        if tables.should_skip_seqlen(model_name, seqlen):
                            continue

                        try:
                            result = _evaluate_model_at_seqlen(
                                model=model,
                                config=config,
                                base_batch=base_batch,
                                seqlen=seqlen,
                                largest_seqlen=largest_seqlen,
                                number_of_test_samples=number_of_test_samples,
                                resolved_device=resolved_device,
                                is_cuda=is_cuda,
                                warmup_iters=warmup_iters,
                                num_classes=num_classes,
                                categorical_inds=categorical_inds,
                                eval_mode=model_eval_mode,
                                subsample_dataset_size=subsample_dataset_size,
                            )
                        except BenchmarkOOMError:
                            tables.mark_oom(model_name, seqlen)
                            print(f"\nOOM: {model_name} at seqlen={seqlen}")
                            if is_cuda:
                                torch.cuda.empty_cache()
                            break

                        for timing_name in TIMING_NAMES:
                            tables.append_timing(model_name, timing_name, seqlen, result[timing_name])

                        for memory_name in MEMORY_NAMES:
                            memory_value = result[memory_name]
                            if memory_value is not None:
                                tables.append_memory(model_name, memory_name, seqlen, memory_value)

                        metric_seqlens = [seqlen]
                        if (
                            evaluate_only_max_seqlen
                            and seqlen == largest_seqlen
                        ):
                            metric_seqlens = seqlen_list

                        for metric_name in METRIC_NAMES:
                            for metric_seqlen in metric_seqlens:
                                tables.append_metric(
                                    model_name,
                                    metric_name,
                                    metric_seqlen,
                                    result[metric_name],
                                )

    return {
        "schema_version": SCHEMA_VERSION,
        "metric_table": tables.metric_table,
        "timing_table": tables.timing_table,
        "memory_table": tables.memory_table,
        "oom_errors": {k: sorted(v) for k, v in tables.oom_errors.items()},
        "metadata": {
            "seqlen_list": list(seqlen_list),
            "num_features": num_features,
            "num_classes": num_classes,
            "number_of_test_samples": number_of_test_samples,
            "number_of_repetitions": number_of_repetitions,
            "device": resolved_device,
            "forward_models": sorted(forward_models_set),
            "data_generation_seed": (
                int(data_generation_seed) if data_generation_seed is not None else None
            ),
            "subsample_dataset_size": (
                int(subsample_dataset_size)
                if subsample_dataset_size is not None
                else None
            ),
            "task_variant": task_variant,
            "task_kwargs": resolved_task_kwargs,
            "precomputed_batches": bool(precomputed_batches is not None),
        },
    }
