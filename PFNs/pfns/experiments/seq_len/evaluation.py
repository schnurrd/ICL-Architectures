from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from tqdm.auto import tqdm

from pfns.scripts.tabular_metrics import auc_metric
from pfns.training_utils import (
    categorical_mask_to_inds,
    compute_losses,
    move_style_and_check_shape,
    move_y_style_and_check_shape,
)
from pfns.utils import get_default_device, torch_nanmean

from .constants import MEMORY_NAMES, METRIC_NAMES, SCHEMA_VERSION, TIMING_NAMES
from .sampling import ClassCoverageBatchGenerator


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
    categorical_inds: list[int] | None,
) -> dict[str, float | None]:
    train_x = base_batch.x[:, :seqlen]
    train_y = base_batch.y[:, :seqlen]
    test_x = base_batch.x[:, largest_seqlen : largest_seqlen + number_of_test_samples]
    test_target_y = base_batch.target_y[
        :,
        largest_seqlen : largest_seqlen + number_of_test_samples,
    ]

    x_train = train_x.to(resolved_device)
    y_train = train_y.to(resolved_device)
    x_test = test_x.to(resolved_device)
    style = move_style_and_check_shape(base_batch.style, train_x, resolved_device)
    y_style = move_y_style_and_check_shape(base_batch.y_style, train_y, resolved_device)

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

    try:
        for _ in range(warmup_iters):
            warm_state = model.incontext_fit(**fit_kwargs)
            _ = model.incontext_predict(warm_state, **pred_kwargs)

        peak_allocated_mb = None
        peak_reserved_mb = None
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated() / (1024**2)
            base_reserved = torch.cuda.memory_reserved() / (1024**2)

        state, fit_ms = _timed(lambda: model.incontext_fit(**fit_kwargs), is_cuda=is_cuda)
        output, pred_ms = _timed(
            lambda: model.incontext_predict(state, **pred_kwargs),
            is_cuda=is_cuda,
        )
        forward_ms = fit_ms + pred_ms

        if is_cuda:
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
            peak_allocated_mb = max(0.0, peak_alloc - base_alloc)
            peak_reserved_mb = max(0.0, peak_reserved - base_reserved)

        context_size_mb = state.size_bytes() / (1024**2)

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
            probs = torch.softmax(output, dim=-1)[valid]
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
    autocast_models: set[str] | None = None,
    device: str | None = None,
    progress_desc: str = "Overall progress",
) -> dict[str, Any]:
    """Run sequence-length evaluation for all models and return nested result tables."""
    if not models:
        raise ValueError("No models provided.")
    if not seqlen_list:
        raise ValueError("seqlen_list cannot be empty.")

    resolved_device = device or get_default_device()
    device_type = torch.device(resolved_device).type
    is_cuda = device_type == "cuda"
    autocast_models = autocast_models or set()
    warmup_iters = 3 if use_warmup_iters else 0

    tables = BenchmarkTables.create(list(models))
    smallest, largest = min(seqlen_list), max(seqlen_list)

    def tprint(*args, **kwargs):
        if print_timing:
            print(*args, **kwargs)

    batch_generator = ClassCoverageBatchGenerator(
        num_batches=number_of_repetitions,
        largest_seqlen=largest,
        smallest_seqlen=smallest,
        num_features=num_features,
        num_classes=num_classes,
        number_of_test_samples=number_of_test_samples,
    )

    for rep, (base_batch, data_gen_ms) in enumerate(
        tqdm(batch_generator, total=number_of_repetitions, desc=progress_desc),
        start=1,
    ):
        tprint(f"Data generation rep {rep}: {data_gen_ms:.2f} ms")
        categorical_inds = categorical_mask_to_inds(base_batch.categorical_mask)

        for model_name, model in models.items():
            config = configs[model_name]
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device_type,
                    enabled=is_cuda and model_name in autocast_models,
                    dtype=torch.bfloat16,
                ):
                    for seqlen in seqlen_list:
                        if tables.should_skip_seqlen(model_name, seqlen):
                            continue

                        try:
                            result = _evaluate_model_at_seqlen(
                                model=model,
                                config=config,
                                base_batch=base_batch,
                                seqlen=seqlen,
                                largest_seqlen=largest,
                                number_of_test_samples=number_of_test_samples,
                                resolved_device=resolved_device,
                                is_cuda=is_cuda,
                                warmup_iters=warmup_iters,
                                categorical_inds=categorical_inds,
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

                        for metric_name in METRIC_NAMES:
                            tables.append_metric(model_name, metric_name, seqlen, result[metric_name])

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
        },
    }
