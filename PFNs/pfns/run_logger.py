from __future__ import annotations

import os
import typing as tp
from dataclasses import dataclass

from . import base_config


@dataclass(frozen=True)
class WandbConfig(base_config.BaseConfig):
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] | None = None
    mode: tp.Literal["online", "offline", "disabled"] = "online"
    dir: str | None = None
    log_every_n_steps: int = 10
    resume: str | None = None


class RunManager(tp.Protocol):
    run_id: str | None

    def log(self, metrics: dict[str, tp.Any], *, step: int | None = None) -> None: ...

    def log_evaluation(
        self,
        *,
        results: tp.Any,
        summary: tp.Any,
        per_dataset: tp.Any,
    ) -> None: ...

    def finish(self) -> None: ...


class NullRunManager:
    run_id = None

    def log(self, metrics: dict[str, tp.Any], *, step: int | None = None) -> None:
        return

    def log_evaluation(
        self,
        *,
        results: tp.Any,
        summary: tp.Any,
        per_dataset: tp.Any,
    ) -> None:
        return

    def finish(self) -> None:
        return


def _init_wandb_run(
    wandb_config: WandbConfig,
    *,
    full_config: dict[str, tp.Any] | None = None,
    run_id: str | None = None,
    resume: str | None = None,
):
    import wandb

    init_kwargs: dict[str, tp.Any] = {
        "project": wandb_config.project,
        "entity": wandb_config.entity,
        "name": wandb_config.name,
        "group": wandb_config.group,
        "tags": wandb_config.tags,
        "mode": wandb_config.mode,
        "dir": wandb_config.dir,
    }
    if full_config is not None:
        init_kwargs["config"] = full_config
    if run_id is not None:
        init_kwargs["id"] = run_id
    if resume is not None:
        init_kwargs["resume"] = resume
    elif wandb_config.resume is not None:
        init_kwargs["resume"] = wandb_config.resume

    return wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})


def _is_main_process() -> bool:
    if "LOCAL_RANK" in os.environ:
        return os.environ.get("LOCAL_RANK", "0") == "0"
    if "SLURM_PROCID" in os.environ:
        return os.environ.get("SLURM_PROCID", "0") == "0"
    if "RANK" in os.environ:
        return os.environ.get("RANK", "0") == "0"
    return True


class WandbRunManager:
    def __init__(
        self,
        wandb_config: WandbConfig,
        *,
        full_config: dict[str, tp.Any],
        run_id: str | None = None,
    ) -> None:
        if wandb_config.mode == "disabled":
            raise ValueError("WandbRunManager cannot be initialized with mode='disabled'.")

        import wandb

        self._wandb = wandb
        resume = "allow" if run_id is not None else None
        self._run = _init_wandb_run(
            wandb_config,
            full_config=full_config,
            run_id=run_id,
            resume=resume,
        )
        self.run_id = None if self._run is None else self._run.id

        wandb.define_metric("trainer/global_step")
        wandb.define_metric("trainer/epoch")
        wandb.define_metric("step/*", step_metric="trainer/global_step")
        wandb.define_metric("epoch/*", step_metric="trainer/epoch")

    def log(self, metrics: dict[str, tp.Any], *, step: int | None = None) -> None:
        self._wandb.log(metrics, step=step)

    def log_evaluation(
        self,
        *,
        results: tp.Any,
        summary: tp.Any,
        per_dataset: tp.Any,
    ) -> None:
        if self._run is None:
            print("wandb.init returned None; skipping evaluation logging.")
            return

        if summary is not None and not summary.empty:
            metrics = {}
            for model in summary.index:
                safe_model = model.replace(" ", "_").replace("/", "_")
                row = summary.loc[model]
                metrics[f"eval/{safe_model}/accuracy_mean"] = float(row["accuracy_mean"])
                metrics[f"eval/{safe_model}/accuracy_std"] = float(row["accuracy_std"])
                metrics[f"eval/{safe_model}/roc_auc_mean"] = float(row["roc_auc_mean"])
                metrics[f"eval/{safe_model}/roc_auc_std"] = float(row["roc_auc_std"])
                metrics[f"eval/{safe_model}/fit_time_mean"] = float(row["fit_time_mean"])
                metrics[f"eval/{safe_model}/predict_time_mean"] = float(row["predict_time_mean"])
            self._wandb.log(metrics)

        if per_dataset is not None and not per_dataset.empty:
            self._wandb.log({"eval/per_dataset": self._wandb.Table(dataframe=per_dataset)})

        if results is not None and not results.empty:
            self._wandb.log({"eval/raw_results": self._wandb.Table(dataframe=results)})

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


def create_run_manager(
    wandb_config: WandbConfig | None,
    *,
    full_config: dict[str, tp.Any],
    run_id: str | None = None,
) -> RunManager:
    if wandb_config is None or wandb_config.mode == "disabled":
        return NullRunManager()
    if not _is_main_process():
        return NullRunManager()
    return WandbRunManager(wandb_config, full_config=full_config, run_id=run_id)
