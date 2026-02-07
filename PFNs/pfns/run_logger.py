from __future__ import annotations

import os

import torch

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

    def save_model(self, file_path: str, aliases: list[str] | None = None) -> None: ...

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

    def save_model(self, file_path: str, aliases: list[str] | None = None) -> None:
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

    return wandb.init(
        **{k: v for k, v in init_kwargs.items() if v is not None}
    )


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
        resume = "must" if run_id is not None else None
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

        if self._run is not None:
             self._run.summary.update(metrics)

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

    def save_model(self, file_path: str, aliases: list[str] | None = None) -> None:
        if self._run is None:
            return

        artifact = self._wandb.Artifact(
            name=f"model-{self.run_id}",
            type="model",
            description=f"Trained model checkpoint",
        )
        artifact.add_file(file_path)
        self._run.log_artifact(artifact, aliases=aliases or [])
        print(f"WandbRunManager: Saved model artifact to wandb with aliases {aliases} under run id {self.run_id}.")

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


def download_model_from_wandb(
    run_path: str,
    destination_path: str | None = None,
) -> str:
    import wandb
    import shutil
    import tempfile
    
    print(f"Attempting to download model from wandb run: {run_path}")
    api = wandb.Api()
    run = api.run(run_path)
    
    if destination_path is None:
        destination_path = run.config['train_state_dict_save_path']
    
    destination_dir = os.path.dirname(destination_path)
    
    if os.path.exists(destination_path):
        try:
            model_data = torch.load(destination_path, map_location='cpu')
            
            local_run_id = model_data['config']['wandb_run_id']
            local_epoch = model_data.get('epoch')
            remote_epoch = run.summary.get("trainer/epoch")
            
            if local_run_id == run.id and local_epoch is not None and (local_epoch == remote_epoch or local_epoch == remote_epoch - 1): # can be off by 1 when interupted mid epoch
                print(f"Model at {destination_path} is already up to date (Run ID: {local_run_id}, Epoch: {local_epoch}). Skipping download.")
                return destination_path
            
        except Exception as e:
            print(f"Could not verify existing model (Error: {e}). Proceeding locally.")
            pass
        os.remove(destination_path) # remove outdated or invalid file
    
    artifacts = run.logged_artifacts()
    for artifact in artifacts:
        if artifact.type == "model" and "latest" in artifact.aliases:
            print(f"Found model artifact: {artifact.name}")
            with tempfile.TemporaryDirectory(prefix="wandb_model_downloads_") as tmp_dir:
                download_dir = artifact.download(root=tmp_dir)
                for root, _, files in os.walk(download_dir):
                    for file in files:
                        if file.endswith(".pt") or len(files) == 1:
                            src = os.path.join(root, file)
                            os.makedirs(destination_dir, exist_ok=True)
                            shutil.copy2(src, destination_path)
                            print(f"Downloaded artifact file {file} to {destination_path}")
                            return destination_path

    raise FileNotFoundError(f"No suitable model artifact found in run {run_path}.")
