#!/usr/bin/env python3
"""
Command-line interface for training PFNs models.
"""

import argparse
import importlib.util
import inspect
import os
import sys
from ast import literal_eval
from pathlib import Path

import pfns.train
from pfns.run_evaluation_cli import (
    run_evaluation,
    print_results_summary,
    summarize_results,
    compute_per_dataset_stats,
)
from pfns.utils import find_project_root, get_default_device
from pfns.run_logger import WandbConfig, create_run_manager, download_model_from_wandb
import wandb

REPO_ROOT = find_project_root(__file__)
DEFAULT_CHECKPOINT_PREFIX = str(REPO_ROOT / "PFNs" / "models_diff")
DEFAULT_WANDB_DIR = str(REPO_ROOT / "wandb")


def _normalize_wandb_run_path(path: str) -> str | None:
    """
    Accept common wandb run path formats and return a normalized path string:
    - entity/project/run_id
    - project/runs/run_id
    """
    parts = [p for p in path.strip("/").split("/") if p]
    if len(parts) != 3:
        return None
    if any(p in {".", ".."} for p in parts):
        return None
    if parts[1] == "runs":
        project, _, run_id = parts
        if not project or not run_id:
            return None
        return f"{project}/runs/{run_id}"

    entity, project, run_id = parts
    if project == "runs" or not entity or not project or not run_id:
        return None
    return f"{entity}/{project}/{run_id}"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a PFNs model using configuration from a Python file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the Python configuration file that defines a 'config' variable",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (e.g., 'cuda:0', 'cpu', 'mps'). If not specified, will auto-detect cuda, but not mps.",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for the model (requires PyTorch 2.0+).",
    )
    parser.set_defaults(compile=False)

    parser.add_argument(
        "--checkpoint-save-load-prefix",
        type=str,
        default=DEFAULT_CHECKPOINT_PREFIX,
        help="Path to save/load checkpoint (and default wandb dir).",
    )

    parser.add_argument(
        "--checkpoint-save-load-suffix",
        type=str,
        default="",
        help="Suffix to add to the checkpoint save/load path. This can e.g. be the seed.",
    )

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable wandb logging (configured via the config file).",
    )
    parser.add_argument(
        "--new-wandb-id",
        action="store_true",
        help="Start a new wandb run ID when resuming from a checkpoint.",
    )
    parser.add_argument(
        "--continue-from-wandb",
        type=str,
        default=None,
        metavar="RUN_PATH",
        help="Continue training from a wandb run (e.g., 'entity/project/run_id' or 'project/runs/run_id'). "
             "Downloads the checkpoint from wandb if not present locally.",
    )

    parser.add_argument(
        "--config-index",
        type=int,
        default=0,
        help="Index of the config to use. This is used to select a config from the config file.",
    )
    parser.add_argument(
        "--config-arg",
        action="append",
        default=[],
        type=parse_config_arg,
        metavar="KEY=VALUE",
        help="Extra get_config kwargs (repeatable), e.g. --config-arg masking='causal_train_only'.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, do not load an existing checkpoint/config even if present; start fresh and overwrite.",
    )
    parser.add_argument(
        "--train-mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override mixed precision setting after loading a checkpoint config.",
    )
    parser.add_argument(
        "--train-mixed-precision-dtype",
        type=str.lower,
        default=None,
        help=(
            "Override mixed precision dtype after loading a checkpoint config. "
            "Supported values include: auto, fp16, bf16, fp32 "
            "(aliases: float16, bfloat16, float32). "
            "'auto' selects bf16 on supported CUDA devices and fp32 otherwise."
        ),
    )

    return parser.parse_args()


def parse_config_arg(pair: str) -> tuple[str, object]:
    """Parse a single KEY=VALUE pair for --config-arg."""
    if "=" not in pair:
        raise argparse.ArgumentTypeError(
            f"Invalid --config-arg {pair!r}, expected KEY=VALUE."
        )
    key, value = pair.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            f"Invalid --config-arg {pair!r}, empty key."
        )
    value = value.strip()
    if value == "":
        return key, ""
    try:
        return key, literal_eval(value)
    except (ValueError, SyntaxError):
        return key, value


def load_config_from_python(
    config_file: str,
    config_index: int,
    *,
    config_kwargs: dict[str, object] | None = None,
    config_base_path: str | None = None,
) -> pfns.train.MainConfig:
    """Load MainConfig from a Python file by accessing the 'config' variable."""
    config_path = Path(config_file)
    if config_base_path is not None:
        config_path = Path(config_base_path) / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if not config_path.suffix.lower() == ".py":
        print(f"Warning: Config file {config_file} doesn't have .py extension")

    try:
        # Load the Python file as a module, creates a ModuleSpec instance
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load spec for {config_file}")

        config_module = importlib.util.module_from_spec(
            spec
        )  # Creates a new module based on spec

        # Add the config file's directory to sys.path temporarily
        config_dir = str(config_path.parent.absolute())
        if config_dir not in sys.path:
            sys.path.insert(0, config_dir)
            path_added = True
        else:
            path_added = False

        try:
            spec.loader.exec_module(config_module)

            if hasattr(config_module, "config") == hasattr(config_module, "get_config"):
                raise ValueError(
                    f"Config file {config_file} must define either a 'config' variable or a 'get_config' function, but not both."
                    f"It has {hasattr(config_module, 'config')=} and {hasattr(config_module, 'get_config')=}."
                )

            if hasattr(config_module, "get_config"):
                signature = inspect.signature(config_module.get_config)
                kwargs: dict[str, object] = {}
                if "config_index" in signature.parameters:
                    kwargs["config_index"] = config_index
                if config_kwargs:
                    for key, value in config_kwargs.items():
                        if key in signature.parameters:
                            kwargs[key] = value
                        else:
                            raise ValueError(
                                f"get_config does not accept {key!r}; remove or rename the argument."
                            )
                config = config_module.get_config(**kwargs)
            else:
                assert (
                    config_index == 0
                ), "config_index is not 0 but get_config is not defined"
                config = config_module.config

            # Validate that it is a MainConfig instance
            if not isinstance(config, pfns.train.MainConfig):
                raise TypeError(
                    f"'config' variable must be a MainConfig instance, got {config.__class__.__name__}"
                )

            print(f"Successfully loaded config from {config_file}")
            return config

        finally:
            # Remove the added path
            if path_added:
                sys.path.remove(config_dir)

    except Exception as e:
        raise ValueError(f"Failed to load config from {config_file}: {e}")


def _update_config(config: pfns.train.MainConfig, **updates) -> pfns.train.MainConfig:
    """Helper to update frozen config dataclass."""
    return config.__class__(**{**config.__dict__, **updates})


def _update_wandb(config: pfns.train.MainConfig, **updates) -> pfns.train.MainConfig:
    """Helper to update wandb config within MainConfig."""
    base = config.wandb or WandbConfig()
    return _update_config(config, wandb=WandbConfig(**{**base.__dict__, **updates}))


def _is_associative_recall_config(config: pfns.train.MainConfig) -> bool:
    """Detect whether the loaded config uses the associative recall prior."""
    for prior_cfg in config.priors:
        prior_names = getattr(prior_cfg, "prior_names", None)
        if isinstance(prior_names, str) and prior_names == "associative_recall":
            return True
        if isinstance(prior_names, (list, tuple)) and "associative_recall" in prior_names:
            return True
    return False


def _build_checkpoint_path(
    prefix: str,
    config_file: str,
    config_index: int,
    run_id: str,
    config_kwargs: dict | None = None,
    suffix: str = "",
) -> str:
    """Build checkpoint directory path: {prefix}/{config_name}_{index}[_{kwargs}][_{suffix}]_{run_id}
    
    Directory name is truncated to 255 bytes if needed (with hash to preserve uniqueness).
    """
    import hashlib
    
    config_name = config_file.split("/")[-1].split(".")[0]
    parts = [config_name, str(config_index)]
    
    # Add config kwargs to path
    if config_kwargs:
        for key in sorted(config_kwargs):
            value = config_kwargs[key]
            if value is not None:
                safe_val = str(value).replace(os.sep, "_").replace(" ", "")
                parts.append(f"{key}_{safe_val}")
    
    if suffix:
        parts.append(suffix)
    
    parts.append(run_id)
    
    dirname = "_".join(parts)
    
    max_len = 255
    if len(dirname.encode('utf-8')) > max_len:
        full_hash = hashlib.sha256(dirname.encode('utf-8')).hexdigest()[:8]
        
        # Format: {truncated_parts}_{hash}_{run_id}
        available_len = max_len - len(run_id.encode('utf-8')) - len(full_hash) - 2  # 2 underscores
        
        parts_without_runid = parts[:-1]
        truncated = "_".join(parts_without_runid).encode('utf-8')[:available_len].decode('utf-8', errors='ignore')
        dirname = f"{truncated}_{full_hash}_{run_id}"
    
    return os.path.join(prefix, dirname)


def main():
    """Main CLI entry point."""
    args = parse_args()
    config_kwargs = dict(args.config_arg)
    print("Loading configuration...")
    config = load_config_from_python(args.config_file, args.config_index, config_kwargs=config_kwargs)

    # --- Handle continuation from wandb ---
    if args.continue_from_wandb is not None:
        assert args.checkpoint_save_load_prefix, "--checkpoint-save-load-prefix required with --continue-from-wandb"

        normalized_run_path = _normalize_wandb_run_path(args.continue_from_wandb)
        if normalized_run_path is None:
            raise ValueError(
                "Invalid --continue-from-wandb path. "
                "Expected 'entity/project/run_id' or 'project/runs/run_id'."
            )
        run_id = normalized_run_path.rstrip("/").split("/")[-1]

        checkpoint_path = download_model_from_wandb(normalized_run_path)
        
        config = pfns.train.load_config(checkpoint_path)
        # overwrite save/load path to local path to avoid backwards compatibility issues
        config = _update_config(config, 
            train_state_dict_load_path=checkpoint_path,
            train_state_dict_save_path=checkpoint_path,
        )
        print(f"Continuing from wandb run: {run_id}")

    # --- Apply CLI overrides ---
    if args.wandb is not None:
        if args.wandb:
            if config.wandb is None:
                raise ValueError("--wandb requires wandb to be configured in config file")
            if config.wandb.mode == "disabled":
                config = _update_wandb(config, mode="online")
        else:
            config = _update_wandb(config, mode="disabled")

    if args.train_mixed_precision is not None:
        config = _update_config(config, train_mixed_precision=args.train_mixed_precision)
    if args.train_mixed_precision_dtype is not None:
        config = _update_config(
            config,
            train_mixed_precision_dtype=args.train_mixed_precision_dtype,
        )

    if config.wandb and not config.wandb.dir:
        config = _update_wandb(config, dir=DEFAULT_WANDB_DIR)

    # --- Initialize wandb ---
    run_manager = create_run_manager(config.wandb, full_config=config.to_dict(), run_id=config.wandb_run_id)
    
    if run_manager.run_id and config.wandb_run_id != run_manager.run_id:
        config = _update_config(config, wandb_run_id=run_manager.run_id)

    # --- Set up checkpoint paths ---
    if args.checkpoint_save_load_prefix and args.continue_from_wandb is None and config.train_state_dict_save_path is None:
        # Use wandb run_id if available, otherwise use fixed ID (allows auto-resume)
        if run_manager.run_id:
            run_id = run_manager.run_id
        else:
            run_id = "default"
            print(f"No wandb enabled - using fixed ID: {run_id} (will auto-resume from same path)")
            print(f"Consider enabling wandb or using unique --checkpoint-save-load-suffix to avoid overwriting checkpoints.")
        
        path = _build_checkpoint_path(
            args.checkpoint_save_load_prefix, args.config_file, args.config_index,
            run_id, config_kwargs, args.checkpoint_save_load_suffix,
        )
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        config = _update_config(config,
            train_state_dict_save_path=checkpoint_path,
            train_state_dict_load_path=checkpoint_path,
        )
        if config.wandb and not config.wandb.dir:
            config = _update_wandb(config, dir=path)
        
        os.makedirs(path, exist_ok=True)
        print(f"Checkpoint path: {checkpoint_path}")

    # --- Update wandb config with all changes ---
    if run_manager.run_id:
        wandb.config.update(config.to_dict(), allow_val_change=True)

    if run_manager.run_id:
        print(f"Wandb run: {run_manager.run_id}")

    should_run_eval = (
        config.train_state_dict_save_path is not None
        and not _is_associative_recall_config(config)
    )
    
    print(f"Starting / Continuing training:")
    print(f"  Epochs: {config.epochs} epochs")
    print(f"  Steps / epoch: {config.steps_per_epoch}")
    print(f"  Test steps / epoch: {config.test_steps_per_epoch}")
    print(f"  Device: {args.device or 'auto-detect'}")
    print(f"  Mixed precision: {config.train_mixed_precision}")
    print(f"  Mixed precision dtype: {config.train_mixed_precision_dtype}")
    
    try:
        result = pfns.train.train(
            c=config,
            device=args.device,
            compile=args.compile,
            overwrite=args.overwrite,
            logger=run_manager,
            finish_logger=False,
        )

        print("\nTraining completed successfully!")
        print(f"Total training time: {result['total_time']:.2f} seconds")
        print(f"Final loss: {result['total_loss']:.6f}")

        if config.train_state_dict_save_path is not None:
            print(f"Model saved to: {config.train_state_dict_save_path}")
            run_manager.save_model(config.train_state_dict_save_path)

        if not should_run_eval:
            if config.train_state_dict_save_path is None:
                print(
                    "Skipping automatic evaluation because no train_state_dict_save_path was provided."
                )
            else:
                print(
                    "Skipping automatic evaluation because associative recall mode is active."
                )
            return

        base_path = os.path.dirname(config.train_state_dict_save_path)
        checkpoint_name = os.path.basename(config.train_state_dict_save_path)
        eval_device = args.device or get_default_device()

        print(
            "\nStarting automatic evaluation on OpenCC benchmark with TabPFN only "
            "(n_splits=5, batch_size_inference=32, preprocess_transforms=['none','power'], "
            "max_features=20, max_samples=1000, max_classes=10)..."
        )

        results = run_evaluation(
            runner="tabpfn",
            model_config={
                "base_path": base_path,
                "checkpoint_name": checkpoint_name,
            },
            device=eval_device,
            benchmark="opencc",
            max_samples=1000,
            max_features=20,
            max_classes=10,
            n_splits=5,
            n_jobs=4,
            random_state=42,
            batch_size_inference=32,
            n_ensemble_configurations=10,
            preprocess_transforms=["none", "power"],
            sample_order_permutation=True,
        )

        print_results_summary(
            results,
            title="Aggregated Results Across All Datasets (TabPFN only)",
        )
        summary = summarize_results(results)
        per_dataset = compute_per_dataset_stats(results)
        run_manager.log_evaluation(
            results=results,
            summary=summary,
            per_dataset=per_dataset,
        )
    finally:
        run_manager.finish()


if __name__ == "__main__":
    main()
