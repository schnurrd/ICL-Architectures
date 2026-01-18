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
    run_tabpfn_evaluation,
    print_results_summary,
    summarize_results,
    compute_per_dataset_stats,
)
from pfns.utils import get_default_device
from pfns.run_logger import WandbConfig, create_run_manager


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
        default=None,
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


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Load configuration from Python file
    config_kwargs = dict(args.config_arg)
    config = load_config_from_python(
        args.config_file,
        args.config_index,
        config_kwargs=config_kwargs,
    )

    def get_filename(config_file):
        return f"{config_file.split('/')[-1].split('.')[0]}"

    def format_config_suffix(kwargs: dict[str, object]) -> str:
        if not kwargs:
            return ""
        parts = []
        for key in sorted(kwargs):
            value = kwargs[key]
            if value is None:
                continue
            safe_key = str(key).replace(os.sep, "_")
            safe_value = str(value).replace(os.sep, "_").replace(" ", "")
            parts.append(f"{safe_key}_{safe_value}")
        return "_" + "_".join(parts) if parts else ""

    if args.checkpoint_save_load_suffix:
        assert (
            args.checkpoint_save_load_prefix is not None
        ), "checkpoint_save_load_prefix is required when checkpoint_save_load_suffix is provided"

    # Override checkpoint paths if specified via CLI
    if args.checkpoint_save_load_prefix is not None:
        assert (
            config.train_state_dict_save_path is None
        ), "train_state_dict_save_path is already set"
        assert (
            config.train_state_dict_load_path is None
        ), "train_state_dict_load_path is already set"

        # Add suffix if it exists
        suffix = f"_{args.config_index}"
        suffix += format_config_suffix(config_kwargs)
        if args.checkpoint_save_load_suffix:
            suffix += f"_{args.checkpoint_save_load_suffix}"

        path = f"{args.checkpoint_save_load_prefix}/{get_filename(args.config_file)}{suffix}"

        config = config.__class__(
            **{
                **config.__dict__,
                "train_state_dict_save_path": path + "/checkpoint.pt",
                "train_state_dict_load_path": path + "/checkpoint.pt",
            }
        )
        os.makedirs(path, exist_ok=True)

    # If no wandb dir is set but we have a train_state_dict_save_path, set wandb dir to this directory
    if config.wandb is not None and config.wandb.dir is None and config.train_state_dict_save_path is not None:
        config = config.__class__(
            **{
                **config.__dict__,
                "wandb": WandbConfig(
                    **{
                        **config.wandb.__dict__,
                        "dir": os.path.dirname(config.train_state_dict_save_path),
                    }
                ),
            }
        )

    if args.wandb is not None: # Wandb CLI flag overrides config file
        if args.wandb:  # enable wandb if requrested via cli
            if config.wandb is None:
                raise ValueError(
                    "--wandb was set, but config.wandb is None. Configure wandb in the config file."
                )
            if config.wandb.mode == "disabled":  # overrides config file
                config = config.__class__(
                    **{
                        **config.__dict__,
                        "wandb": WandbConfig(
                            **{
                                **config.wandb.__dict__,
                                "mode": "online",
                            }
                        ),
                    }
                )
        else:  # disable wandb if requested via cli
            existing = config.wandb or WandbConfig()
            config = config.__class__(
                **{
                    **config.__dict__,
                    "wandb": WandbConfig(**{**existing.__dict__, "mode": "disabled"}),
                }
            )

    # We overwrite the config with the one from the checkpoint if it exists
    # as there is some randomness in the config and we want to use the exact
    # same config again. When --overwrite is set, skip loading so we start fresh.
    if not args.overwrite and pfns.train.should_load_checkpoint(config):
        config = pfns.train.load_config(
            config.train_state_dict_load_path,
        )
        if args.new_wandb_id and config.wandb_run_id is not None:
            config = config.__class__(**{**config.__dict__, "wandb_run_id": None})

    if args.train_mixed_precision is not None:
        config = config.__class__(
            **{
                **config.__dict__,
                "train_mixed_precision": args.train_mixed_precision,
            }
        )

    print("Starting training with configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Steps per epoch: {config.steps_per_epoch}")
    print(f"  Device: {args.device or 'auto-detect'}")
    print(f"  Mixed precision: {config.train_mixed_precision}")

    run_manager = create_run_manager(
        config.wandb,
        full_config=config.to_dict(),
        run_id=config.wandb_run_id,
    )
    if run_manager.run_id is not None and config.wandb_run_id != run_manager.run_id:
        config = config.__class__(
            **{**config.__dict__, "wandb_run_id": run_manager.run_id}
        )
    should_run_eval = config.train_state_dict_save_path is not None
    if run_manager.run_id is not None:
        print("wandb logging enabled.")

    try:
        result = pfns.train.train(
            c=config,
            device=args.device,
            compile=args.compile,
            overwrite=args.overwrite,
            logger=run_manager,
            finish_logger=not should_run_eval,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)

    print("\nTraining completed successfully!")
    print(f"Total training time: {result['total_time']:.2f} seconds")
    print(f"Final loss: {result['total_loss']:.6f}")

    if config.train_state_dict_save_path is not None:
        print(f"Model saved to: {config.train_state_dict_save_path}")

    if config.train_state_dict_save_path is None:
        print(
            "Skipping automatic evaluation because no train_state_dict_save_path was provided."
        )
        return

    base_path = os.path.dirname(config.train_state_dict_save_path)
    checkpoint_name = os.path.basename(config.train_state_dict_save_path)
    eval_device = args.device or get_default_device()

    print(
        "\nStarting automatic evaluation on OpenCC benchmark with TabPFN only "
        "(n_splits=5, batch_size_inference=16, preprocess_transforms=['none','power','robust'], "
        "max_features=20, max_samples=1000, max_classes=10)..."
    )

    try:
        results = run_tabpfn_evaluation(
            base_path=base_path,
            checkpoint_name=checkpoint_name,
            device=eval_device,
            benchmark="opencc",
            max_samples=1000,
            max_features=20,
            max_classes=10,
            n_splits=5,
            only_tabpfn=True,
            n_jobs=4,
            batch_size_inference=16,
            n_ensemble_configurations=32,
            preprocess_transforms=["none", "power", "robust"],
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
