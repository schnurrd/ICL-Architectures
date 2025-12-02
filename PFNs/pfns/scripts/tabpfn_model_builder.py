import os
from functools import partial

import torch

from pfns.base_config import BaseConfig
from pfns.train import MainConfig


def _strip_compiled_model_prefix(state_dict):
    """
    Strips the '_orig_mod.' prefix from state dict keys that were saved from a torch.compile() model.
    """
    prefix = "_orig_mod."
    return {k.removeprefix(prefix): v for k, v in state_dict.items()}


def load_model_only_inference(path, filename, device="cpu"):
    """
    Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """

    checkpoint = torch.load(os.path.join(path, filename), map_location="cpu")

    if "config" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing the serialized training config under key 'config'."
        )

    config: BaseConfig = MainConfig.from_dict(checkpoint["config"])

    model = config.model.create_model()
    model_state = checkpoint["model_state_dict"]

    # Handle state dicts saved from torch.compile() models
    model_state = _strip_compiled_model_prefix(model_state)

    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model, config
