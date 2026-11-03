"""Reproducibility, device selection and checkpoint I/O."""

from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from catdog import CLASS_NAMES
from catdog.models import MODEL_NAMES, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS requested but not available")
    return device


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    model_name: str,
    image_size: int,
    epoch: int,
    validation_metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # File handles also work with non-ASCII Windows paths on older PyTorch builds.
    with path.open("wb") as checkpoint_file:
        torch.save(
            {
                "model_name": model_name,
                "image_size": image_size,
                "class_names": list(CLASS_NAMES),
                "epoch": epoch,
                "validation_metrics": validation_metrics,
                "model_state": model.state_dict(),
            },
            checkpoint_file,
        )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    model_name = checkpoint["model_name"]
    image_size = checkpoint["image_size"]
    if model_name not in MODEL_NAMES or not isinstance(image_size, int) or image_size < 8:
        raise ValueError(f"Invalid checkpoint model metadata: {checkpoint_path}")
    if tuple(checkpoint["class_names"]) != CLASS_NAMES:
        raise ValueError(f"Checkpoint class order differs from {CLASS_NAMES}")
    model = build_model(model_name, image_size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
