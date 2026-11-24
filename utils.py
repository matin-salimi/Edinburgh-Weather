"""Small shared helpers used by several scripts."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from config import TARGET_INDEX


def set_seed(seed: int) -> None:
    """Make Python, NumPy and PyTorch random number generators reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def to_celsius(scaled_values, scaler):
    """Convert standardized temperature values back to degrees Celsius."""
    return (
        scaled_values * scaler.scale_[TARGET_INDEX]
        + scaler.mean_[TARGET_INDEX]
    )


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)