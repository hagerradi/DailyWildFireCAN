"""
Utility helpers shared across the project.
"""
from __future__ import annotations

import os
import random

import torch
import numpy as np


# Test folder names used by the Hugging Face archives -> names used in the paper.
SPLIT_ALIASES = {"test_easy": "test_time", "test_medium": "test_space", "test_hard": "test_spacetime"}


def resolve_split_dir(base_data_path: str, split: str) -> str:
    """Path of ``split`` in ``base_data_path``, accepting both naming schemes.

    ``test_easy``/``test_medium``/``test_hard`` fall back to ``test_time``/``test_space``/
    ``test_spacetime`` (the folder names in the released archives) when only those exist.
    """
    path = os.path.join(base_data_path, split)
    alias = SPLIT_ALIASES.get(split)
    if not os.path.isdir(path) and alias and os.path.isdir(os.path.join(base_data_path, alias)):
        return os.path.join(base_data_path, alias)
    return path


def seed_everything(seed: int) -> None:
    """Fix the random seed for full reproducibility across all devices and backends.

    Sets the seed for Python's built-in :mod:`random`, the ``PYTHONHASHSEED``
    environment variable, PyTorch (CPU and all CUDA devices), and enables
    CuDNN deterministic mode so that convolution algorithms are chosen
    deterministically.

    .. note::
        ``torch.backends.cudnn.deterministic = True`` and
        ``torch.backends.cudnn.benchmark = False`` guarantee reproducible
        results but may reduce throughput on fixed-size inputs where CuDNN
        auto-tuning would otherwise select faster kernels.

    Args:
        seed: Integer seed value to use everywhere.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False