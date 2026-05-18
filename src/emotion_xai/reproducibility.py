"""Reproducibility helpers shared by training and XAI scripts."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Global random seed used across data splitting, training, and XAI sampling.
    deterministic:
        If True, requests deterministic PyTorch algorithms where possible. This can
        reduce speed and may fail for some GPU operations, so it is disabled by
        default for the original research workflow.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
