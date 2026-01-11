"""Random seed configuration for reproducibility."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and hash-based operations.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
