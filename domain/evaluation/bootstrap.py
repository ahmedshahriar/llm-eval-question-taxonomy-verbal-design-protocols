"""Bootstrap confidence interval computation."""

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap CI for a statistic (e.g., accuracy, kappa).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        stat_fn: Function that computes a metric from (y_true, y_pred)
        n_boot: Number of bootstrap samples
        alpha: Significance level (e.g., 0.05 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower, upper) confidence interval bounds
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)
    stats = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample_idx = rng.choice(idx, size=n, replace=True)
        with warnings.catch_warnings():
            # Suppress expected warnings from degenerate bootstrap samples
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="y_pred contains classes not in y_true",
            )
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            stats[b] = stat_fn(y_true[sample_idx], y_pred[sample_idx])

    lower = float(np.percentile(stats, 100 * (alpha / 2)))
    upper = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lower, upper
