"""Data batching and grouping utilities."""

import logging

import pandas as pd

from infrastructure.config.models import RunConfig

logger = logging.getLogger(__name__)


def detect_question_and_label_columns_from_test_data(
    cfg: RunConfig,
    df: pd.DataFrame,
) -> tuple[str, str | None]:
    """
    Resolve the question column (required) and the human label column (optional) from config for test data.

    Args:
        cfg: RunConfig instance
        df: Test DataFrame

    Returns:
        Tuple of (question_col, label_col) where label_col may be None

    Raises:
        KeyError: If configured question column not found in DataFrame
    """
    question_col = cfg.columns.test_question_col
    label_col = cfg.columns.test_category_col

    if question_col not in df.columns:
        raise KeyError(
            f"Configured test_question_col='{question_col}' not found in test dataset columns: {list(df.columns)}"
        )

    # Human label col is optional
    if label_col is None or not str(label_col).strip():
        return question_col, None

    if label_col not in df.columns:
        logger.warning(
            "Configured test_category_col='%s' not found in test dataset. "
            "Proceeding with label_col=None (inference only; evaluation will be skipped).",
            label_col,
        )
        return question_col, None

    return question_col, label_col


def group_test_dataframe(test_df: pd.DataFrame, group_col: str | None) -> list[tuple[str, pd.DataFrame]]:
    """
    Group test data by group_col if present; otherwise return a single group.

    Args:
        test_df: Test DataFrame
        group_col: Optional column name to group by

    Returns:
        List of (group_name, group_dataframe) tuples
    """
    if not group_col or group_col not in test_df.columns:
        return [("All", test_df)]

    group_order: list = []
    seen = set()
    for v in test_df[group_col]:
        if v not in seen:
            group_order.append(v)
            seen.add(v)

    groups: list[tuple[str, pd.DataFrame]] = []
    for v in group_order:
        group_df = test_df[test_df[group_col] == v]
        groups.append((str(v), group_df))

    return groups


def iter_segments(n_items: int, batch_size: int | None) -> list[tuple[int, int]]:
    """
    Return (start, end) index pairs for segmenting a list of length n_items.

    Args:
        n_items: Total number of items to segment
        batch_size: Size of each batch, or None to return entire range

    Returns:
        List of (start, end) index pairs
    """
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or None")

    if batch_size is None:
        return [(0, n_items)]

    segments: list[tuple[int, int]] = []
    start = 0
    while start < n_items:
        end = min(start + batch_size, n_items)
        segments.append((start, end))
        start = end
    return segments
