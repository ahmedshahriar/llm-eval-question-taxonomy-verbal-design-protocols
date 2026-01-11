"""Prediction serialization utilities."""

import json
import logging
from pathlib import Path

import pandas as pd

from application.constants import (
    HUMAN_LABEL_KEY,
    HUMAN_SUBCATEGORY_KEY,
    HUMAN_SUBCATEGORY_NORMALIZED_KEY,
    LLM_SUBCATEGORY_NORMALIZED_KEY,
    ORIGINAL_INDEX_KEY,
    PRED_LABEL_COL,
    PRED_SUBCATEGORY_COL,
    QUESTION_KEY,
)
from infrastructure.config import RunConfig

logger = logging.getLogger(__name__)


def attach_and_serialize_predictions(
    cfg: RunConfig,
    test_df: pd.DataFrame,
    question_col: str,
    label_col: str | None,
    pred_map: dict[int, str],
    subcat_map: dict[int, str | None] | None,
    predictions_path: Path,
) -> tuple[pd.DataFrame, Path]:
    """
    Attach predictions (and optional subcategories) to the DataFrame and write predictions_path as a compact JSON list.
    """
    test_df_out = test_df.copy()
    human_subcat_col = (
        cfg.columns.test_subcategory_col
        if (cfg.columns.test_subcategory_col and cfg.columns.test_subcategory_col in test_df_out.columns)
        else None
    )
    test_df_out[PRED_LABEL_COL] = test_df_out.index.map(lambda idx: pred_map.get(idx))

    if subcat_map is not None:
        # Only create the column if we are in subcategory mode
        test_df_out[PRED_SUBCATEGORY_COL] = test_df_out.index.map(lambda idx: subcat_map.get(idx))

    records: list[dict] = []
    for _, row in test_df_out.iterrows():
        record: dict[str, object] = {}

        if ORIGINAL_INDEX_KEY in test_df_out.columns:
            record[ORIGINAL_INDEX_KEY] = row[ORIGINAL_INDEX_KEY]

        record[QUESTION_KEY] = row[question_col]

        # Human top-level category (LLQ/DRQ/GDQ)
        record[HUMAN_LABEL_KEY] = row[label_col] if label_col is not None else None

        # Human subcategory (raw)
        if human_subcat_col is not None:
            raw_human_sub = row[human_subcat_col]
            record[HUMAN_SUBCATEGORY_KEY] = raw_human_sub
            # Normalized human subcategory (optional extra field)
            record[HUMAN_SUBCATEGORY_NORMALIZED_KEY] = cfg.taxonomy.normalize_subcategory(raw_human_sub)
        else:
            record[HUMAN_SUBCATEGORY_KEY] = None
            record[HUMAN_SUBCATEGORY_NORMALIZED_KEY] = None

        # LLM predictions
        record[PRED_LABEL_COL] = row[PRED_LABEL_COL]

        if PRED_SUBCATEGORY_COL in test_df_out.columns:
            raw_llm_sub = row[PRED_SUBCATEGORY_COL]
            record[PRED_SUBCATEGORY_COL] = raw_llm_sub
            record[LLM_SUBCATEGORY_NORMALIZED_KEY] = cfg.taxonomy.normalize_subcategory(raw_llm_sub)
        else:
            record[PRED_SUBCATEGORY_COL] = None
            record[LLM_SUBCATEGORY_NORMALIZED_KEY] = None

        records.append(record)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info("Saved predictions JSON: %s", predictions_path)

    return test_df_out, predictions_path
