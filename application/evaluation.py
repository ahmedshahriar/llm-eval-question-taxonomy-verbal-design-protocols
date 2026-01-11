"""Evaluation workflow and summary logging."""

import logging
from pathlib import Path

import pandas as pd
from opik import track

from application.constants import PRED_LABEL_COL, PRED_SUBCATEGORY_COL
from domain.evaluation.metrics import compute_classification_metrics
from infrastructure.config.models import RunConfig

logger = logging.getLogger(__name__)


@track(
    name="Question.labelling.evaluation",
    type="general",
    metadata={"task": "question_labelling_evaluation"},
)
def run_evaluation_if_labels_available(
    cfg: RunConfig,
    test_df_out: pd.DataFrame,
    label_col: str | None,
) -> tuple[dict, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Compute metrics if human labels are present; otherwise return empty metrics.

    Top-level metrics (LLQ/DRQ/GDQ) are always computed when label_col is provided.
    Subcategory metrics are computed only when:
      - label_subcategories == True
      - test_subcategory_col exists in test_df_out
      - predicted subcategory column exists and is not entirely missing


    Args:
        cfg: RunConfig instance
        test_df_out: DataFrame with predictions attached
        label_col: Name of human label column (optional)

    Returns:
        Tuple of (metrics dict, confusion_matrix DataFrame, subcategory confusion_matrix DataFrame)
    """

    if label_col is None:
        logger.info("No human labels provided; skipping metric computation.")
        return {}, None, None

    # ----- Top-level metrics -----
    y_true = test_df_out[label_col].fillna("").astype(str).str.strip().str.upper().values
    y_pred = test_df_out[PRED_LABEL_COL].fillna("").astype(str).str.strip().str.upper().values

    metrics, cm_df = compute_classification_metrics(y_true, y_pred, cfg.stats.labels_order, cfg.stats)

    sub_cm_df: pd.DataFrame | None = None
    logger.info("Accuracy: %.4f", metrics["accuracy"])

    # ----- Optional: subcategory metrics -----
    if cfg.label_subcategories:
        human_subcat_col = (
            cfg.columns.test_subcategory_col
            if (cfg.columns.test_subcategory_col and cfg.columns.test_subcategory_col in test_df_out.columns)
            else None
        )

        if (
            human_subcat_col is not None
            and PRED_SUBCATEGORY_COL in test_df_out.columns
            and test_df_out[PRED_SUBCATEGORY_COL].notna().any()
        ):
            y_true_sub = test_df_out[human_subcat_col].apply(cfg.taxonomy.normalize_subcategory).values
            y_pred_sub = test_df_out[PRED_SUBCATEGORY_COL].apply(cfg.taxonomy.normalize_subcategory).values

            # Derive the label set from data
            sub_labels_set = set(y_true_sub.tolist()) | set(y_pred_sub.tolist())
            sub_labels_list = sorted(lbl for lbl in sub_labels_set if lbl not in {"", "nan", "None"})

            if sub_labels_list:
                sub_metrics, sub_cm_df = compute_classification_metrics(
                    y_true_sub, y_pred_sub, sub_labels_list, cfg.stats
                )

                # Prefix with 'sub_' so they don't collide with top-level metrics
                for key, value in sub_metrics.items():
                    metrics[f"sub_{key}"] = value
        else:
            logger.info(
                "label_subcategories=True but '%s' or '%s' is missing; skipping subcategory metrics.",
                human_subcat_col,
                PRED_SUBCATEGORY_COL,
            )

    return metrics, cm_df, sub_cm_df


def log_evaluation_summary(
    metrics: dict,
    cm_df: pd.DataFrame | None,
    sub_cm_df: pd.DataFrame | None,
    label_col: str | None,
    usage_stats: dict[str, int | float],
    predictions_path: Path,
    metrics_path: Path,
) -> None:
    """
    Log a concise, human-readable evaluation summary.

    Args:
        metrics: Dictionary of computed metrics
        cm_df: Confusion matrix DataFrame (optional)
        sub_cm_df: Subcategory confusion matrix DataFrame (optional)
        label_col: Name of human label column (optional)
        usage_stats: Dictionary of token usage and costs
        predictions_path: Path to predictions JSON file
        metrics_path: Path to metrics JSON file
    """
    logger.info("=== Evaluation Summary ===")

    if label_col is not None and cm_df is not None:
        logger.debug("Confusion matrix (rows=true, cols=pred):\n%s", cm_df)
        logger.info(
            "Accuracy: %.4f (95%% CI [%.4f, %.4f])",
            metrics["accuracy"],
            metrics["accuracy_ci_95"][0],
            metrics["accuracy_ci_95"][1],
        )
        logger.info(
            "Balanced accuracy: %.4f (95%% CI [%.4f, %.4f])",
            metrics["balanced_accuracy"],
            metrics["balanced_accuracy_ci_95"][0],
            metrics["balanced_accuracy_ci_95"][1],
        )
        logger.info(
            "Macro F1: %.4f (95%% CI [%.4f, %.4f])",
            metrics["macro_f1"],
            metrics["macro_f1_ci_95"][0],
            metrics["macro_f1_ci_95"][1],
        )
        logger.info("Weighted F1: %.4f", metrics["weighted_f1"])
        logger.info(
            "Cohen's kappa: %.4f (95%% CI [%.4f, %.4f])",
            metrics["cohen_kappa"],
            metrics["cohen_kappa_ci_95"][0],
            metrics["cohen_kappa_ci_95"][1],
        )

        # per-label/category metrics
        logger.info("Per-class precision: %s", metrics["precision_per_class"])
        logger.info("Per-class recall: %s", metrics["recall_per_class"])
        logger.info("Per-class F1: %s", metrics["f1_per_class"])
        logger.info("Per-class support: %s", metrics["support_per_class"])

        # ----- Subcategory metrics (optional) -----
        # any(k.startswith("sub_") for k in (metrics or {}).keys())
        if sub_cm_df is not None and "sub_accuracy" in metrics:
            logger.info("--- Subcategory metrics ---")
            logger.debug(
                "Subcategory confusion matrix (rows=true, cols=pred):\n%s",
                sub_cm_df,
            )
            logger.info(
                "Subcategory accuracy: %.4f (95%% CI [%.4f, %.4f])",
                metrics["sub_accuracy"],
                metrics["sub_accuracy_ci_95"][0],
                metrics["sub_accuracy_ci_95"][1],
            )
            logger.info(
                "Subcategory macro F1: %.4f (95%% CI [%.4f, %.4f])",
                metrics["sub_macro_f1"],
                metrics["sub_macro_f1_ci_95"][0],
                metrics["sub_macro_f1_ci_95"][1],
            )
            logger.info(
                "Subcategory Cohen's kappa: %.4f (95%% CI [%.4f, %.4f])",
                metrics["sub_cohen_kappa"],
                metrics["sub_cohen_kappa_ci_95"][0],
                metrics["sub_cohen_kappa_ci_95"][1],
            )
    else:
        logger.info("No human labels provided; skipped metric computation.")

    logger.info("--- Usage/Cost ---")
    logger.info(
        "Token usage: input=%d, output=%d, total=%d",
        usage_stats["total_input_tokens"],
        usage_stats["total_output_tokens"],
        usage_stats["total_tokens"],
    )
    # if usage_stats.get("cost_tracking_enabled", True):
    #     logger.info(
    #         "Estimated cost: input=$%.6f, output=$%.6f, total=$%.6f",
    #         usage_stats["total_input_cost_usd"],
    #         usage_stats["total_output_cost_usd"],
    #         usage_stats["total_cost_usd"],
    #     )
    # else:
    #     logger.info("Estimated cost: disabled (pricing env vars missing/invalid).")
    logger.info(
        "Estimated cost: input=$%.6f, output=$%.6f, total=$%.6f",
        usage_stats["total_input_cost_usd"],
        usage_stats["total_output_cost_usd"],
        usage_stats["total_cost_usd"],
    )

    logger.info("--- Artifacts ---")
    logger.info("Predictions JSON: %s", predictions_path)
    logger.info("Metrics JSON: %s", metrics_path)
