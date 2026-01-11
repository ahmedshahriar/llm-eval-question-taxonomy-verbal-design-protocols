"""Classification metrics computation with confidence intervals."""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from opik import track
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from domain.evaluation.bootstrap import bootstrap_ci
from infrastructure.config.models import StatsConfig


@track(
    name="LLM Question Labelling Metrics Computation",
    type="general",
    metadata={"task": "question_labelling_metrics"},
)
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels_order: Sequence[str],
    stats_cfg: StatsConfig,
) -> tuple[dict, pd.DataFrame]:
    """
    Compute confusion matrix and a suite of metrics, including CIs.

    Args:
        y_true: True labels array
        y_pred: Predicted labels array
        labels_order: Canonical ordering of labels (e.g., ['LLQ', 'DRQ', 'GDQ'])
        stats_cfg: Statistics configuration (seed, n_boot, alpha)

    Returns:
        Tuple of (metrics dict, confusion_matrix DataFrame)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="y_pred contains classes not in y_true",
        )
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in labels_order],
            columns=[f"pred_{label}" for label in labels_order],
        )

        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, labels=labels_order)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=labels_order, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, labels=labels_order, average="weighted")

        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels_order,
            zero_division=0,
        )

    precision_per_class = dict(zip(labels_order, np.round(precision.astype(float), 4).tolist(), strict=False))
    recall_per_class = dict(zip(labels_order, np.round(recall.astype(float), 4).tolist(), strict=False))
    f1_per_class_dict = dict(zip(labels_order, np.round(f1_per_class.astype(float), 4).tolist(), strict=False))
    support_per_class = dict(zip(labels_order, support.tolist(), strict=False))

    # Bootstrap CIs
    kappa_ci_low, kappa_ci_high = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        stat_fn=lambda yt, yp: cohen_kappa_score(yt, yp, labels=labels_order),
        n_boot=stats_cfg.n_boot,
        alpha=stats_cfg.alpha,
        seed=stats_cfg.seed,
    )
    acc_ci_low, acc_ci_high = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        stat_fn=lambda yt, yp: accuracy_score(yt, yp),
        n_boot=stats_cfg.n_boot,
        alpha=stats_cfg.alpha,
        seed=stats_cfg.seed,
    )
    bal_acc_ci_low, bal_acc_ci_high = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        stat_fn=lambda yt, yp: balanced_accuracy_score(yt, yp),
        n_boot=stats_cfg.n_boot,
        alpha=stats_cfg.alpha,
        seed=stats_cfg.seed,
    )
    macro_f1_ci_low, macro_f1_ci_high = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        stat_fn=lambda yt, yp: f1_score(
            yt,
            yp,
            labels=labels_order,
            average="macro",
        ),
        n_boot=stats_cfg.n_boot,
        alpha=stats_cfg.alpha,
        seed=stats_cfg.seed,
    )

    metrics = {
        "labels": list(labels_order),
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "accuracy_ci_95": [acc_ci_low, acc_ci_high],
        "balanced_accuracy": balanced_acc,
        "balanced_accuracy_ci_95": [bal_acc_ci_low, bal_acc_ci_high],
        "macro_f1": macro_f1,
        "macro_f1_ci_95": [macro_f1_ci_low, macro_f1_ci_high],
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "cohen_kappa_ci_95": [kappa_ci_low, kappa_ci_high],
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class_dict,
        "support_per_class": support_per_class,
    }

    return metrics, cm_df
