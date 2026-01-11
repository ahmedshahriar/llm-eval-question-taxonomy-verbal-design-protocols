"""
Evaluation metrics and statistical analysis.

Provides:
- Classification metrics (accuracy, F1, kappa)
- Bootstrap confidence intervals
- Subcategory alignment tables

Most functions are pure (depend only on numpy, pandas, sklearn); *_and_save helpers write outputs to disk.
"""

from domain.evaluation.bootstrap import bootstrap_ci
from domain.evaluation.metrics import compute_classification_metrics
from domain.evaluation.tables import (
    compute_subcategory_alignment_table,
    compute_subcategory_alignment_table_and_save,
)

__all__ = [
    "compute_classification_metrics",
    "bootstrap_ci",
    "compute_subcategory_alignment_table",
    "compute_subcategory_alignment_table_and_save",
]
