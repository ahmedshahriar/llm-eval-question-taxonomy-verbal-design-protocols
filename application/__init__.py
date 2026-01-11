"""
Application layer: Use cases and workflow orchestration.

This layer coordinates between domain logic and infrastructure,
implementing the main workflows for inference and evaluation.

This module exposes high-level entry points for running LLM-based question labeling workflows.
"""

from application.batching import (
    detect_question_and_label_columns_from_test_data,
    group_test_dataframe,
    iter_segments,
)
from application.evaluation import log_evaluation_summary, run_evaluation_if_labels_available
from application.inference import call_llm_for_batch, run_inference
from application.prompting import build_examples_text, build_user_prompt
from application.serialize import attach_and_serialize_predictions

__all__ = [
    # Main workflows
    "run_inference",
    "call_llm_for_batch",
    "run_evaluation_if_labels_available",
    "log_evaluation_summary",
    # Data utilities
    "group_test_dataframe",
    "iter_segments",
    "detect_question_and_label_columns_from_test_data",
    "attach_and_serialize_predictions",
    # Prompting utilities
    "build_examples_text",
    "build_user_prompt",
]
