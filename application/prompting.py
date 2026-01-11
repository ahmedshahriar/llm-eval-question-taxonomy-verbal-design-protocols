"""Prompt construction utilities (pure functions)."""

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from infrastructure.config.models import RunConfig
from infrastructure.prompting.manager import PromptObj

logger = logging.getLogger(__name__)


def build_examples_text(icl_demo_sample: pd.DataFrame, cfg: RunConfig) -> str:
    """
    Build the few-shot ICL demonstrations block from a labeled ICL demo sample.

    This is a pure function - it formats data but doesn't load files.
      - cfg.columns.icl_demo_question_col must match the question column in the ICL demo file.
      - cfg.columns.icl_demo_category_col must match the top-level label column.
      - cfg.columns.icl_demo_subcategory_col (optional) is used only if cfg.include_subcategory_in_icl_demo=True.

    Args:
        icl_demo_sample: DataFrame containing labeled ICL examples
        cfg: RunConfig instance

    Returns:
        Formatted string of ICL examples

    Raises:
        ValueError: If required columns are not configured
        KeyError: If configured columns not found in DataFrame
    """
    # Shuffle icl demo sample deterministically
    icl_demo_sample = icl_demo_sample.sample(
        frac=1,  # Sample 100% of the data
        random_state=cfg.stats.seed,
    ).reset_index(drop=True)

    logger.debug("Building ICL examples: n_rows=%d, seed=%s", len(icl_demo_sample), cfg.stats.seed)

    q_col = cfg.columns.icl_demo_question_col
    cat_col = cfg.columns.icl_demo_category_col
    sub_col = cfg.columns.icl_demo_subcategory_col

    # Question column from config
    if not q_col:
        logger.error("Missing icl_demo_question_col in config")
        raise ValueError("cfg.columns.icl_demo_question_col is required to build ICL examples.")
    if q_col not in icl_demo_sample.columns:
        raise KeyError(f"Configured icl_demo_question_col='{q_col}' not found in ICL demo dataset columns.")

    # Category column from config
    if not cat_col:
        raise ValueError("cfg.columns.icl_demo_category_col is required to build ICL examples.")
    if cat_col not in icl_demo_sample.columns:
        raise KeyError(f"Configured icl_demo_category_col='{cat_col}' not found in ICL demo dataset columns.")

    use_sub = bool(cfg.include_subcategory_in_icl_demo)
    sub_col_name: str = ""

    if use_sub:
        if not sub_col:
            raise ValueError(
                "cfg.columns.icl_demo_subcategory_col is required when include_subcategory_in_icl_demo=True."
            )
        if sub_col not in icl_demo_sample.columns:
            raise KeyError(f"Configured icl_demo_subcategory_col='{sub_col}' not found in ICL demo dataset columns.")
        sub_col_name: str = sub_col

    example_lines: list[str] = []
    for _, row in icl_demo_sample.iterrows():
        q = str(row[q_col]).strip()
        label = str(row[cat_col]).strip()
        if use_sub:
            subcat = str(row[sub_col_name]).strip()
            example_lines.append(f"Question: {q}\nLabel: {label}\nSubcategory: {subcat}")
        else:
            example_lines.append(f"Question: {q}\nLabel: {label}")

    logger.debug("Built %d ICL examples", len(example_lines))
    return "\n\n".join(example_lines)


def build_user_prompt(user_prompt: PromptObj, questions: Sequence[str], examples_text: str) -> str:
    """
    Build user prompt by formatting template with questions and examples.

    Args:
        user_prompt: PromptObj object containing the user prompt template
        questions: Sequence of question strings
        examples_text: Formatted ICL examples text

    Returns:
        Formatted user prompt string

    Raises:
        ValueError: If prompt template missing required placeholders
    """
    # Build questions_block
    questions_block = "\n".join(f"{i + 1}. {q.strip()}" for i, q in enumerate(questions))
    logger.debug("Building user prompt template: %s", questions_block)

    try:
        rendered = user_prompt.format(questions_block=questions_block, examples_text=examples_text)
    except KeyError as e:
        raise ValueError(
            f"Prompt template missing required placeholder: {e}. "
            f"Template must contain {{{{questions_block}}}} and {{{{examples_text}}}}"
        ) from e

    if isinstance(rendered, str):
        return rendered

    # Handle common structured formats (e.g., OpenAI Chat messages)
    if isinstance(rendered, list):
        parts: list[str] = []
        for msg in rendered:
            if not isinstance(msg, dict):
                continue
            content: Any = msg.get("content")

            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append(block["text"])

        return "\n".join(p for p in parts if p).strip()

    raise TypeError(f"Prompt.format() returned unsupported type: {type(rendered)}")
