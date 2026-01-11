"""LLM inference workflow."""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from opik import opik_context, track

from application.batching import group_test_dataframe, iter_segments
from application.constants import BATCH_OUTPUT_PAYLOAD
from application.prompting import build_user_prompt
from domain.schemas import BatchLabels
from infrastructure.config.models import RunConfig
from infrastructure.observability.logging import clear_batch_context, get_log_context, set_log_context
from infrastructure.prompting.manager import PromptObj
from infrastructure.providers.base import ProviderAdapter

logger = logging.getLogger(__name__)


@track(
    type="llm",
    metadata={"task": "question_labelling_batch"},
)
def call_llm_for_batch(
    cfg: RunConfig,
    adapter: ProviderAdapter,
    system_prompt: PromptObj,
    user_prompt: PromptObj,
    examples_text: str,
    questions: Sequence[str],
    batch_id: int,
) -> tuple[BatchLabels, dict[str, Any]]:
    """
    Call the configured provider adapter for a single batch.

    Returns:
        - Parsed BatchLabels (Pydantic)
        - Dict with token counts, cache metadata, costs, and output payload
    """
    user_prompt_str = build_user_prompt(user_prompt, questions, examples_text)
    system_prompt_str = system_prompt.prompt

    # Name the span (provider-specific usage/cost metadata is attached inside the adapter)
    opik_context.update_current_span(
        name=f"labelling.batch-{batch_id:03d}",
    )

    # Call the provider adapter
    parsed, result = adapter.call_batch(
        system_text=system_prompt_str,
        user_text=user_prompt_str,
        response_model=BatchLabels,
        batch_id=batch_id,
        questions=questions,
        extra_trace_meta=get_log_context(),
    )

    return parsed, result


@track(type="general")
def run_inference(
    cfg: RunConfig,
    adapter: ProviderAdapter,
    system_prompt: PromptObj,
    user_prompt: PromptObj,
    examples_text: str,
    test_df: pd.DataFrame,
    question_col: str,
    output_batch_dir: Path,
) -> tuple[dict[int, str], dict[int, str | None], dict[str, int | float]]:
    """Run the LLM labelling over all questions and return predictions, subcategories, and usage stats."""

    run_metadata = {
        "provider": cfg.provider,
        "model": cfg.model,
        "batch_size": cfg.batch_size,
        "group_by": cfg.group_col,
        "label_subcategories": cfg.label_subcategories,
        "include_subcategory_in_icl_demo": cfg.include_subcategory_in_icl_demo,
        "user_prompt_name": user_prompt.name,
    }
    opik_context.update_current_span(
        name=f"Labelling.inference.{cfg.provider.value}_{cfg.model}", metadata=run_metadata
    )

    pred_map: dict[int, str] = {}
    subcat_map: dict[int, str | None] = {}

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_input_cost = 0.0
    total_output_cost = 0.0
    total_cost = 0.0

    groups = group_test_dataframe(test_df, cfg.group_col)
    logger.info("Total test questions: %d", len(test_df))

    batch_id = 0
    for team_value, group_df in groups:
        group_questions = group_df[question_col].fillna("").astype(str).tolist()
        group_indices = group_df.index.tolist()

        segments = iter_segments(len(group_questions), cfg.batch_size)

        for seg_start, seg_end in segments:
            batch_id += 1

            set_log_context(batch_id=batch_id)

            seg_questions = group_questions[seg_start:seg_end]
            seg_indices = group_indices[seg_start:seg_end]

            logger.info(
                "Processing batch %d (Team %s, items %d to %d of %d)...",
                batch_id,
                team_value,
                seg_start,
                seg_end - 1,
                len(group_questions),
            )

            # Call LLM with pricing info
            batch_labels, batch_result = call_llm_for_batch(
                cfg=cfg,
                adapter=adapter,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                examples_text=examples_text,
                questions=seg_questions,
                batch_id=batch_id,
            )

            # Extract results
            it = batch_result["input_tokens"]
            ot = batch_result["output_tokens"]
            tt = batch_result["total_tokens"]

            batch_input_cost = batch_result["batch_input_cost"]
            batch_output_cost = batch_result["batch_output_cost"]
            batch_total_cost = batch_result["batch_total_cost"]

            logger.info(
                "Batch %d: provider=%s, model=%s, input_tokens=%d, output_tokens=%d, total_tokens=%d",
                batch_id,
                cfg.provider.value,
                cfg.model,
                it,
                ot,
                tt,
            )

            logger.info(
                "Batch %d cost: input=$%.6f, output=$%.6f, total=$%.6f",
                batch_id,
                batch_input_cost,
                batch_output_cost,
                batch_total_cost,
            )

            # Save raw output for this batch
            batch_output_path = output_batch_dir / f"batch_{batch_id:03d}_raw.json"
            with batch_output_path.open("w", encoding="utf-8") as f:
                json.dump(batch_result[BATCH_OUTPUT_PAYLOAD], f, ensure_ascii=False, indent=2)

            # Sanity-check sizes
            if len(batch_labels.items) != len(seg_questions):
                raise RuntimeError(
                    f"Batch {batch_id}: expected {len(seg_questions)} outputs, got {len(batch_labels.items)}"
                )

            # Map prompt indices (1..len(seg_indices)) to original DataFrame indices
            index_to_df_index: dict[int, int] = {
                prompt_idx: df_idx
                for prompt_idx, df_idx in zip(
                    range(1, len(seg_indices) + 1),
                    seg_indices,
                    strict=False,
                )
            }

            # Assign predictions to global index, with index & text cross-checks
            for item in batch_labels.items:
                if item.index not in index_to_df_index:
                    raise RuntimeError(f"Batch {batch_id}: model returned invalid index {item.index}")
                df_idx = index_to_df_index[item.index]

                raw_orig = test_df.loc[df_idx, question_col]
                original_q = ("" if pd.isna(raw_orig) else str(raw_orig)).strip()
                if item.question.strip() != original_q:
                    logger.warning(
                        "Batch %d: question text mismatch at index %d.\nOriginal: %r\nModel:    %r",
                        batch_id,
                        item.index,
                        original_q,
                        item.question,
                    )

                # Top-level category
                pred_map[df_idx] = item.label

                # Optional subcategory
                sub_val = getattr(item, "subcategory", None)
                if sub_val is not None:
                    subcat_map[df_idx] = cfg.taxonomy.normalize_subcategory(sub_val)

            # Accumulate totals
            total_input_tokens += it
            total_output_tokens += ot
            total_tokens += tt
            total_input_cost += batch_input_cost
            total_output_cost += batch_output_cost
            total_cost += batch_total_cost

    logger.info("All batches processed.")
    logger.info(
        "Total tokens: input=%d, output=%d, total=%d",
        total_input_tokens,
        total_output_tokens,
        total_tokens,
    )
    logger.info(
        "Total cost: input=$%.6f, output=$%.6f, total=$%.6f",
        total_input_cost,
        total_output_cost,
        total_cost,
    )

    # Update parent span with aggregated totals
    opik_context.update_current_span(
        usage={
            "prompt_tokens": total_input_tokens,
            "completion_tokens": total_output_tokens,
            "total_tokens": total_tokens,
        },
        total_cost=float(total_cost),
        metadata={
            "total_batches": batch_id,
            "total_questions": len(test_df),
            "total_input_cost_usd": round(total_input_cost, 6),
            "total_output_cost_usd": round(total_output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_per_question": round(total_cost / len(test_df), 6) if len(test_df) > 0 else 0.0,
        },
    )

    usage_stats = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_input_cost_usd": total_input_cost,
        "total_output_cost_usd": total_output_cost,
        "total_cost_usd": total_cost,
    }

    clear_batch_context()
    # NOTE: subcat_map may be empty if prompts do not ask for subcategories.
    return pred_map, subcat_map, usage_stats
