"""
CLI entrypoint for the question-labelling pipeline.

This script performs the following steps:
- loads .env, configs/experiment.yaml
- creates a per-run output folder under outputs/
- loads prompts (provider-aware), optional ICL examples
- runs batched inference via the selected provider adapter
- serializes predictions, runs evaluation if human labels are available
- computes subcategory alignment table (if applicable)
- saves metrics to JSON
- logs a human-readable summary of results
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import opik
from dotenv import load_dotenv
from opik import opik_context, track

from application import (
    build_examples_text,
    log_evaluation_summary,
    run_evaluation_if_labels_available,
)
from application.batching import detect_question_and_label_columns_from_test_data
from application.constants import (
    BATCH_OUTPUTS_DIRNAME,
    CONFIG_SNAPSHOT_FILENAME,
    DATA_FINGERPRINT_FILENAME,
    LOG_FILENAME,
    METRICS_FILENAME,
    OUTPUT_ROOT,
    PRED_LABEL_COL,
    PRED_SUBCATEGORY_COL,
    PREDICTIONS_FILENAME,
    PROMPT_METADATA_FILENAME,
    PROMPT_OUTPUTS_DIRNAME,
    SUBCAT_ALIGNMENT_FILENAME,
)
from application.inference import run_inference
from application.serialize import attach_and_serialize_predictions
from domain.evaluation import compute_subcategory_alignment_table_and_save
from infrastructure.config import load_run_config
from infrastructure.constants import EXPERIMENT_FILE
from infrastructure.io import ensure_exists, read_table
from infrastructure.observability import configure_logging, make_run_tag, set_log_context
from infrastructure.prompting.manager import PromptManager, PromptRole
from infrastructure.providers import make_adapter
from infrastructure.utils import set_seed

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LLM question labelling pipeline")
    p.add_argument(
        "--experiment",
        type=str,
        default=str(EXPERIMENT_FILE),
        help="Path to experiment.yaml (default: configs/experiment.yaml)",
    )
    p.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use Mock adapter instead of calling a real provider.",
    )
    p.add_argument(
        "--console-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console log level",
    )
    p.add_argument(
        "--file-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="File log level",
    )
    return p.parse_args()


@track(
    name="Question.labelling",
    type="general",
    metadata={"task": "question_labelling"},
    capture_input=False,
    capture_output=False,
    flush=True,
)
def main() -> None:
    args = _parse_args()

    env_file = Path(args.env)
    ensure_exists(env_file, "environment variables file")
    load_dotenv(env_file, override=True)

    experiment_path = Path(args.experiment)
    ensure_exists(experiment_path, "experiment.yaml")

    cfg = load_run_config(experiment_path)
    set_seed(cfg.stats.seed)

    opik.configure()

    # ---- Per-run output folder ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bs = "all" if cfg.batch_size is None else str(cfg.batch_size)

    run_id = (
        f"{ts}_"
        f"{cfg.provider.value if not args.mock else 'mock_provider'}_"
        f"{cfg.model if not args.mock else 'mock_model'}_"
        f"bs{bs}_"
        f"subcat{int(cfg.label_subcategories)}_"
        f"icl{int(cfg.include_icl_demo)}_"
        f"iclsub{int(cfg.include_subcategory_in_icl_demo)}"
    )

    # Create output directories
    run_dir = OUTPUT_ROOT / run_id
    prompts_snapshot_dir = run_dir / PROMPT_OUTPUTS_DIRNAME
    batch_dir = run_dir / BATCH_OUTPUTS_DIRNAME

    run_dir.mkdir(parents=True, exist_ok=True)
    prompts_snapshot_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / LOG_FILENAME
    configure_logging(
        log_file=log_path,
        console_level=getattr(logging, args.console_level),
        file_level=getattr(logging, args.file_level),
    )

    set_log_context(
        run_id_full=run_id,
        provider=str(cfg.provider.value),
        model=str(cfg.model),
    )

    logger.info("Starting run: run_id=%s (run_tag=%s)", run_id, make_run_tag(run_id))
    logger.info("Run output directory: %s", run_dir)

    # Load test data
    logger.info("Loading test data from %s...", cfg.test_file_path)
    test_df = read_table(cfg.test_file_path)
    logger.info("Test data loaded: %d rows, %d columns", test_df.shape[0], test_df.shape[1])

    # Resolve columns from cfg + test_df
    question_col, label_col = detect_question_and_label_columns_from_test_data(cfg=cfg, df=test_df)

    # Build ICL examples (optional)
    train_df = None
    if cfg.include_icl_demo:
        if cfg.icl_demo_file_path is None:
            raise ValueError("include_icl_demo=True but icl_demo_file_path is None")
        logger.info("Loading ICL demo data from %s...", cfg.icl_demo_file_path)
        train_df = read_table(cfg.icl_demo_file_path)
        logger.info("ICL demo data loaded: %d rows, %d columns", train_df.shape[0], train_df.shape[1])
        examples_text = build_examples_text(train_df, cfg)
    else:
        examples_text = ""
        logger.info("ICL demos disabled (include_icl_demo=False).")

    # Load prompts
    pm = PromptManager(prompts_root=cfg.prompts_root)
    system_prompt = pm.get_prompt(
        provider=cfg.provider,
        role=PromptRole.SYSTEM,
        cfg=cfg,
        override_path=cfg.system_prompt_path,
    )
    user_prompt = pm.get_prompt(
        provider=cfg.provider,
        role=PromptRole.USER,
        cfg=cfg,
        override_path=cfg.user_prompt_path,
    )

    # Save snapshot config + prompts + fingerprints
    (run_dir / CONFIG_SNAPSHOT_FILENAME).write_text(
        json.dumps(cfg.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    # Save snapshot of the prompts used
    (prompts_snapshot_dir / "system.txt").write_text(system_prompt.prompt, encoding="utf-8")
    (prompts_snapshot_dir / "user.txt").write_text(user_prompt.prompt, encoding="utf-8")
    (prompts_snapshot_dir / PROMPT_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "system_prompt": {
                    "name": system_prompt.name,
                    "commit": getattr(system_prompt, "commit", None),
                    "metadata": getattr(system_prompt, "metadata", None),
                },
                "user_prompt": {
                    "name": user_prompt.name,
                    "commit": getattr(user_prompt, "commit", None),
                    "metadata": getattr(user_prompt, "metadata", None),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # Save data fingerprint
    (run_dir / DATA_FINGERPRINT_FILENAME).write_text(
        json.dumps(
            {
                "icl_demo_file": str(cfg.icl_demo_file_path) if cfg.include_icl_demo else None,
                "test_file": str(cfg.test_file_path),
                "icl_rows": int(train_df.shape[0]) if cfg.include_icl_demo else None,  # ty: ignore
                "test_rows": int(test_df.shape[0]),
                "icl_columns": list(train_df.columns) if cfg.include_icl_demo else None,  # ty: ignore
                "test_columns": list(test_df.columns),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # update current trace with prompt info
    opik_context.update_current_span(
        name=f"Labelling.run.setup.{cfg.provider if not args.mock else 'mock_default_provider'}_{cfg.model}",
        metadata={
            "provider": cfg.provider,
            "model": cfg.model,
            "run_id": run_id,
        },
        prompts=[system_prompt, user_prompt],  # ty: ignore
    )

    # Init provider adapter
    logger.info("Initializing provider (provider=%s, model=%s)...", cfg.provider.value, cfg.model)
    adapter = make_adapter(cfg, use_mock=bool(args.mock))

    # Inference
    pred_map, subcat_map, usage_stats = run_inference(
        cfg=cfg,
        adapter=adapter,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        examples_text=examples_text,
        test_df=test_df,
        question_col=question_col,
        output_batch_dir=batch_dir,
    )

    # Attach predictions + save JSON
    test_df_out, predictions_path = attach_and_serialize_predictions(
        cfg=cfg,
        test_df=test_df,
        question_col=question_col,
        label_col=label_col,
        pred_map=pred_map,
        subcat_map=subcat_map if cfg.label_subcategories else None,
        predictions_path=run_dir / PREDICTIONS_FILENAME,
    )

    # Evaluation (only if human labels exist)
    metrics, cm_df, sub_cm_df = run_evaluation_if_labels_available(
        cfg=cfg,
        test_df_out=test_df_out,
        label_col=label_col,
    )

    # Merge usage stats into metrics and save
    metrics.update(usage_stats)
    metrics_path = run_dir / METRICS_FILENAME
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Saved metrics to %s", metrics_path)

    # Subcategory alignment table (if available)
    human_subcat_col = (
        cfg.columns.test_subcategory_col
        if (cfg.columns.test_subcategory_col and cfg.columns.test_subcategory_col in test_df_out.columns)
        else None
    )

    if label_col is not None and human_subcat_col is not None and PRED_SUBCATEGORY_COL in test_df_out.columns:
        subcat_table_path = compute_subcategory_alignment_table_and_save(
            cfg=cfg,
            df=test_df_out,
            output_dir=run_dir,
            human_top_label_col=label_col,
            human_subcat_col=human_subcat_col,
            llm_top_label_col=PRED_LABEL_COL,
            llm_subcat_col=PRED_SUBCATEGORY_COL,
            filename=SUBCAT_ALIGNMENT_FILENAME,
        )
        logger.info("Saved subcategory alignment table to %s", subcat_table_path)

    # Human-readable summary
    log_evaluation_summary(
        metrics=metrics,
        cm_df=cm_df,
        sub_cm_df=sub_cm_df,
        label_col=label_col,
        usage_stats=usage_stats,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
    )

    logger.info("Detailed log: %s", log_path)


if __name__ == "__main__":
    main()
