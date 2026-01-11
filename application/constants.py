"""Application-level constants."""

from pathlib import Path

# Keys for serialization
ORIGINAL_INDEX_KEY = "og_index"
QUESTION_KEY = "question"
HUMAN_LABEL_KEY = "human_label"
HUMAN_SUBCATEGORY_KEY = "human_subcategory"
HUMAN_SUBCATEGORY_NORMALIZED_KEY = "human_subcategory_normalized"
LLM_SUBCATEGORY_NORMALIZED_KEY = "LLM_Subcategory_normalized"

# Column names for predictions
PRED_LABEL_COL = "LLM_Label"
PRED_SUBCATEGORY_COL = "LLM_Subcategory"
BATCH_OUTPUT_PAYLOAD = "batch_output_payload"

# Output filenames
PREDICTIONS_FILENAME = "test_predictions.json"
METRICS_FILENAME = "metrics.json"
SUBCAT_ALIGNMENT_FILENAME = "subcategory_alignment_table.csv"
CONFIG_SNAPSHOT_FILENAME = "config.resolved.json"
DATA_FINGERPRINT_FILENAME = "data_fingerprint.json"
PROMPT_METADATA_FILENAME = "prompt_metadata.json"

# Output directory structure
OUTPUT_ROOT = Path("outputs")
BATCH_OUTPUTS_DIRNAME = "batches"
PROMPT_OUTPUTS_DIRNAME = "prompts"
LOG_FILENAME = "run.log"
