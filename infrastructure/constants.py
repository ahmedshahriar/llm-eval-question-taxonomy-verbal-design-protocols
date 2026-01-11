from pathlib import Path

# Repo-root conventional directories/files (overrideable via experiment.yaml)
CONFIG_DIR = Path("configs")
PROVIDERS_DIR = CONFIG_DIR / "providers"
EXPERIMENT_FILE = CONFIG_DIR / "experiment.yaml"
TAXONOMY_FILE = CONFIG_DIR / "taxonomy.yaml"

PROMPTS_DIR = Path("prompts")
DATA_DIR = Path("dataset")
