# LLM Evaluation: Question Taxonomy Classification

This codebase provides a modular and configurable pipeline to evaluate large language models (LLMs) on the task of classifying questions from verbal design protocols according to the [Eris (2004) taxonomy](http://dx.doi.org/10.1007/978-1-4419-8943-7). The pipeline is designed for easy customization of prompts, models, and datasets, and includes integration with Opik for experiment tracking.

## Project Structure

Directory overview:

```
.
├─ main.py                          # CLI entrypoint. Loads config, runs application pipeline.
├─ pyproject.toml                   # Dependencies and project metadata
├─ configs/
│  ├─ experiment.yaml               # Primary run configuration (model, prompts, data, etc.)
│  ├─ taxonomy.yaml                 # Eris’ taxonomy definition (labels, hierarchy)
│  └─ providers/                    # Provider-specific model configs (params and pricing)
│     └─ <provider>.yaml            # One file per provider (e.g., openai/anthropic/bedrock/ollama...)
├─ prompts/                         # Prompt templates (by provider and role)
│  └─ <provider>/                   # One folder per provider (prompt variants)
│     ├─ system/
│     │  └─ *.txt                   # System prompt files
│     └─ user/
│        └─ *.txt                   # User prompt files
├─ dataset/                         # Datasets
│  └─ ...
├─ outputs/                         # Run outputs (created at runtime)
│  └─ <run_id>/
│     ├─ run.log
│     ...
├─ application/                     # Orchestration layer (batching, inference, evaluation, serialization)
├─ domain/                          # Schemas, taxonomy, evaluation logic
├─ infrastructure/                  # External integrations (I/O, providers, logging)
│  ├─ config/                       # Config models, Provider-param model registry
│  ├─ prompting/                    # Prompt manager (opik/offline)
│  ├─ io/                           # Dataset/artifact I/O
│  ├─ observability/                # Configure logging
│  ├─ providers/                    # Provider adapter implementations and factory
```

## Quick Start

### Prerequisites

- **Python ≥3.11**
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended) or use `pip`/`venv`.

### Installation

```bash
# Clone repository
git clone <repo-url>
cd llm-eval-question-taxonomy-verbal-design-protocols

# Install dependencies (requires Python >=3.11)
uv sync
```

### Setup

1. Create a `.env` file. An example is provided in `.env.example`:

```bash
cp .env.example .env # The file must be at your repository's root!
```

Edit `.env` and add your API keys:

```bash
# LLM Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
...
# Opik Configuration (for experiment tracking)
OPIK_API_KEY=...
OPIK_WORKSPACE=your-workspace
OPIK_PROJECT_NAME=your-project-name
```

2. Configure the experiment run in `configs/experiment.yaml`.
3. Modify prompt templates in `prompts/<provider>/system/` and `prompts/<provider>/user/` as needed.
4. Place your dataset files in the `dataset/` directory. A sample dataset is provided.

### Running the Pipeline

```bash
# Standard run with API calls
python main.py

# Mock run without API calls (for testing)
python main.py --mock

# Custom config file
python main.py --experiment configs/my-experiment.yaml

# Adjust logging verbosity
python main.py --console-level DEBUG --file-level DEBUG
```

### Understanding the Output

After a successful run, outputs are saved to `outputs/<run_id>/`:

```
outputs/<run_id>/
├─ <config_snapshot>.json              # Exact configuration used
├─ <data_fingerprint>.json             # Dataset metadata (rows, columns)
├─ prompts/
│  ├─ system.txt                       # System prompt used
│  ├─ user.txt                         # User prompt used
│  └─ <prompt_metadata>.json           # Prompt versioning info (Opik)
├─ batches/
│  ├─ batch_001_raw.json               # Raw API responses per batch
│  └─ ...
├─ <predictions>.json                  # All predictions with ground truth
├─ <metrics>.json                      # Evaluation metrics, token usage
├─ <subcategory_alignment>.csv         # Per-subcategory alignment breakdown
└─ run.log                             # Detailed execution log
```


## Development

### Project Architecture

The codebase follows a layered architecture:

- **`application/`**: High-level workflows (inference, evaluation, serialization)
- **`domain/`**: Core business logic (taxonomy, metrics, schemas)
- **`infrastructure/`**: External integrations (APIs, I/O, prompt library, observability)

### Adding a New Provider

1. Create adapter in `infrastructure/providers/<provider>.py`:

```python
from infrastructure.config.models import Provider
from infrastructure.providers.base import ProviderAdapter
from infrastructure.providers.registry import register_adapter

class MyProviderAdapter(ProviderAdapter):
   def call_batch(self, *args): ...

register_adapter(Provider.MY_PROVIDER, MyProviderAdapter)
```

2. Add config model in `infrastructure/config/models.py`
3. Create provider YAML in `configs/providers/my-provider.yaml`
4. Add prompts in `prompts/my-provider/`

## License
This repository is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
