# LLM Evaluation: Question Taxonomy Classification

This codebase provides a modular and configurable pipeline to evaluate large language models (LLMs) on the task of classifying questions from verbal design protocols according to the [Eris (2004) taxonomy](http://dx.doi.org/10.1007/978-1-4419-8943-7). The pipeline is designed for easy customization of prompts, models, and datasets, and includes integration with [**Opik**](https://github.com/comet-ml/opik) for experiment tracking.

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
│  ├─ config/                       # Config models and provider parameter registry
│  ├─ prompting/                    # Prompt manager (opik/offline)
│  ├─ io/                           # Dataset/artifact I/O
│  ├─ observability/                # Logging configuration
│  ├─ providers/                    # Provider adapters and factory
```

## Quick Start

### Prerequisites

- **Python ≥3.11**
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended) or use `pip` + `venv`.

### Installation

```bash
# Clone repository
git clone git@github.com:ahmedshahriar/llm-eval-question-taxonomy-verbal-design-protocols.git
cd llm-eval-question-taxonomy-verbal-design-protocols

# Install dependencies (requires Python >=3.11)
# includes dev group by default
uv sync
```

### Setup

1. Create a `.env` file. An example is provided in `.env.example`:

```bash
cp .env.example .env  # Must be at the repository root
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

After a successful run, artifacts are saved to `outputs/<run_id>/` (filenames may vary by configuration, but the structure is consistent):

```
outputs/<run_id>/
├─ config_snapshot.json              # Exact configuration used
├─ data_fingerprint.json             # Dataset metadata (rows, columns)
├─ prompts/
│  ├─ system.txt                     # System prompt used
│  ├─ user.txt                       # User prompt used
│  └─ prompt_metadata.json           # Prompt versioning info (Opik)
├─ batches/
│  ├─ batch_001_raw.json             # Raw API responses per batch
│  └─ ...
├─ predictions.json                  # All predictions with ground truth
├─ metrics.json                      # Evaluation metrics, token usage
├─ sub-category_alignment_table.csv  # Per-sub-category alignment breakdown
└─ run.log                           # Detailed execution log
```

## Usage

### Custom Prompts

Prompts are organized by provider and role. To customize:

1. Navigate to `prompts/<provider>/`
2. Edit system prompts in `system/<category|sub-category>/`
3. Edit user prompts in `user/label/<category|sub-category>/icl-demo/<none|category|sub-category>/`

Or specify custom prompt paths in `experiment.yaml`:

```yaml
system_prompt_path: prompts/openai/system/my-custom-system.txt
user_prompt_path: prompts/openai/user/my-custom-user.txt
```

### Provider Configuration

Edit `configs/providers/<provider>.yaml` to add models or adjust pricing, for example:

#### OpenAI with Prompt-Caching

```yaml
provider: openai
models:
  gpt-4.1-2025-04-14:
    params:
      service_tier: "default"
      temperature: 0.0
      prompt_cache_key: "<your-prompt-cache-key>"
      prompt_cache_retention: "1h"
    pricing:
      input_per_1m: 2.50
      cached_input_per_1m: 1.25
      output_per_1m: 10.00
```

> [!IMPORTANT]
> **Prompt-caching cost note:** OpenAI/Anthropic return cache-specific token counts (e.g., OpenAI `usage.prompt_tokens_details.cached_tokens`), but Opik’s cost tracking is an **estimate** and isn’t documented as cache-discount aware (and may be `None` for unsupported models). This repo computes costs manually using `configs/providers/<provider>.yaml` and logs to Opik.

#### Ollama (local)

To run locally with Ollama, add a provider config at `configs/providers/ollama.yaml` and set your run to use it.

1) Start Ollama and pull a model (example):
```bash
ollama serve
ollama pull qwen3:8b
```

2) Configure Ollama in `configs/providers/ollama.yaml` (example):
```yaml
provider: ollama
models:
  qwen3:8b:
    params:
      base_url: "http://localhost:11434"
      temperature: 0
      seed: 42
      num_ctx: 8192  # Context length. Ollama default: 4096; max: 32768
      think: false
      keep_alive: "10m"
    pricing: {}
```

3) Select it in `configs/experiment.yaml`:

```yaml
provider: ollama
model: qwen3:8b
```

### Experiment Tracking with Opik

The pipeline automatically logs:
- Prompts (with versioning)
- Token usage per batch
- Costs (input/output/total)
- Evaluation metrics

View traces at [app.comet.com/opik](https://www.comet.com/site/products/opik/)

To disable prompt registration in Opik (e.g., for local testing), set the following in `experiment.yaml`:

```yaml
# experiment.yaml
prompts_register_in_opik: false
```

### Batch Processing

Adjust batch size based on context window and cost considerations in `experiment.yaml`:

```yaml
batch_size: 50  # Process 50 questions per API call
# batch_size: null  # Process all questions in a single call
```

Smaller batches = more API calls but better error recovery.

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

# Implement the provider adapter
class MyProviderAdapter(ProviderAdapter):
   def call_batch(self, *args): ...

# Register the adapter
register_adapter(Provider.MY_PROVIDER, MyProviderAdapter)
```

2. Add config model in `infrastructure/config/models.py`
3. Create provider YAML in `configs/providers/my-provider.yaml`
4. Add prompts in `prompts/my-provider/`

#### Scaffold prompt folders for a new provider

Create a new provider prompt bundle by copying an existing provider:

```bash
# Create prompts/<provider-name>/ with the same structure as prompts/openai/
uv run python tools/scaffold_prompts.py --provider <provider-name> --from openai
```

Or create the required folder structure with stub files:

```bash
# Create empty prompts/<provider-name>/ structure
uv run python  tools/scaffold_prompts.py --provider <provider-name> --empty
```

### Pre-commit

Install hooks (one-time per clone):

```bash
uv sync
uv run pre-commit install
```

Run all hooks manually:

```bash
uv run pre-commit run --all-files
```

### Running Tests

```bash
# Install dependencies (includes dev group by default)
uv sync

# Run the test suite
pytest

# Run specific test
pytest tests/unit/test_anthropic_cache_math.py
```

### Code Quality

```bash
# Full quality gate (recommended)
uv run pre-commit run --all-files

# Or run tools individually:

# Lint (and auto-fix where possible)
ruff check . --fix

# Format
ruff format .

# Static type checking
ty check
```

## Reference
- Eris, Ö. (2004). *Effective Inquiry for Innovative Engineering Design*. Springer. [DOI](https://doi.org/10.1007/978-1-4419-8943-7)

## License
This repository is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
