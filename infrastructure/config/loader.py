"""Configuration loading from YAML files."""

from pathlib import Path
from typing import Any

import yaml

from domain.taxonomy.loader import parse_taxonomy_config
from domain.taxonomy.normalizer import Taxonomy
from infrastructure.config.models import (
    DataColumnsConfig,
    Provider,
    ProviderConfig,
    ProviderModelConfig,
    RunConfig,
    StatsConfig,
)
from infrastructure.constants import DATA_DIR, PROMPTS_DIR, PROVIDERS_DIR, TAXONOMY_FILE

from .registry import PARAM_MODEL_BY_PROVIDER


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file and return as dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict in {path}, got {type(data)}")

    return data


def load_taxonomy_config(path: Path) -> Taxonomy:
    """
    Load taxonomy from YAML file.

    This function handles file I/O, then delegates parsing to domain layer.
    """
    data = _load_yaml(path)
    return parse_taxonomy_config(data)


def load_provider_config(providers_dir: Path, provider: Provider) -> ProviderConfig:
    """
    Load a provider YAML (e.g., configs/providers/openai.yaml) into a ProviderConfig.

    Args:
        providers_dir: Directory containing provider YAML files
        provider: Provider enum value

    Returns:
        ProviderConfig with models and pricing

    Raises:
        ValueError: If YAML is missing required keys or has invalid types
    """
    path = providers_dir / f"{provider.value}.yaml"
    data = _load_yaml(path)

    if not isinstance(data, dict):
        raise ValueError(f"Provider YAML must parse to a mapping/dict: {path}")

    if "provider" not in data:
        raise ValueError(f"Provider YAML missing required key 'provider': {path}")
    try:
        file_provider = Provider(data["provider"])
    except Exception as e:
        raise ValueError(f"Invalid provider value {data.get('provider')!r} in {path}") from e

    if file_provider is not provider:
        raise ValueError(f"Provider YAML mismatch: expected {provider.value}, got {file_provider.value} in {path}")

    models_raw = data.get("models") or {}
    if not isinstance(models_raw, dict) or not models_raw:
        raise ValueError(f"Provider YAML missing/invalid 'models' mapping: {path}")

    models: dict[str, ProviderModelConfig] = {
        str(model_name): ProviderModelConfig(
            params=dict((block or {}).get("params") or {}),
            pricing=dict((block or {}).get("pricing") or {}),
        )
        for model_name, block in models_raw.items()
    }

    return ProviderConfig(provider=file_provider, models=models)


def load_run_config(experiment_path: Path) -> RunConfig:
    """
    Load experiment.yaml and construct a fully-resolved RunConfig.

    Conventions (required for adding providers):
    - The Provider enum value must match the RunConfig field name used for provider-specific params.
      Example: if Provider.OPENAI.value == "openai", RunConfig must define an `openai` field
      (e.g., `openai: Optional[OpenAIConfig]`).
    - This naming convention allows provider parameter models to be bound dynamically from the registry.
    """
    exp = _load_yaml(experiment_path)

    if "provider" not in exp:
        raise ValueError("experiment.yaml missing required key: provider")
    if "model" not in exp:
        raise ValueError("experiment.yaml missing required key: model")
    if "test_file" not in exp or not exp.get("test_file"):
        raise ValueError("experiment.yaml missing required key: test_file")

    provider = Provider(str(exp["provider"]).strip().lower())
    model = str(exp["model"]).strip()

    data_dir = Path(exp.get("data_dir", str(DATA_DIR)))
    prompts_root = Path(exp.get("prompts_root", str(PROMPTS_DIR)))
    taxonomy_file = Path(exp.get("taxonomy_file", str(TAXONOMY_FILE)))
    providers_dir = Path(exp.get("providers_dir", str(PROVIDERS_DIR)))

    icl_demo_file = exp.get("icl_demo_file")
    test_file = exp["test_file"]

    include_icl_demo = bool(exp.get("include_icl_demo", True))
    include_sub_in_icl = bool(exp.get("include_subcategory_in_icl_demo", True))
    label_subcategories = bool(exp.get("label_subcategories", True))

    columns = DataColumnsConfig(
        icl_demo_question_col=exp.get("icl_demo_question_col"),
        icl_demo_category_col=exp.get("icl_demo_category_col"),
        icl_demo_subcategory_col=exp.get("icl_demo_subcategory_col"),
        test_question_col=exp["test_question_col"],
        test_category_col=exp.get("test_category_col"),
        test_subcategory_col=exp.get("test_subcategory_col"),
    )

    # preserve original behavior
    if not label_subcategories:
        columns.test_subcategory_col = None

    icl_demo_file_path: Path | None = (data_dir / icl_demo_file) if icl_demo_file else None
    test_file_path = data_dir / test_file

    taxonomy = load_taxonomy_config(taxonomy_file)

    prov_cfg = load_provider_config(providers_dir, provider)
    if model not in prov_cfg.models:
        raise KeyError(
            f"Model '{model}' not found in {providers_dir / (provider.value + '.yaml')}. "
            f"Available: {list(prov_cfg.models.keys())}"
        )

    provider_model: ProviderModelConfig = prov_cfg.models[model]
    params = provider_model.params or {}

    # Bind provider params using the registry
    param_model_cls = PARAM_MODEL_BY_PROVIDER.get(provider)
    if param_model_cls is None:
        raise ValueError(f"No param model registered for provider: {provider.value}")

    # This is how we stay provider-agnostic without hardcoded field-name mapping:
    # we rely on the convention: field name == provider.value
    if provider.value not in RunConfig.model_fields:
        raise ValueError(
            f"RunConfig has no field '{provider.value}'. "
            f"Add `'{provider.value}': Optional[<YourProviderConfig>] = None` to RunConfig "
            f"(field name must match Provider.value)."
        )

    run_kwargs = {provider.value: param_model_cls(**params)}

    stats = StatsConfig(**(exp.get("stats") or {}))

    system_prompt_path = exp.get("system_prompt_path")
    user_prompt_path = exp.get("user_prompt_path")

    cfg = RunConfig(
        provider=provider,
        model=model,
        batch_size=exp.get("batch_size"),
        prompts_root=prompts_root,
        prompts_register_in_opik=bool(exp.get("prompts_register_in_opik", True)),
        label_subcategories=label_subcategories,
        include_icl_demo=include_icl_demo,
        include_subcategory_in_icl_demo=include_sub_in_icl,
        group_col=exp.get("group_col"),
        columns=columns,
        stats=stats,
        taxonomy_file=taxonomy_file,
        taxonomy=taxonomy,
        providers_dir=providers_dir,
        provider_model=provider_model,
        icl_demo_file_path=icl_demo_file_path,
        test_file_path=test_file_path,
        system_prompt_path=Path(system_prompt_path) if system_prompt_path else None,
        user_prompt_path=Path(user_prompt_path) if user_prompt_path else None,
        **run_kwargs,
    )

    return cfg
