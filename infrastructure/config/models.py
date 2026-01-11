"""Configuration models (Pydantic classes)."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from domain.taxonomy.normalizer import Taxonomy
from infrastructure.constants import PROVIDERS_DIR, TAXONOMY_FILE


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class DataColumnsConfig(BaseModel):
    """Column name mapping for ICL demo and test datasets."""

    # ICL demo columns (conditionally required)
    icl_demo_question_col: str | None = None
    icl_demo_category_col: str | None = None
    icl_demo_subcategory_col: str | None = None

    # Test columns (at least question is always required)
    test_question_col: str
    test_category_col: str | None = None
    test_subcategory_col: str | None = None


class OpenAIConfig(BaseModel):
    """OpenAI-specific configuration."""

    service_tier: str | None = None
    temperature: int | float | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None


class AnthropicConfig(BaseModel):
    """Anthropic-specific configuration."""

    service_tier: str | None = None
    temperature: int | float | None = None
    max_tokens: int
    betas: list[str]
    cache_ttl: str | None = None


class StatsConfig(BaseModel):
    """
    Configuration for evaluation statistics.

    Defaults match the original script behavior.
    """

    seed: int = 42
    n_boot: int = 2000
    alpha: float = 0.05
    labels_order: list[str] = Field(default_factory=lambda: ["LLQ", "DRQ", "GDQ"])


class ProviderModelConfig(BaseModel):
    """Per-model configuration and pricing."""

    params: dict[str, Any] = Field(default_factory=dict)
    pricing: dict[str, Any] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    """Provider-level configuration."""

    provider: Provider
    models: dict[str, ProviderModelConfig]


class ModelPricingOpenAI(BaseModel):
    """OpenAI pricing structure."""

    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float

    @property
    def input_cost_per_token(self) -> float:
        return self.input_per_1m / 1_000_000

    @property
    def cached_input_cost_per_token(self) -> float:
        return self.cached_input_per_1m / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        return self.output_per_1m / 1_000_000


class ModelPricingAnthropic(BaseModel):
    """Anthropic pricing structure with prompt caching tiers."""

    input_per_1m: float
    output_per_1m: float
    cache_write_5m_per_1m: float
    cache_write_1h_per_1m: float
    cache_read_per_1m: float

    @property
    def input_cost_per_token(self) -> float:
        return self.input_per_1m / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        return self.output_per_1m / 1_000_000

    @property
    def cache_write_5m_cost_per_token(self) -> float:
        return self.cache_write_5m_per_1m / 1_000_000

    @property
    def cache_write_1h_cost_per_token(self) -> float:
        return self.cache_write_1h_per_1m / 1_000_000

    @property
    def cache_read_cost_per_token(self) -> float:
        return self.cache_read_per_1m / 1_000_000


class RunConfig(BaseModel):
    """
    Runtime configuration.
    - Loaded from experiment.yaml
    - Validated and enriched by configuration loader
    - Consumed by provider adapters and experiment runner
    - Contains both user-specified and resolved fields
    """

    provider: Provider = Field(default=Provider.OPENAI, description="LLM provider backend to use.")
    model: str = Field(..., description="Model identifier for the selected provider.")
    batch_size: int | None = None

    # Prompt handling
    prompts_root: Path = Field(
        default_factory=lambda: Path("prompts"),
        description="Root directory containing provider-organized prompt templates.",
    )
    prompts_register_in_opik: bool = Field(
        default=True,
        description="Register prompts in Opik library. If False, load from disk only.",
    )

    # Prompt selection flags
    label_subcategories: bool = Field(
        default=True,
        description="If true, prompt the LLM to output subcategories in addition to top-level labels.",
    )
    include_icl_demo: bool = Field(
        default=True,
        description="If true, include ICL demonstrations and select an icl-demo prompt template.",
    )
    include_subcategory_in_icl_demo: bool = Field(
        default=True,
        description="If true, include subcategory lines inside ICL demonstrations. "
        "Ignored when include_icl_demo=False.",
    )

    # Columns
    columns: DataColumnsConfig

    # Provider config (resolved by loader)
    openai: OpenAIConfig | None = None
    anthropic: AnthropicConfig | None = None
    # FUTURE: Add Google provider support (requires GoogleConfig model + adapter implementation)

    # Stats
    stats: StatsConfig = Field(default_factory=StatsConfig)

    # Taxonomy (resolved by loader)
    taxonomy_file: Path = Field(default_factory=lambda: TAXONOMY_FILE)
    taxonomy: Taxonomy = Field(default_factory=Taxonomy)

    # Provider models directory + selected model block (resolved by loader)
    providers_dir: Path = Field(default_factory=lambda: PROVIDERS_DIR)
    provider_model: ProviderModelConfig = Field(default_factory=ProviderModelConfig)

    icl_demo_file_path: Path | None = Field(
        default=None,
        description="Path to the ICL demonstration examples dataset file (Excel or CSV).",
    )
    test_file_path: Path = Field(..., description="Path to the test dataset file (Excel or CSV).")

    group_col: str | None = Field(
        default=None,
        description="Optional grouping column (e.g., Team). If missing/None, no grouping is applied.",
    )

    # Optional prompt overrides (resolved by loader)
    system_prompt_path: Path | None = None
    user_prompt_path: Path | None = None

    @model_validator(mode="after")
    def _validate(self) -> "RunConfig":
        # Preserve original normalization behavior
        if self.include_subcategory_in_icl_demo and not self.include_icl_demo:
            self.include_subcategory_in_icl_demo = False

        if self.group_col is not None and not str(self.group_col).strip():
            self.group_col = None

        if not self.columns.test_question_col:
            raise ValueError("columns.test_question_col is required in experiment.yaml")

        if self.include_icl_demo:
            if not self.icl_demo_file_path:
                raise ValueError("icl_demo_file must be set when include_icl_demo=True")
            if not self.columns.icl_demo_question_col or not self.columns.icl_demo_category_col:
                raise ValueError("ICL demo columns are required when include_icl_demo=True")
            if self.include_subcategory_in_icl_demo and not self.columns.icl_demo_subcategory_col:
                raise ValueError(
                    "columns.icl_demo_subcategory_col is required when include_subcategory_in_icl_demo=True"
                )

        if self.label_subcategories and not self.columns.test_subcategory_col:
            raise ValueError("columns.test_subcategory_col is required when label_subcategories=True")

        return self
