"""
Configuration management: models, loading, and validation.

Handles:
- RunConfig: Main experiment configuration
- Provider configs: OpenAI, Anthropic settings
- Taxonomy loading from YAML
- Environment variable overrides

The loader module performs file I/O; models are pure Pydantic classes.
"""

from infrastructure.config.loader import (
    load_provider_config,
    load_run_config,
    load_taxonomy_config,
)
from infrastructure.config.models import (
    AnthropicConfig,
    # Column mapping
    DataColumnsConfig,
    ModelPricingAnthropic,
    # Pricing models
    ModelPricingOpenAI,
    # Provider configs
    OpenAIConfig,
    # Enums
    Provider,
    ProviderConfig,
    ProviderModelConfig,
    # Main config
    RunConfig,
    # Stats config
    StatsConfig,
)

__all__ = [
    # Main config (most commonly used)
    "RunConfig",
    "load_run_config",
    # Enums
    "Provider",
    # Data columns
    "DataColumnsConfig",
    # Provider configs
    "OpenAIConfig",
    "AnthropicConfig",
    "ProviderModelConfig",
    "ProviderConfig",
    # Pricing
    "ModelPricingOpenAI",
    "ModelPricingAnthropic",
    # Stats
    "StatsConfig",
    # Loaders
    "load_provider_config",
    "load_taxonomy_config",
]
