"""
Infrastructure layer: External dependencies and I/O boundaries.

Contains adapters for:
- LLM providers (OpenAI, Anthropic, Mock)
- Configuration loading (YAML, environment)
- Prompt management (disk, Opik)
- Observability (logging, tracing)

This is the only layer that performs I/O operations.
"""

# Most commonly used - exposed at top level for convenience
from infrastructure.config import (
    Provider,
    RunConfig,
    StatsConfig,
    load_run_config,
)
from infrastructure.providers import ProviderAdapter, make_adapter

__all__ = [
    # Provider adapters (most commonly used)
    "make_adapter",
    "ProviderAdapter",
    # Configuration (most commonly used)
    "load_run_config",
    "RunConfig",
    "Provider",
    "StatsConfig",
]
