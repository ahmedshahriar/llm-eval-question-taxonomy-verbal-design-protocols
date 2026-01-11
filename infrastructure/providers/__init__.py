"""
LLM provider adapters.

Implements the adapter pattern for different LLM backends:
- OpenAI (GPT-4.1 with prompt caching)
- Anthropic (Claude Sonnet 4.5 with prompt caching)
- Mock (for testing)

All adapters implement the ProviderAdapter interface.
"""

from infrastructure.providers.anthropic import AnthropicAdapter
from infrastructure.providers.base import ProviderAdapter
from infrastructure.providers.factory import make_adapter
from infrastructure.providers.mock import MockAdapter
from infrastructure.providers.openai import OpenAIAdapter

__all__ = [
    # Abstract base
    "ProviderAdapter",
    # Concrete implementations
    "OpenAIAdapter",
    "AnthropicAdapter",
    "MockAdapter",
    # Factory (most commonly used)
    "make_adapter",
]
