import logging

from infrastructure.config.models import Provider

from .base import ProviderAdapter

logger = logging.getLogger(__name__)

# Provider -> Adapter class
_ADAPTER_REGISTRY: dict[Provider, type[ProviderAdapter]] = {}


def register_adapter(provider: Provider, adapter_cls: type[ProviderAdapter], *, override: bool = False) -> None:
    """Register an adapter class for a provider.

    This is the plugin hook: provider modules call this at import time.
    """
    if (provider in _ADAPTER_REGISTRY) and not override:
        existing = _ADAPTER_REGISTRY[provider]
        raise RuntimeError(
            f"Adapter already registered for provider={provider.value}: {existing.__name__}. "
            f"Use override=True to replace."
        )
    _ADAPTER_REGISTRY[provider] = adapter_cls
    logger.debug("Registered adapter for provider=%s: %s", provider.value, adapter_cls.__name__)


def get_adapter_class(provider: Provider) -> type[ProviderAdapter] | None:
    """Return the registered adapter class (or None if not registered yet)."""
    return _ADAPTER_REGISTRY.get(provider)
