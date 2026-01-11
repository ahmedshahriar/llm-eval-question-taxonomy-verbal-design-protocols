"""Factory for creating provider adapters."""

import importlib
import logging

from infrastructure.config.models import Provider, RunConfig

from .base import ProviderAdapter
from .mock import MockAdapter
from .registry import get_adapter_class

logger = logging.getLogger(__name__)


def _ensure_provider_imported(provider: Provider) -> None:
    """
    Lazy-import the provider module to trigger `register_adapter(...)`.

    Convention:
      - Provider enum value MUST match module filename under infrastructure/providers/
        e.g., Provider.OPENAI.value == "openai" -> infrastructure/providers/openai.py
    """
    module_name = f"{__package__}.{provider.value}"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if getattr(e, "name", None) == module_name:
            raise RuntimeError(
                f"No provider module found for provider='{provider.value}'. "
                f"Expected file: infrastructure/providers/{provider.value}.py"
            ) from e
        raise


def make_adapter(
    cfg: RunConfig,
    *,
    use_mock: bool = False,
    mock_fixtures: dict[int, dict] | None = None,
) -> ProviderAdapter:
    """
    Factory function to create the appropriate provider adapter.
    Args:
        cfg: Run configuration containing provider settings
        use_mock: If True, use the MockAdapter regardless of cfg
        mock_fixtures: Optional fixtures for the MockAdapter
    Returns:
        An instance of ProviderAdapter for the specified provider.
    Raises:
        RuntimeError: If the provider is unsupported.
    """
    if use_mock:
        return MockAdapter(cfg=cfg, fixtures=mock_fixtures)

    # 1) Try registry first (maybe already imported elsewhere)
    adapter_cls = get_adapter_class(cfg.provider)

    # 2) If not registered yet, import the provider module by convention, then retry
    if adapter_cls is None:
        _ensure_provider_imported(cfg.provider)
        adapter_cls = get_adapter_class(cfg.provider)

    if adapter_cls is None:
        raise RuntimeError(
            f"Provider '{cfg.provider.value}' did not register an adapter. "
            f"Make sure {cfg.provider.value}.py calls register_adapter(...)."
        )

    # Standard constructor path
    return adapter_cls.from_cfg(cfg)  # type: ignore[attr-defined]
