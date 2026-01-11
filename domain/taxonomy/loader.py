"""Parse taxonomy configuration from YAML dict."""

from typing import Any

from domain.taxonomy.normalizer import Taxonomy


def parse_taxonomy_config(data: dict[str, Any]) -> Taxonomy:
    """
    Parse pre-loaded YAML dict into Taxonomy object.

    This is a pure function - it does NOT perform file I/O.
    The YAML loading happens in infrastructure.config.loader.

    Args:
        data: Dictionary from yaml.safe_load()

    Returns:
        Taxonomy object with normalized aliases

    Raises:
        ValueError: If required keys are missing or have wrong types
    """
    canonical = data.get("canonical_subcategories", []) or []
    aliases_raw = data.get("aliases", {}) or {}
    ordering = data.get("ordering", {}) or {}

    if not isinstance(canonical, list):
        raise ValueError("canonical_subcategories must be a list")
    if not isinstance(aliases_raw, dict):
        raise ValueError("aliases must be a mapping")
    if not isinstance(ordering, dict):
        raise ValueError("ordering must be a mapping")

    aliases = {str(k).strip().lower(): str(v).strip() for k, v in aliases_raw.items()}
    return Taxonomy(
        canonical_subcategories=list(canonical),
        aliases=dict(aliases),
        ordering=dict(ordering),
    )
