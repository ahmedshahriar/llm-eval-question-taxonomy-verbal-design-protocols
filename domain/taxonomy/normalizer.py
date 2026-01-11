"""Taxonomy normalization utilities."""

import re
from functools import cached_property

from pydantic import BaseModel, Field


class Taxonomy(BaseModel):
    """Eris Question Asking Taxonomy configuration and normalization."""

    canonical_subcategories: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)  # lower(raw) -> canonical
    ordering: dict[str, list[str]] = Field(default_factory=dict)  # LLQ/DRQ/GDQ -> list of subcats

    @cached_property
    def _canonical_lookup(self) -> dict[str, str]:
        """Build lowercase lookup table for canonical subcategories."""
        return {str(c).strip().lower(): str(c).strip() for c in self.canonical_subcategories}

    def normalize_subcategory(self, raw: object) -> str:
        """
        Normalize subcategory labels to canonical form.

        Examples:
            >>> taxonomy = Taxonomy()
            >>> taxonomy.normalize_subcategory("Instrumental / Procedural")
            'Instrumental/Procedural'
            >>> taxonomy.normalize_subcategory("  rationale  ")
            'Rationale/Function/Goal Orientation'

        Args:
            raw: Raw subcategory value (can be None, str, or other types)

        Returns:
            Normalized canonical subcategory string, or empty string if invalid
        """
        if raw is None:
            return ""
        s = str(raw).strip()
        if not s:
            return ""
        s_norm = re.sub(r"\s+", " ", s)
        s_norm = re.sub(r"\s*/\s*", "/", s_norm)
        key = s_norm.lower()

        if key in self._canonical_lookup:
            return self._canonical_lookup[key]
        if key in self.aliases:
            return self.aliases[key]
        return s_norm
