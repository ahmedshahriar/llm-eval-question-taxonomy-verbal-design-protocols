"""
Taxonomy management: normalization, validation, and configuration.

This module handles Eris Question Asking Taxonomy operations.
All functions in this module are pure (no file I/O).
"""

from domain.taxonomy.loader import parse_taxonomy_config
from domain.taxonomy.normalizer import Taxonomy

__all__ = [
    "Taxonomy",
    "parse_taxonomy_config",
]
