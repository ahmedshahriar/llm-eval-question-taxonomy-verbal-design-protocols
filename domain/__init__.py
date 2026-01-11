"""
Domain layer: Business logic with minimal external dependencies.

Contains:
- schemas: Pydantic models for questions and batches
- taxonomy: Taxonomy normalization and configuration
- evaluation: Metrics computation and statistical analysis
"""

from domain.schemas import BatchLabels, QuestionLabel

__all__ = [
    "QuestionLabel",
    "BatchLabels",
]
