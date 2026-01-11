"""Pydantic models for structured LLM outputs."""

from typing import Literal

from pydantic import BaseModel, Field

Subcategory = Literal[
    # LLQ
    "Verification",
    "Disjunctive",
    "Definition",
    "Example",
    "Feature Specification",
    "Concept Completion",
    "Quantification",
    "Comparison",
    "Judgmental",
    # DRQ
    "Interpretation",
    "Rationale/Function/Goal Orientation",
    "Causal Antecedent",
    "Causal Consequent",
    "Expectational",
    "Instrumental/Procedural",
    "Enablement",  # both in DRQ and GDQ
    # GDQ
    "Proposal/Negotiation",
    "Method Generation",
    "Scenario Creation",
    "Ideation",
]


class QuestionLabel(BaseModel):
    """Single labelled question returned by the model."""

    index: int = Field(
        ...,
        description="The integer index of the question from the numbered list.",
    )
    question: str = Field(
        ...,
        description="The exact question text (trimmed) corresponding to one input question.",
    )
    label: Literal["LLQ", "DRQ", "GDQ"] = Field(
        ...,
        description="Top-level question type: 'LLQ', 'DRQ', or 'GDQ'.",
    )
    subcategory: Subcategory | None = Field(
        default=None,
        description=(
            "Fine-grained subcategory from Eris (2004), compatible with the top-level label. "
            "(e.g. 'Verification', 'Causal Antecedent', 'Ideation'). "
            "Optional to support runs where subcategory labelling is disabled."
        ),
    )


class BatchLabels(BaseModel):
    """Batch of labelled questions, one item per input question."""

    items: list[QuestionLabel] = Field(
        ...,
        description="One entry per input question, in the same order as provided.",
    )
