"""Base adapter interface for LLM providers."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from domain.schemas import BatchLabels
from infrastructure.config.models import Provider, RunConfig

logger = logging.getLogger(__name__)


class ProviderAdapter(ABC):
    """
    Abstract base class for LLM provider adapters.
    Common interface for provider backends (OpenAI, Anthropic, etc.).

    All concrete adapters must implement:
    - call_batch(): Make an LLM API call with structured output
    """

    provider: Provider
    cfg: RunConfig
    client: Any
    pricing: Any | None

    supports_structured_outputs: bool = True
    supports_prompt_caching: bool = False
    supports_token_usage: bool = True

    def __init__(
        self,
        *,
        cfg: RunConfig,
        client: Any,
        pricing: Any | None,
    ) -> None:
        self.cfg = cfg
        self.provider = cfg.provider
        self.client = client
        self.pricing = pricing

    @property
    def model(self) -> str:
        # Single source of truth (no duplicated "model: str" fields)
        return self.cfg.model

    def _validate_indices(
        self,
        *,
        parsed: BaseModel,
        questions: Sequence[str] | None,
        batch_id: int,
    ) -> None:
        """Validate that the model returned exactly one label for each input question.

        Expectations:
        - Indices should be 1..N (matching the numbered list passed to the model).
        - We tolerate the common 0..N-1 mistake by shifting indices +1 once (with a warning).
        - Otherwise, we fail fast with a detailed message (missing/duplicate/out-of-range).
        """
        if questions is None:
            return

        items = getattr(parsed, "items", None)
        if not isinstance(items, list):
            return

        n = len(questions)
        expected = list(range(1, n + 1))

        indices: list[int | None] = [getattr(it, "index", None) for it in items]
        indices_not_none = [idx for idx in indices if idx is not None]
        if len(indices_not_none) != len(indices):
            # if any(idx is None for idx in indices):
            logger.warning(
                "Batch %d: Some returned items have index=None; skipping strict index validation.",
                batch_id,
            )
            return

        # Type guard
        try:
            indices_int = [int(idx) for idx in indices_not_none]
        except (TypeError, ValueError) as err:
            raise ValueError(f"Batch {batch_id}: Non-integer indices returned: {indices!r}") from err

        # Tolerate 0-based indexing exactly once.
        if sorted(indices_int) == list(range(0, n)):
            logger.warning(
                "Batch %d: Model returned 0-based indices (0..%d). Shifting to 1-based (1..%d).",
                batch_id,
                n - 1,
                n,
            )
            for it in items:
                it.index = int(it.index) + 1
            indices_int = [i + 1 for i in indices_int]

        # Detailed checks: duplicates, missing, out-of-range
        out_of_range = sorted({i for i in indices_int if i < 1 or i > n})
        if out_of_range:
            raise ValueError(f"Batch {batch_id}: Out-of-range indices {out_of_range}; expected 1..{n}.")

        missing = [i for i in expected if i not in indices_int]
        duplicates = sorted({i for i in indices_int if indices_int.count(i) > 1})

        if missing or duplicates or len(indices_int) != n:
            msg = f"Batch {batch_id}: Index mismatch (expected exactly 1..{n})."
            if len(indices_int) != n:
                msg += f" Returned {len(indices_int)} items."
            if missing:
                msg += f" Missing indices: {missing}."
            if duplicates:
                msg += f" Duplicate indices: {duplicates}."
            raise ValueError(msg)

        # Finally, ensure set matches exactly (ordering doesn't matter here)
        if set(indices_int) != set(expected):
            extra = sorted(set(indices_int) - set(expected))
            missing2 = sorted(set(expected) - set(indices_int))
            msg = f"Batch {batch_id}: Index set mismatch."
            if missing2:
                msg += f" Missing indices: {missing2}."
            if extra:
                msg += f" Extra indices: {extra}."
            raise ValueError(msg)

    def _opik_usage(self, *, input_tokens: int, output_tokens: int, total_tokens: int) -> dict[str, int]:
        # Opik dashboard expects OpenAI-style keys
        return {
            "prompt_tokens": int(input_tokens),
            "completion_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }

    @abstractmethod
    def call_batch(
        self,
        *,
        system_text: str,
        user_text: str,
        response_model: type[BatchLabels],
        batch_id: int,
        questions: Sequence[str] | None = None,
        extra_trace_meta: dict[str, Any] | None = None,
    ) -> tuple[BatchLabels, dict[str, Any]]:
        """Run one batch call and return (parsed_model, normalized_meta).
        Call the LLM provider with structured output.

        Args:
            system_text: System prompt text
            user_text: User prompt text (formatted with questions)
            response_model: Pydantic model for structured output
            batch_id: Batch identifier for logging
            questions: List of questions in this batch
            extra_trace_meta: Optional extra metadata for tracing/logging

        Returns:
            Tuple of (parsed response model, normalized metadata dict)
        """

        raise NotImplementedError
