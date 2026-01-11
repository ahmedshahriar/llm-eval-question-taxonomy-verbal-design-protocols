"""OpenAI provider adapter with prompt caching support."""

import logging
from collections.abc import Sequence
from typing import Any

from openai import OpenAI
from opik import opik_context
from opik.integrations.openai import track_openai

from domain.schemas import BatchLabels
from infrastructure.config.models import ModelPricingOpenAI, Provider, RunConfig

from .base import ProviderAdapter
from .registry import register_adapter

logger = logging.getLogger(__name__)


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI API with prompt caching."""

    supports_prompt_caching: bool = True

    @classmethod
    def from_cfg(cls, cfg: RunConfig) -> "OpenAIAdapter":
        if cfg.openai is None:
            raise ValueError("Provider=openai but cfg.openai is missing")
        client: Any = track_openai(OpenAI())
        pricing = ModelPricingOpenAI(**cfg.provider_model.pricing) if cfg.provider_model.pricing else None
        if pricing is None:
            raise ValueError(
                "Missing OpenAI pricing configuration (required for cost parity / cost tracking). "
                "Add per-token pricing for the selected model in the run config."
            )
        return cls(cfg=cfg, client=client, pricing=pricing)

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
        """Call OpenAI API with structured output and prompt caching."""
        provider_cfg = self.cfg.openai
        if provider_cfg is None:
            raise ValueError("OpenAIAdapter requires cfg.openai")

        result: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_meta": {},
            "batch_input_cost": 0.0,
            "batch_output_cost": 0.0,
            "batch_total_cost": 0.0,
            "batch_output_payload": None,
        }

        # Build kwargs defensively: some SDK versions reject explicit None values.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system_text,
            "input": user_text,
            "text_format": response_model,
        }
        if provider_cfg.service_tier is not None:
            kwargs["service_tier"] = provider_cfg.service_tier
        if provider_cfg.temperature is not None:
            kwargs["temperature"] = provider_cfg.temperature
        if provider_cfg.prompt_cache_key is not None:
            kwargs["prompt_cache_key"] = provider_cfg.prompt_cache_key
        if provider_cfg.prompt_cache_retention is not None:
            kwargs["prompt_cache_retention"] = provider_cfg.prompt_cache_retention

        response = self.client.responses.parse(**kwargs)

        usage = response.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)

        result["input_tokens"] = input_tokens
        result["output_tokens"] = output_tokens
        result["total_tokens"] = total_tokens

        cached_input_tokens = 0
        details = getattr(usage, "input_tokens_details", None) or getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached_input_tokens = int(getattr(details, "cached_tokens", 0) or 0)

        uncached = max(input_tokens - cached_input_tokens, 0)
        result["cache_meta"] = {
            "cached_input_tokens": cached_input_tokens,
            "uncached_input_tokens": uncached,
        }

        if self.pricing is not None:
            result["batch_input_cost"] = (
                uncached * self.pricing.input_cost_per_token
                + cached_input_tokens * self.pricing.cached_input_cost_per_token
            )
            result["batch_output_cost"] = output_tokens * self.pricing.output_cost_per_token
            result["batch_total_cost"] = result["batch_input_cost"] + result["batch_output_cost"]

        parsed: BatchLabels = response.output_parsed
        self._validate_indices(parsed=parsed, questions=questions, batch_id=batch_id)
        result["batch_output_payload"] = parsed.model_dump()

        meta = {
            "batch_id": batch_id,
            "service_tier": provider_cfg.service_tier,
            "temperature": provider_cfg.temperature,
            "prompt_cache_key": provider_cfg.prompt_cache_key,
            "prompt_cache_retention": provider_cfg.prompt_cache_retention,
            "openai_uncached_tokens": uncached,
            "openai_cached_tokens": cached_input_tokens,
            "batch_input_cost_usd": round(result["batch_input_cost"], 6),
            "batch_output_cost_usd": round(result["batch_output_cost"], 6),
            "batch_total_cost_usd": round(result["batch_total_cost"], 6),
        }
        if extra_trace_meta:
            meta.update(extra_trace_meta)

        opik_context.update_current_span(
            provider=self.provider.value,
            model=self.model,
            usage=self._opik_usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens),
            total_cost=float(result["batch_total_cost"]),
            metadata=meta,
        )

        logger.info(
            "Batch %d: OpenAI - total_tokens=%d (uncached=%d, cached=%d), cost=$%.6f",
            batch_id,
            total_tokens,
            uncached,
            cached_input_tokens,
            result["batch_total_cost"],
        )

        return parsed, result


register_adapter(Provider.OPENAI, OpenAIAdapter)
