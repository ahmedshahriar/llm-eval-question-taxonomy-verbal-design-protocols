"""Anthropic provider adapter with prompt caching support."""

import logging
from collections.abc import Sequence
from typing import Any

from anthropic import Anthropic, transform_schema
from opik import opik_context
from opik.integrations.anthropic import track_anthropic

from domain.schemas import BatchLabels
from infrastructure.config.models import ModelPricingAnthropic, Provider, RunConfig

from .base import ProviderAdapter
from .registry import register_adapter

logger = logging.getLogger(__name__)


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic API with prompt caching."""

    supports_prompt_caching: bool = True

    @classmethod
    def from_cfg(cls, cfg: RunConfig) -> "AnthropicAdapter":
        if cfg.anthropic is None:
            raise ValueError("Provider=anthropic but cfg.anthropic is missing")
        client: Any = track_anthropic(Anthropic())
        pricing = ModelPricingAnthropic(**cfg.provider_model.pricing) if cfg.provider_model.pricing else None
        if pricing is None:
            raise ValueError(
                "Missing Anthropic pricing configuration (required for cost parity / cost tracking). "
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
        """Call Anthropic API with structured output and prompt caching."""

        provider_cfg = self.cfg.anthropic
        if provider_cfg is None:
            raise ValueError("AnthropicAdapter requires cfg.anthropic")

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

        # Only include cache_control when ttl is set; some SDK/API versions reject ttl=None.
        system_block: dict[str, Any] = {
            "type": "text",
            "text": system_text,
        }
        if provider_cfg.cache_ttl:
            system_block["cache_control"] = {
                "type": "ephemeral",
                "ttl": provider_cfg.cache_ttl,
            }

        message = self.client.beta.messages.create(
            model=self.model,
            max_tokens=provider_cfg.max_tokens,
            temperature=provider_cfg.temperature,
            service_tier=provider_cfg.service_tier,
            betas=provider_cfg.betas,
            system=[system_block],
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}],
                }
            ],
            output_format={
                "type": "json_schema",
                "schema": transform_schema(response_model),
            },
        )

        raw_text = message.content[0].text
        try:
            parsed = response_model.model_validate_json(raw_text)  # pydantic v2
        except Exception as e:
            raise ValueError(
                f"Failed to parse Claude response as {response_model.__name__}. Raw text:\n{raw_text}"
            ) from e

        self._validate_indices(parsed=parsed, questions=questions, batch_id=batch_id)

        usage = message.usage

        uncached_after_breakpoint = int(getattr(usage, "input_tokens", 0) or 0)
        cache_creation_total = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        cache_creation_obj = getattr(usage, "cache_creation", None)
        created_5m = 0
        created_1h = 0
        if isinstance(cache_creation_obj, dict):
            created_5m = int(cache_creation_obj.get("ephemeral_5m_input_tokens", 0) or 0)
            created_1h = int(cache_creation_obj.get("ephemeral_1h_input_tokens", 0) or 0)
        elif cache_creation_obj is not None:
            created_5m = int(getattr(cache_creation_obj, "ephemeral_5m_input_tokens", 0) or 0)
            created_1h = int(getattr(cache_creation_obj, "ephemeral_1h_input_tokens", 0) or 0)

        breakdown_sum = created_5m + created_1h
        if breakdown_sum > 0:
            if 0 < cache_creation_total != breakdown_sum:
                logger.warning(
                    "Anthropic cache_creation mismatch: cache_creation_input_tokens=%d but breakdown_sum=%d",
                    cache_creation_total,
                    breakdown_sum,
                )
            cache_creation_total = breakdown_sum
        else:
            if cache_creation_total > 0:
                if provider_cfg.cache_ttl == "1h":
                    created_1h = cache_creation_total
                else:
                    created_5m = cache_creation_total

        input_tokens = uncached_after_breakpoint + cache_creation_total + cache_read_tokens
        total_tokens = input_tokens + output_tokens

        result["input_tokens"] = input_tokens
        result["output_tokens"] = output_tokens
        result["total_tokens"] = total_tokens

        result["cache_meta"] = {
            "uncached_input_tokens": uncached_after_breakpoint,
            "cache_read_input_tokens": cache_read_tokens,
            "cache_creation_5m_input_tokens": created_5m,
            "cache_creation_1h_input_tokens": created_1h,
        }

        baseline_input_cost = 0.0
        input_savings = 0.0
        savings_pct = 0.0

        if self.pricing is not None:
            baseline_input_cost = input_tokens * self.pricing.input_cost_per_token
            result["batch_input_cost"] = (
                uncached_after_breakpoint * self.pricing.input_cost_per_token
                + cache_read_tokens * self.pricing.cache_read_cost_per_token
                + created_5m * self.pricing.cache_write_5m_cost_per_token
                + created_1h * self.pricing.cache_write_1h_cost_per_token
            )
            result["batch_output_cost"] = output_tokens * self.pricing.output_cost_per_token
            result["batch_total_cost"] = result["batch_input_cost"] + result["batch_output_cost"]

            input_savings = baseline_input_cost - result["batch_input_cost"]
            savings_pct = (input_savings / baseline_input_cost * 100) if baseline_input_cost > 0 else 0.0

        cache_hit = cache_read_tokens > 0
        cache_utilization_pct = (cache_read_tokens / input_tokens) * 100 if input_tokens > 0 else 0.0
        cacheable_total = cache_read_tokens + cache_creation_total
        cacheable_hit_rate = (cache_read_tokens / cacheable_total) * 100 if cacheable_total > 0 else 0.0

        meta = {
            "batch_id": batch_id,
            "service_tier": provider_cfg.service_tier,
            "cache_ttl": provider_cfg.cache_ttl,
            "max_tokens": provider_cfg.max_tokens,
            "temperature": provider_cfg.temperature,
            "betas": provider_cfg.betas,
            "anthropic_uncached_tokens": uncached_after_breakpoint,
            "anthropic_cache_read_tokens": cache_read_tokens,
            "anthropic_cache_write_5m_tokens": created_5m,
            "anthropic_cache_write_1h_tokens": created_1h,
            "anthropic_cache_creation_tokens": cache_creation_total,
            "baseline_input_cost_usd": round(baseline_input_cost, 6),
            "batch_input_cost_usd": round(result["batch_input_cost"], 6),
            "batch_output_cost_usd": round(result["batch_output_cost"], 6),
            "batch_total_cost_usd": round(result["batch_total_cost"], 6),
            "input_savings_usd": round(input_savings, 6),
            "savings_percent": round(savings_pct, 1),
            "cache_hit": cache_hit,
            "cache_utilization_pct": round(cache_utilization_pct, 2),
            "cacheable_hit_rate_pct": round(cacheable_hit_rate, 2),
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
            "Batch %d: Anthropic - total_tokens=%d (uncached=%d, cache_read=%d, cache_write=%d [5m=%d, 1h=%d]), "
            "cost=$%.6f, savings=$%.6f (%.1f%%)",
            batch_id,
            total_tokens,
            uncached_after_breakpoint,
            cache_read_tokens,
            cache_creation_total,
            created_5m,
            created_1h,
            result["batch_total_cost"],
            input_savings,
            savings_pct,
        )

        result["batch_output_payload"] = parsed.model_dump()
        return parsed, result


register_adapter(Provider.ANTHROPIC, AnthropicAdapter)
