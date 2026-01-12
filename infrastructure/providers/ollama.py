import logging
from collections.abc import Sequence
from typing import Any

import httpx
from opik import opik_context

from domain import BatchLabels
from infrastructure import Provider, ProviderAdapter, RunConfig
from infrastructure.providers.registry import register_adapter

logger = logging.getLogger(__name__)


class OllamaAdapter(ProviderAdapter):
    """
    Ollama backend using Ollama's native Chat API: POST /api/chat

    - Uses JSON schema constrained output via `format` (json schema object)
    - Token usage comes from `prompt_eval_count` and `eval_count`
    - Cost is treated as $0.00 by default (local inference)
    """

    supports_structured_outputs: bool = True
    supports_prompt_caching: bool = False
    supports_token_usage: bool = True

    @classmethod
    def from_cfg(cls, cfg: RunConfig) -> "OllamaAdapter":
        base_url = cfg.ollama.base_url.rstrip("/")
        client = httpx.Client(
            base_url=base_url,
            timeout=cfg.ollama.timeout_s,
            headers={"Content-Type": "application/json"},
        )
        # No per-token pricing for local by default
        return cls(cfg=cfg, client=client, pricing=None)

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
        # Ollama supports format as either "json" or a JSON schema object
        schema_obj = response_model.model_json_schema()

        # Ollama "options" are optional; only send non-None values
        options: dict[str, Any] = {}
        for k in ("temperature", "seed", "num_ctx", "top_p", "top_k", "num_predict"):
            v = getattr(self.cfg.ollama, k, None)
            if v is not None:
                options[k] = v

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "format": schema_obj,
        }

        if options:
            payload["options"] = options
        if self.cfg.ollama.keep_alive is not None:
            payload["keep_alive"] = self.cfg.ollama.keep_alive
        if self.cfg.ollama.think is not None:
            payload["think"] = self.cfg.ollama.think

        resp = self.client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        raw_text = ((data.get("message") or {}).get("content")) or ""
        try:
            parsed = response_model.model_validate_json(raw_text)
        except Exception as e:
            raise ValueError(
                f"Failed to parse Ollama response as {response_model.__name__}. Raw text:\n{raw_text}"
            ) from e

        self._validate_indices(parsed=parsed, questions=questions, batch_id=batch_id)

        input_tokens = int(data.get("prompt_eval_count", 0) or 0)
        output_tokens = int(data.get("eval_count", 0) or 0)
        total_tokens = int(input_tokens + output_tokens)

        # Keep the same meta keys your pipeline expects
        result: dict[str, Any] = {
            "provider": self.provider.value,
            "model": self.model,
            "batch_id": batch_id,
            "raw_text": raw_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            # Local inference: default $0.00
            "batch_input_cost": 0.0,
            "batch_output_cost": 0.0,
            "batch_total_cost": 0.0,
            "batch_output_payload": None,
        }

        meta = {
            "batch_id": batch_id,
            "base_url": self.cfg.ollama.base_url,
            "keep_alive": self.cfg.ollama.keep_alive,
            "think": self.cfg.ollama.think,
            "ollama_options": options,
            "batch_total_cost_usd": 0.0,
        }
        if extra_trace_meta:
            meta.update(extra_trace_meta)

        # Match your existing Opik conventions (OpenAI-style usage keys)
        opik_context.update_current_span(
            provider=self.provider.value,
            model=self.model,
            usage=self._opik_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
            total_cost=0.0,
            metadata=meta,
        )

        logger.info(
            "Batch %d: Ollama - total_tokens=%d (in=%d, out=%d), cost=$%.6f",
            batch_id,
            total_tokens,
            input_tokens,
            output_tokens,
            0.0,
        )

        return parsed, result


register_adapter(Provider.OLLAMA, OllamaAdapter)
