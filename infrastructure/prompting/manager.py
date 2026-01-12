"""
Prompt loading and management.
Handles loading prompts from disk and optionally registering them in the Opik prompt library.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

import opik
from opik import Prompt, PromptType

from infrastructure.config.models import Provider, RunConfig
from infrastructure.io import ensure_exists, read_text

logger = logging.getLogger(__name__)


class PromptRole(Enum):
    """Role of the prompt in the LLM interaction."""

    SYSTEM = "system"
    USER = "user"

    def default_relative_path(self, cfg: "RunConfig") -> Path:
        """Return the provider-relative prompt path, matching the on-disk directory structure."""
        label_dir = "sub-category" if cfg.label_subcategories else "category"

        if self is PromptRole.SYSTEM:
            # prompts/<provider>/system/<category|sub-category>/chat-system.txt
            return Path(self.value) / label_dir / "chat-system.txt"

        if self is PromptRole.USER:
            # prompts/<provider>/user/label/<category|sub-category>/
            # icl-demo/<none|category|sub-category>/classify-questions.txt
            if not cfg.include_icl_demo:
                icl_dir = "none"
            else:
                icl_dir = "sub-category" if cfg.include_subcategory_in_icl_demo else "category"

            return Path(self.value) / "label" / label_dir / "icl-demo" / icl_dir / "classify-questions.txt"

        raise ValueError(f"Unsupported role: {self}")


@dataclass(frozen=True)
class LocalPrompt:
    """Lightweight prompt wrapper for disk-only prompting (no Opik prompt library writes)."""

    name: str
    prompt: str
    metadata: dict[str, Any]

    def format(self, **kwargs: Any) -> str:
        """Format the prompt template with variables using Mustache syntax."""
        # Minimal mustache replacement for {{key}} placeholders.
        rendered = self.prompt

        # for k, v in kwargs.items():
        #     pattern = r"\{\{\s*" + re.escape(k) + r"\s*\}\}"
        #     rendered = re.sub(pattern, lambda m, v=v: str(v), rendered)
        for k in sorted(kwargs, key=lambda x: len(str(x)), reverse=True):
            v = str(kwargs[k])
            # Try exact match first (faster)
            placeholder = f"{{{{{k}}}}}"
            if placeholder in rendered:
                rendered = rendered.replace(placeholder, v)
            else:
                # Fallback to regex for whitespace tolerance
                pattern = re.compile(r"\{\{\s*" + re.escape(str(k)) + r"\s*\}\}")
                rendered = pattern.sub(v, rendered)
        return rendered


PromptObj: TypeAlias = Prompt | LocalPrompt


class PromptManager:
    """
    Manages prompt loading from disk and (optionally) registers them in the Opik prompt library

    Prompts are organized as:
        prompts/
        ├─ <vendor1>/
        │  ├─ system/<condition>/chat-system.txt
        │  └─ user/<label>/<condition>/icl-demo/<condition>/classify-questions.txt
        └─ <vendor2>/
           ├─ system/<condition>/chat-system.txt
           └─ user/<label>/<condition>/icl-demo/<condition>/classify-questions.txt
    """

    def __init__(self, prompts_root: Path):
        self.prompts_root = prompts_root
        self.client = opik.Opik()
        self._cache: dict[tuple[str, str, str, int, bool], PromptObj] = {}

    def _get_prompt_path(
        self,
        provider: Provider,
        role: PromptRole,
        cfg: "RunConfig",
        override_path: Path | None = None,
    ) -> Path:
        if override_path is not None:
            return override_path
        return self.prompts_root / provider.value / role.default_relative_path(cfg)

    def _make_opik_prompt_name(self, provider: Provider, role: PromptRole, path: Path) -> str:
        """
        Build a stable Opik prompt name derived from provider, role, and relative path.

        Format:
          {provider}.{role}.{relative_path_with_dots}

        - If `path` is under prompts_root/provider, use that relative path to avoid
          duplicating provider segments (e.g., avoid `openai.system.openai.system...`).
        - Otherwise, fall back to a normalized path string.
        """
        provider_root = (self.prompts_root / provider.value).resolve()
        p = path.resolve()

        try:
            rel = p.relative_to(provider_root)
            rel_str = rel.as_posix()
        except ValueError:
            # Path is outside provider_root (e.g., override_path elsewhere). Use a normalized path string.
            rel_str = p.as_posix()

        if rel_str.endswith(".txt"):
            rel_str = rel_str.removesuffix(".txt")

        # rel_str = rel_str.strip("/").replace("/", ".")
        # return f"{provider.value}.{role.value}.{rel_str}"
        prefix = f"{role.value}/"
        if rel_str.startswith(prefix):
            rel_str = rel_str[len(prefix) :]

        rel_str = rel_str.replace("/", ".")
        return f"{provider.value}.{role.value}.{rel_str}"

    def get_prompt(
        self,
        provider: Provider,
        role: PromptRole,
        cfg: "RunConfig",
        override_path: Path | None = None,
    ) -> PromptObj:
        """
        Load a prompt from disk and (optionally) register it in the Opik prompt library.

        :param provider:
        :param role: Prompt role (SYSTEM or USER) to resolve the relative path.
        :param cfg: (RunConfig): Runtime configuration
        :param override_path: If provided, use this file instead of the provider/default path

        :return:
        PromptObj: object implementing the minimal prompt interface (`name`, `prompt`, `metadata`, `format`)

        Raises:
        FileNotFoundError: if the resolved prompt file does not exist.
        ValueError: for invalid prompt content or failures during optional Opik registration
        """

        prompt_path = self._get_prompt_path(provider, role, cfg, override_path)
        ensure_exists(prompt_path, f"{provider.value}:{role.value}-prompt")
        mtime_ns = prompt_path.stat().st_mtime_ns

        cache_key = (
            provider.value,
            role.value,
            str(prompt_path),
            mtime_ns,
            cfg.prompts_register_in_opik,
        )

        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt_text = read_text(prompt_path)
        prompt_name = self._make_opik_prompt_name(provider, role, prompt_path)

        metadata = {
            "provider": provider.value,
            "role": role.value,
            "source_path": str(prompt_path),
            "label_subcategories": cfg.label_subcategories,
            "include_icl_demo": cfg.include_icl_demo,
            "include_subcategory_in_icl_demo": cfg.include_subcategory_in_icl_demo,
        }

        if cfg.prompts_register_in_opik:
            # Creates/versions the prompt in the Opik prompt library.
            try:
                prompt_obj: PromptObj = Prompt(
                    name=prompt_name,
                    prompt=prompt_text,
                    type=PromptType.MUSTACHE,
                    metadata=metadata,
                )
            except Exception as e:
                raise ValueError(f"Failed to create/register prompt '{prompt_name}' in Opik prompt library.") from e
        else:
            prompt_obj = LocalPrompt(name=prompt_name, prompt=prompt_text, metadata=metadata)

        self._cache[cache_key] = prompt_obj
        logger.info("Loaded %s prompt from %s as %s", role.value, prompt_path, prompt_name)
        return prompt_obj

    def load_prompt(self, name: str, commit: str | None = None) -> Any:
        # Optional helper to fetch a specific prompt version from Opik.
        if commit:
            return self.client.get_prompt(name=name, commit=commit)
        return self.client.get_prompt(name=name)
