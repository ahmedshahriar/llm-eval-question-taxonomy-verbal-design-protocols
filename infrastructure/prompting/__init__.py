"""
Prompt management: loading from disk and Opik integration.

Handles:
- Loading prompt templates from filesystem
- Optional registration in Opik prompt library
- Mustache-style template rendering

The PromptManager handles both local (disk-only) and Opik-integrated prompts.
"""

from infrastructure.prompting.manager import (
    LocalPrompt,
    PromptManager,
)

__all__ = [
    "PromptManager",
    "LocalPrompt",
]
