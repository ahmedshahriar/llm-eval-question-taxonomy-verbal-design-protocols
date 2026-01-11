"""
Observability: structured logging and context management.

Provides:
- Contextual logging with run/batch IDs
- Log rotation and file management
- Third-party library log level control
"""

from infrastructure.observability.logging import (
    clear_batch_context,
    configure_logging,
    make_run_tag,
    set_log_context,
)

__all__ = [
    "configure_logging",
    "set_log_context",
    "clear_batch_context",
    "make_run_tag",
]
