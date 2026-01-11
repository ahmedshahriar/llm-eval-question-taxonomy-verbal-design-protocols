"""
Logging setup with contextvars-based metadata injection.

- Adds run_tag and batch_id into every log line (via contextvars).
- Supports console-only logging OR console + rotating file logs.
- Tunes noisy third-party library loggers (httpx, openai, anthropic, etc.).
"""

import contextvars
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Context variables for dynamic log metadata
cv_run_tag = contextvars.ContextVar("run_tag", default="-")
cv_batch_id = contextvars.ContextVar("batch_id", default="-")

# Optional: keep full IDs in context for metadata (not printed every line)
cv_provider = contextvars.ContextVar("provider", default="-")
cv_model = contextvars.ContextVar("model", default="-")
cv_run_id_full = contextvars.ContextVar("run_id_full", default="-")


def make_run_tag(run_id_full: str, length: int = 8) -> str:
    """
    Stable short tag derived from the full run_id.
    8 hex chars is usually enough; bump to 10–12 if running many jobs/day.
    Uses BLAKE2s for collision resistance.
    """
    h = hashlib.blake2s(run_id_full.encode("utf-8"), digest_size=8).hexdigest()
    return h[:length]


class ContextInjectFilter(logging.Filter):
    """Inject context variables into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run = cv_run_tag.get() or "-"
        record.batch = cv_batch_id.get() or "-"
        return True


def set_log_context(
    *,
    run_id_full: str | None = None,
    batch_id: int | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Update logging context (thread-safe via contextvars)."""
    if run_id_full is not None:
        cv_run_id_full.set(str(run_id_full))
        cv_run_tag.set(make_run_tag(str(run_id_full)))

    # Batch: set as zero-padded string
    if batch_id is not None:
        cv_batch_id.set(f"{int(batch_id):03d}")

    # Optional (not printed each line in this Option A formatter)
    if provider is not None:
        cv_provider.set(str(provider))
    if model is not None:
        cv_model.set(str(model))


def get_log_context() -> dict[str, str]:
    """
    Return the current context in a convenient dict form.

    Useful for:
      - attaching `extra_trace_meta` to LLM calls,
      - adding consistent metadata to JSON artifacts,
      - debugging/log correlation.
    """
    return {
        "run_tag": str(cv_run_tag.get() or "-"),
        "run_id_full": str(cv_run_id_full.get() or "-"),
        "batch_id": str(cv_batch_id.get() or "-"),
        "provider": str(cv_provider.get() or "-"),
        "model": str(cv_model.get() or "-"),
    }


def clear_batch_context() -> None:
    """Reset batch context to default (keep run info)."""
    cv_batch_id.set("-")


def configure_logging(
    *,
    log_file: Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10_000_000,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configure application logging with contextvars support.

    Args:
        log_file: Path to log file
        console_level: Minimum level for console output (default: INFO)
        file_level: Minimum level for file output (default: DEBUG)
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Clear existing handlers to avoid duplicate logs if called multiple times
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)  # keep root permissive; handlers enforce levels

    # Formatter includes context fields injected by ContextInjectFilter
    console_fmt = "%(asctime)s [%(levelname)s] r=%(run)s b=%(batch)s | %(message)s"
    file_fmt = "%(asctime)s [%(levelname)s] %(name)s | r=%(run)s b=%(batch)s | %(message)s"

    console_formatter = logging.Formatter(console_fmt, datefmt="%H:%M:%S")
    file_formatter = logging.Formatter(file_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    ctx_filter = ContextInjectFilter()

    # Console handler (human-readable, INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(console_formatter)
    ch.addFilter(ctx_filter)
    root.addHandler(ch)


    # File handler (detailed, DEBUG+, with rotation)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(file_level)
        fh.setFormatter(file_formatter)
        fh.addFilter(ctx_filter)
        root.addHandler(fh)


    # Third-party library log levels
    # Reduce noise from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # LLM provider SDKs
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    # Keep opik at INFO (useful for debugging traces)
    logging.getLogger("opik").setLevel(logging.INFO)

    logging.getLogger(__name__).info(
        "Logging configured (console_level=%s, file=%s)",
        logging.getLevelName(console_level),
        str(log_file) if log_file is not None else "None",
    )
