"""I/O utilities: filesystem operations and dataset loading."""

from infrastructure.io.datasets import read_table
from infrastructure.io.fs import ensure_exists, read_text

__all__ = [
    "ensure_exists",
    "read_text",
    "read_table",
]
