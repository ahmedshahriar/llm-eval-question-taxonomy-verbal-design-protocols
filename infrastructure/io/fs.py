"""Filesystem utility functions."""

from pathlib import Path


def ensure_exists(path: Path, what: str) -> None:
    """
    Check that a path exists, raise FileNotFoundError if not.

    Args:
        path: Path to check
        what: Description of what this path represents (for error message)

    Raises:
        FileNotFoundError: If path does not exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {what} at: {path}")


def read_text(path: Path) -> str:
    """
    Read text file with UTF-8 encoding and strip whitespace.

    Args:
        path: Path to text file

    Returns:
        File contents with leading/trailing whitespace removed
    """
    return path.read_text(encoding="utf-8").strip()
