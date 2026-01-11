"""Dataset loading utilities."""

from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """
    Read tabular data file (Excel or CSV) based on file extension.

    Supported formats:
    - Excel: .xlsx, .xls
    - CSV: .csv

    Args:
        path: Path to data file

    Returns:
        pandas DataFrame

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
        Exception: If file cannot be read (pandas exceptions)
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .xlsx, .xls, .csv")
