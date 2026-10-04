"""
src/data_loader.py
==================
Load raw and processed PVDAQ CSV files, apply column normalisation,
and return clean DataFrames ready for the pipeline.

This module is intentionally simple — it is the single source of truth
for reading data from disk. Preprocessing logic lives in preprocessing.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    CLEAN_CSV,
    COLUMN_MAP,
    RAW_MERGED_CSV,
    REQUIRED_COLUMNS,
    TIMESTAMP_COL,
)

log = logging.getLogger(__name__)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw dataset columns to application-level names.

    Uses COLUMN_MAP from config.  Unknown columns are preserved unchanged.
    Matching is case-insensitive and whitespace-stripped.
    """
    rename: dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_MAP and COLUMN_MAP[key] != col:
            rename[col] = COLUMN_MAP[key]
    if rename:
        log.debug("Column renames applied: %s", rename)
    return df.rename(columns=rename)


def _validate_required_columns(df: pd.DataFrame, source: str) -> None:
    """
    Raise ValueError if any required column is missing after normalisation.

    Parameters
    ----------
    df : DataFrame after column normalisation
    source : label used in error messages (e.g. file path)
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing in '{source}': {sorted(missing)}\n"
            f"Available columns: {sorted(df.columns)}"
        )


def load_raw_merged(path: Path | None = None) -> pd.DataFrame:
    """
    Load the merged raw CSV produced by scripts/download_data.py.

    Parameters
    ----------
    path : Optional path override. Defaults to config.RAW_MERGED_CSV.

    Returns
    -------
    pd.DataFrame with normalised column names.

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ValueError if required columns are missing.
    """
    csv_path = Path(path) if path else RAW_MERGED_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw merged CSV not found: {csv_path}\n"
            "Run 'python scripts/download_data.py' first."
        )

    log.info("Loading raw merged CSV: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    log.info("  Loaded %d rows × %d columns", len(df), len(df.columns))

    df = _normalise_columns(df)
    _validate_required_columns(df, str(csv_path))

    # Parse timestamp column
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=True)
        nat_count = df[TIMESTAMP_COL].isna().sum()
        if nat_count:
            log.warning("  %d unparseable timestamps set to NaT", nat_count)

    return df


def load_clean(path: Path | None = None) -> pd.DataFrame:
    """
    Load the preprocessed/cleaned CSV produced by src/preprocessing.py.

    Parameters
    ----------
    path : Optional path override. Defaults to config.CLEAN_CSV.

    Returns
    -------
    pd.DataFrame, sorted by timestamp.

    Raises
    ------
    FileNotFoundError if the file does not exist.
    """
    csv_path = Path(path) if path else CLEAN_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed clean CSV not found: {csv_path}\n"
            "Run 'python src/preprocessing.py' first."
        )

    log.info("Loading clean CSV: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    log.info("  Loaded %d rows × %d columns", len(df), len(df.columns))

    # Restore datetime type
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)

    return df


def dataset_summary(df: pd.DataFrame) -> str:
    """
    Return a human-readable summary string for a loaded DataFrame.

    Includes row count, date range, missing-value statistics, and column list.
    """
    lines: list[str] = ["Dataset Summary", "=" * 50]
    lines.append(f"  Rows:        {len(df):,}")
    lines.append(f"  Columns:     {len(df.columns)}")
    lines.append(f"  Column list: {list(df.columns)}")

    if TIMESTAMP_COL in df.columns:
        ts = df[TIMESTAMP_COL]
        lines.append(f"  Date range:  {ts.min()}  →  {ts.max()}")
        span = ts.max() - ts.min()
        lines.append(f"  Time span:   {span}")

    lines.append("  Missing values per column:")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        if n_miss:
            pct = 100 * n_miss / len(df)
            lines.append(f"    {col:<30} {n_miss:>6,}  ({pct:.1f}%)")

    dtypes = df.dtypes.to_dict()
    lines.append("  Data types:")
    for col, dtype in dtypes.items():
        lines.append(f"    {col:<30} {str(dtype)}")

    return "\n".join(lines)
