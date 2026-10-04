"""
src/preprocessing.py
====================
Data cleaning and preprocessing pipeline for solar PV data.

This module:
1. Loads raw merged CSV from download_data.py
2. Parses and sorts timestamps
3. Removes duplicates and physically impossible values
4. Handles missing values appropriately
5. Keeps nighttime data (irradiance < 5 W/m²) for realistic zero-power regime
6. Selects only features actually present in the data
7. Saves cleaned CSV for model training

Usage
-----
    python src/preprocessing.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLEAN_CSV,
    FEATURE_COLS_ORDERED,
    NIGHTTIME_IRRADIANCE_THRESHOLD,
    PHYSICAL_BOUNDS,
    REQUIRED_COLUMNS,
    TARGET_COL,
    TIMESTAMP_COL,
)
from src.data_loader import load_raw_merged

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows based on timestamp."""
    initial_count = len(df)
    df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep="first")
    removed = initial_count - len(df)
    if removed > 0:
        log.info("Removed %d duplicate rows", removed)
    return df


def _remove_physically_impossible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with physically impossible values based on PHYSICAL_BOUNDS.
    
    Values outside the bounds are assumed to be sensor errors.
    """
    initial_count = len(df)
    
    for col, (min_val, max_val) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        
        # Count violations
        mask = (df[col] < min_val) | (df[col] > max_val)
        violations = mask.sum()
        
        if violations > 0:
            log.warning(
                "Column '%s': %d rows outside physical bounds [%.2f, %.2f]",
                col, violations, min_val, max_val
            )
            df = df[~mask]
    
    removed = initial_count - len(df)
    if removed > 0:
        log.info("Removed %d rows with physically impossible values", removed)
    
    return df


def _handle_negative_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle negative power values.
    
    Small negative values (< -0.1 kW) are likely inverter standby readings.
    Large negative values indicate sensor errors and are removed.
    """
    if TARGET_COL not in df.columns:
        return df
    
    initial_count = len(df)
    
    # Count small negatives (likely inverter standby)
    small_negatives = (df[TARGET_COL] < 0) & (df[TARGET_COL] >= -0.1)
    small_count = small_negatives.sum()
    
    # Count large negatives (sensor errors)
    large_negatives = df[TARGET_COL] < -0.1
    large_count = large_negatives.sum()
    
    if small_count > 0:
        log.info("Clamping %d small negative power values to 0 (inverter standby)", small_count)
        df.loc[small_negatives, TARGET_COL] = 0
    
    if large_count > 0:
        log.warning("Removing %d rows with large negative power (sensor error)", large_count)
        df = df[~large_negatives]
    
    removed = initial_count - len(df)
    if removed > 0:
        log.info("Removed %d rows due to negative power", removed)
    
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Strategy:
    - For short gaps (< 3 consecutive NaNs): interpolate
    - For longer gaps: drop rows
    - Always require timestamp, power, and irradiance to be non-null
    """
    initial_count = len(df)
    
    # Check missing in required columns
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                log.warning("Column '%s': %d missing values (%.1f%%)", col, missing, 100 * missing / len(df))
    
    # Drop rows where required columns are missing
    df = df.dropna(subset=REQUIRED_COLUMNS)
    
    # For optional features, interpolate short gaps
    for col in FEATURE_COLS_ORDERED:
        if col in df.columns and col not in REQUIRED_COLUMNS:
            missing_before = df[col].isna().sum()
            if missing_before > 0:
                # Linear interpolation
                df[col] = df[col].interpolate(method="linear", limit=3)
                missing_after = df[col].isna().sum()
                filled = missing_before - missing_after
                if filled > 0:
                    log.info("Interpolated %d missing values in '%s'", filled, col)
                # Drop remaining NaNs (long gaps)
                df = df.dropna(subset=[col])
    
    removed = initial_count - len(df)
    if removed > 0:
        log.info("Removed %d rows due to missing values", removed)
    
    return df


def _add_daytime_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a flag indicating whether the row is daytime or nighttime.
    
    Nighttime is defined as irradiance < NIGHTTIME_IRRADIANCE_THRESHOLD.
    This flag is useful for analysis but not used as a model feature.
    """
    if "irradiance" in df.columns:
        df["is_daytime"] = df["irradiance"] >= NIGHTTIME_IRRADIANCE_THRESHOLD
        daytime_count = df["is_daytime"].sum()
        nighttime_count = len(df) - daytime_count
        log.info("Daytime rows: %d, Nighttime rows: %d", daytime_count, nighttime_count)
    return df


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the features that are actually present in the data.
    
    Ensures the model uses only available columns without fabricating data.
    """
    # Always keep timestamp and target
    cols_to_keep = [TIMESTAMP_COL, TARGET_COL]
    
    # Add features that are present
    for col in FEATURE_COLS_ORDERED:
        if col in df.columns:
            cols_to_keep.append(col)
    
    # Add auxiliary columns if present
    for col in ["dc_power", "is_daytime"]:
        if col in df.columns:
            cols_to_keep.append(col)
    
    # Select columns
    df = df[cols_to_keep]
    
    log.info("Selected columns: %s", cols_to_keep)
    return df


def preprocess() -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Returns
    -------
    Cleaned DataFrame ready for model training.
    """
    log.info("=" * 60)
    log.info("Preprocessing Pipeline")
    log.info("=" * 60)
    
    # Load raw data
    df = load_raw_merged()
    log.info("Loaded raw data: %d rows × %d columns", len(df), len(df.columns))
    
    # Parse timestamp if not already done
    if not pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COL]):
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=True)
        df = df.dropna(subset=[TIMESTAMP_COL])
    
    # Sort chronologically
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    log.info("Sorted by timestamp")
    
    # Remove duplicates
    df = _remove_duplicates(df)
    
    # Remove physically impossible values
    df = _remove_physically_impossible(df)
    
    # Handle negative power
    df = _handle_negative_power(df)
    
    # Handle missing values
    df = _handle_missing_values(df)
    
    # Add daytime flag
    df = _add_daytime_flag(df)
    
    # Select features
    df = _select_features(df)
    
    # Final sort
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    
    log.info("")
    log.info("Preprocessing complete: %d rows × %d columns", len(df), len(df.columns))
    log.info("Date range: %s → %s", df[TIMESTAMP_COL].min(), df[TIMESTAMP_COL].max())
    
    return df


def main() -> int:
    """Run preprocessing and save cleaned data."""
    df = preprocess()
    
    # Save cleaned data
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    log.info("Saved cleaned data: %s", CLEAN_CSV)
    
    # Print summary statistics
    log.info("")
    log.info("Summary Statistics:")
    for col in [TARGET_COL] + [c for c in FEATURE_COLS_ORDERED if c in df.columns]:
        if col in df.columns:
            log.info("  %s: min=%.2f, max=%.2f, mean=%.2f, median=%.2f",
                    col, df[col].min(), df[col].max(), df[col].mean(), df[col].median())
    
    log.info("")
    log.info("Next step:")
    log.info("  python src/train.py")
    log.info("  — or open notebooks/01_data_exploration.ipynb")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
