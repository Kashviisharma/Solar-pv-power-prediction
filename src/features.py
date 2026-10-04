"""
src/features.py
===============
Feature selection and engineering utilities for solar PV prediction.

This module provides:
- Feature selection based on what's actually in the data
- Time-based feature engineering (hour, month, etc.)
- Feature-target split
- Train/test split with time-aware ordering

All feature engineering is done without data leakage — only information
available at prediction time is used.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLS_ORDERED,
    TARGET_COL,
    TIMESTAMP_COL,
    TRAIN_FRACTION,
)

log = logging.getLogger(__name__)


def get_available_features(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns that are actually present in the data.
    
    This ensures the model only uses available columns without fabricating data.
    """
    available = []
    for col in FEATURE_COLS_ORDERED:
        if col in df.columns:
            available.append(col)
    
    log.info("Available features: %s", available)
    return available


def split_features_target(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).
    
    Parameters
    ----------
    df : DataFrame with timestamp, features, and target
    feature_cols : Optional list of feature columns. If None, uses get_available_features()
    
    Returns
    -------
    X : DataFrame of features
    y : Series of target values
    """
    if feature_cols is None:
        feature_cols = get_available_features(df)
    
    # Ensure all feature columns exist
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Feature columns missing from DataFrame: {missing}")
    
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    
    log.info("Feature-target split: X shape %s, y shape %s", X.shape, y.shape)
    return X, y


def time_based_train_test_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into training and test sets.
    
    This is critical for time-series data to prevent data leakage.
    Random splitting would allow the model to "see the future."
    
    Parameters
    ----------
    df : DataFrame sorted by timestamp
    train_fraction : Fraction of data to use for training (first N% chronologically)
    
    Returns
    -------
    train_df : First train_fraction of data
    test_df : Remaining data (represents future time periods)
    """
    if not pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COL]):
        raise ValueError(f"DataFrame must have {TIMESTAMP_COL} as datetime")
    
    # Ensure sorted
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    
    split_idx = int(len(df) * train_fraction)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    log.info("Time-based split:")
    log.info("  Train: %d rows (%.1f%%), %s → %s",
            len(train_df), 100 * len(train_df) / len(df),
            train_df[TIMESTAMP_COL].min(), train_df[TIMESTAMP_COL].max())
    log.info("  Test:  %d rows (%.1f%%), %s → %s",
            len(test_df), 100 * len(test_df) / len(df),
            test_df[TIMESTAMP_COL].min(), test_df[TIMESTAMP_COL].max())
    
    return train_df, test_df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features from timestamp.
    
    These features are available at prediction time (we know the current time).
    
    Features added:
    - hour_of_day (0-23)
    - month_of_year (1-12)
    - day_of_year (1-366)
    - is_weekend (boolean)
    
    Note: These are optional and may not improve performance over raw irradiance.
    They are provided for experimentation.
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COL]):
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    
    df["hour_of_day"] = df[TIMESTAMP_COL].dt.hour
    df["month_of_year"] = df[TIMESTAMP_COL].dt.month
    df["day_of_year"] = df[TIMESTAMP_COL].dt.dayofyear
    df["is_weekend"] = df[TIMESTAMP_COL].dt.dayofweek >= 5  # Saturday=5, Sunday=6
    
    log.info("Added time features: hour_of_day, month_of_year, day_of_year, is_weekend")
    return df


def remove_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove time-based features, keeping only raw sensor measurements.
    
    This is the default approach — the model learns from irradiance and temperature
    directly, which are the actual physical drivers of PV power.
    """
    time_cols = ["hour_of_day", "month_of_year", "day_of_year", "is_weekend"]
    cols_to_drop = [c for c in time_cols if c in df.columns]
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        log.info("Removed time features: %s", cols_to_drop)
    
    return df
