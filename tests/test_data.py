"""
tests/test_data.py
==================
Tests for data loading and column validation.

These tests use the synthetic sample fixture from conftest.py and do NOT
require the real PVDAQ dataset to be downloaded.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import FEATURE_COLS_ORDERED, REQUIRED_COLUMNS, TARGET_COL, TIMESTAMP_COL


class TestRequiredColumns:
    """Verify that required column constants are correctly defined."""

    def test_required_columns_non_empty(self):
        assert len(REQUIRED_COLUMNS) > 0

    def test_timestamp_in_required(self):
        assert TIMESTAMP_COL in REQUIRED_COLUMNS

    def test_target_in_required(self):
        assert TARGET_COL in REQUIRED_COLUMNS

    def test_irradiance_in_required(self):
        assert "irradiance" in REQUIRED_COLUMNS


class TestSampleDataFixture:
    """Test that the shared sample fixture has the expected structure."""

    def test_has_required_columns(self, sample_df: pd.DataFrame):
        for col in REQUIRED_COLUMNS:
            assert col in sample_df.columns, f"Missing required column: {col}"

    def test_timestamp_is_datetime(self, sample_df: pd.DataFrame):
        assert pd.api.types.is_datetime64_any_dtype(sample_df[TIMESTAMP_COL])

    def test_power_is_numeric(self, sample_df: pd.DataFrame):
        assert pd.api.types.is_numeric_dtype(sample_df[TARGET_COL])

    def test_irradiance_is_numeric(self, sample_df: pd.DataFrame):
        assert pd.api.types.is_numeric_dtype(sample_df["irradiance"])

    def test_no_negative_irradiance(self, sample_df: pd.DataFrame):
        assert (sample_df["irradiance"] >= 0).all()

    def test_power_non_negative(self, sample_df: pd.DataFrame):
        assert (sample_df[TARGET_COL] >= 0).all()

    def test_row_count(self, sample_df: pd.DataFrame):
        assert len(sample_df) == 5


class TestColumnMapping:
    """Test column mapping from raw to application names."""

    def test_column_map_has_irradiance(self):
        from scripts.download_data import COLUMN_MAP
        assert "poa_irradiance" in COLUMN_MAP
        assert COLUMN_MAP["poa_irradiance"] == "irradiance"

    def test_column_map_has_power(self):
        from scripts.download_data import COLUMN_MAP
        assert "ac_power" in COLUMN_MAP
        assert COLUMN_MAP["ac_power"] == "power"

    def test_column_map_has_timestamp(self):
        from scripts.download_data import COLUMN_MAP
        assert "measdatetime" in COLUMN_MAP
        assert COLUMN_MAP["measdatetime"] == "timestamp"

    def test_normalise_columns_renames(self):
        from scripts.download_data import COLUMN_MAP
        # Simulate a raw DataFrame with PVDAQ column names
        raw_df = pd.DataFrame({
            "measdatetime": ["2012-01-01 12:00:00"],
            "ac_power": [1.5],
            "poa_irradiance": [700.0],
        })
        # Apply manual normalisation
        rename = {col: COLUMN_MAP[col.strip().lower()]
                  for col in raw_df.columns
                  if col.strip().lower() in COLUMN_MAP}
        result = raw_df.rename(columns=rename)
        assert "timestamp" in result.columns
        assert "power" in result.columns
        assert "irradiance" in result.columns
