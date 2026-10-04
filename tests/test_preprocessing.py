"""
tests/test_preprocessing.py
===========================
Tests for data preprocessing pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    _add_daytime_flag,
    _handle_missing_values,
    _handle_negative_power,
    _remove_duplicates,
    _remove_physically_impossible,
    _select_features,
    preprocess,
)


class TestRemoveDuplicates:
    """Test duplicate removal."""
    
    def test_removes_exact_duplicates(self, sample_df: pd.DataFrame):
        """Remove rows with identical timestamps."""
        # Add a duplicate
        df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
        initial_count = len(df)
        
        result = _remove_duplicates(df)
        
        assert len(result) < initial_count
        assert len(result) == initial_count - 1


class TestRemovePhysicallyImpossible:
    """Test removal of physically impossible values."""
    
    def test_removes_irradiance_above_max(self, sample_df: pd.DataFrame):
        """Remove irradiance values above physical maximum."""
        df = sample_df.copy()
        df.loc[0, "irradiance"] = 2000.0  # Above 1500 W/m² max
        
        result = _remove_physically_impossible(df)
        
        assert len(result) < len(df)
    
    def test_removes_negative_irradiance(self, sample_df: pd.DataFrame):
        """Remove negative irradiance values."""
        df = sample_df.copy()
        df.loc[0, "irradiance"] = -10.0
        
        result = _remove_physically_impossible(df)
        
        assert len(result) < len(df)


class TestHandleNegativePower:
    """Test handling of negative power values."""
    
    def test_clamps_small_negatives(self, sample_df: pd.DataFrame):
        """Clamp small negative values to zero."""
        df = sample_df.copy()
        df.loc[0, "power"] = -0.05  # Small negative
        
        result = _handle_negative_power(df)
        
        assert result.loc[0, "power"] == 0.0
    
    def test_removes_large_negatives(self, sample_df: pd.DataFrame):
        """Remove large negative values."""
        df = sample_df.copy()
        df.loc[0, "power"] = -5.0  # Large negative
        
        result = _handle_negative_power(df)
        
        assert len(result) < len(df)


class TestHandleMissingValues:
    """Test missing value handling."""
    
    def test_interpolates_short_gaps(self, sample_df: pd.DataFrame):
        """Interpolate short gaps of missing values."""
        df = sample_df.copy()
        df.loc[1, "ambient_temperature"] = pd.NA
        df.loc[2, "ambient_temperature"] = pd.NA
        
        result = _handle_missing_values(df)
        
        # Should interpolate these values
        assert result["ambient_temperature"].isna().sum() < df["ambient_temperature"].isna().sum()
    
    def test_removes_required_column_nans(self, sample_df: pd.DataFrame):
        """Remove rows where required columns are NaN."""
        df = sample_df.copy()
        df.loc[0, "power"] = pd.NA
        
        result = _handle_missing_values(df)
        
        assert len(result) < len(df)


class TestAddDaytimeFlag:
    """Test daytime flag addition."""
    
    def test_adds_is_daytime_column(self, sample_df: pd.DataFrame):
        """Add is_daytime column."""
        result = _add_daytime_flag(sample_df)
        
        assert "is_daytime" in result.columns
    
    def test_nighttime_has_false_flag(self, sample_df: pd.DataFrame):
        """Nighttime rows should have is_daytime=False."""
        result = _add_daytime_flag(sample_df)
        
        # Find nighttime row (irradiance < 5)
        nighttime = result[result["irradiance"] < 5]
        assert (nighttime["is_daytime"] == False).all()


class TestSelectFeatures:
    """Test feature selection."""
    
    def test_selects_available_features(self, sample_df: pd.DataFrame):
        """Select only features that are present."""
        result = _select_features(sample_df)
        
        # Should have timestamp, power, and available features
        assert "timestamp" in result.columns
        assert "power" in result.columns
        assert "irradiance" in result.columns
    
    def test_includes_dc_power_if_present(self, sample_df: pd.DataFrame):
        """Include dc_power if present."""
        result = _select_features(sample_df)
        
        if "dc_power" in sample_df.columns:
            assert "dc_power" in result.columns
