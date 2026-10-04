"""
tests/test_model.py
==================
Tests for model training and prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_METADATA, MODEL_PATH
from src.predict import Predictor, reset_predictor


class TestPredictor:
    """Test Predictor class."""
    
    def test_loads_model(self):
        """Test that predictor can load the model."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        predictor = Predictor()
        
        assert predictor.model is not None
        assert predictor.metadata is not None
        assert predictor.features is not None
    
    def test_predict_single(self):
        """Test single prediction."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        predictor = Predictor()
        
        features = {
            "irradiance": 850.0,
            "ambient_temperature": 25.0,
            "panel_temperature": 43.0,
            "wind_speed": 2.5,
        }
        
        result = predictor.predict(features)
        
        assert "predicted_power_kw" in result
        assert result["predicted_power_kw"] >= 0
        assert "model_name" in result
        assert "model_version" in result
    
    def test_predict_missing_features(self):
        """Test prediction with missing features raises error."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        predictor = Predictor()
        
        features = {
            "irradiance": 850.0,
            # Missing other features
        }
        
        with pytest.raises(ValueError, match="Missing required features"):
            predictor.predict(features)
    
    def test_predict_batch(self):
        """Test batch prediction."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        predictor = Predictor()
        
        features_list = [
            {
                "irradiance": 850.0,
                "ambient_temperature": 25.0,
                "panel_temperature": 43.0,
                "wind_speed": 2.5,
            },
            {
                "irradiance": 0.0,
                "ambient_temperature": 15.0,
                "panel_temperature": 15.0,
                "wind_speed": 1.0,
            },
        ]
        
        results = predictor.predict_batch(features_list)
        
        assert len(results) == 2
        assert all("predicted_power_kw" in r for r in results)
    
    def test_get_model_info(self):
        """Test getting model information."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        predictor = Predictor()
        
        info = predictor.get_model_info()
        
        assert "model_name" in info
        assert "model_version" in info
        assert "features" in info
        assert "metrics" in info
    
    def test_singleton_predictor(self):
        """Test singleton predictor instance."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        from src.predict import get_predictor, reset_predictor
        
        reset_predictor()
        
        p1 = get_predictor()
        p2 = get_predictor()
        
        assert p1 is p2  # Same instance


class TestModelMetadata:
    """Test model metadata file."""
    
    def test_metadata_exists(self):
        """Test that metadata file exists after training."""
        if not MODEL_METADATA.exists():
            pytest.skip("Metadata file not found - run training first")
        
        assert MODEL_METADATA.exists()
    
    def test_metadata_structure(self):
        """Test that metadata has correct structure."""
        if not MODEL_METADATA.exists():
            pytest.skip("Metadata file not found - run training first")
        
        with open(MODEL_METADATA, "r") as f:
            metadata = json.load(f)
        
        required_keys = [
            "model_name",
            "model_version",
            "features",
            "metrics",
            "train_samples",
            "test_samples",
        ]
        
        for key in required_keys:
            assert key in metadata, f"Missing key: {key}"
    
    def test_metrics_are_numbers(self):
        """Test that metrics are numeric values."""
        if not MODEL_METADATA.exists():
            pytest.skip("Metadata file not found - run training first")
        
        with open(MODEL_METADATA, "r") as f:
            metadata = json.load(f)
        
        metrics = metadata["metrics"]
        
        assert isinstance(metrics["mae"], (int, float))
        assert isinstance(metrics["rmse"], (int, float))
        assert isinstance(metrics["r2"], (int, float))
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
        assert 0 <= metrics["r2"] <= 1
