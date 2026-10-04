"""
src/predict.py
==============
Prediction module for loading the trained model and making predictions.

This module provides a simple interface for:
- Loading the saved model
- Making predictions on new data
- Handling feature order and validation

Usage
-----
    from src.predict import Predictor
    
    predictor = Predictor()
    result = predictor.predict({
        "irradiance": 850.0,
        "ambient_temperature": 25.0,
        "panel_temperature": 43.0,
        "wind_speed": 2.5,
    })
    print(result)  # {"predicted_power_kw": 1.52, ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_METADATA, MODEL_PATH

log = logging.getLogger(__name__)


class Predictor:
    """
    Predictor class for loading model and making predictions.
    
    The model is loaded once at initialization and reused for all predictions.
    """
    
    def __init__(self, model_path: Path | None = None, metadata_path: Path | None = None):
        """
        Initialize the predictor by loading the model and metadata.
        
        Parameters
        ----------
        model_path : Optional path to model file. Defaults to config.MODEL_PATH.
        metadata_path : Optional path to metadata file. Defaults to config.MODEL_METADATA.
        """
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else MODEL_METADATA
        
        self.model = None
        self.metadata = None
        self.features = None
        
        self._load()
    
    def _load(self) -> None:
        """Load the model and metadata from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        log.info("Loading model from: %s", self.model_path)
        self.model = joblib.load(self.model_path)
        
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        self.features = self.metadata["features"]
        
        log.info("Loaded model: %s (version %s)", self.metadata["model_name"], self.metadata["model_version"])
        log.info("Features: %s", self.features)
    
    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """
        Make a prediction for a single set of features.
        
        Parameters
        ----------
        features : Dictionary of feature names to values
        
        Returns
        -------
        Dictionary with predicted_power_kw and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load() first.")
        
        # Validate that all required features are present
        missing = set(self.features) - set(features.keys())
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Ensure features are in the correct order
        feature_values = [features[f] for f in self.features]
        
        # Create DataFrame for prediction
        X = pd.DataFrame([feature_values], columns=self.features)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Ensure prediction is non-negative
        prediction = max(0.0, float(prediction))
        
        return {
            "predicted_power_kw": round(prediction, 4),
            "model_name": self.metadata["model_name"],
            "model_version": self.metadata["model_version"],
            "features_used": self.features,
        }
    
    def predict_batch(self, features_list: list[dict[str, float]]) -> list[dict[str, Any]]:
        """
        Make predictions for multiple sets of features.
        
        Parameters
        ----------
        features_list : List of feature dictionaries
        
        Returns
        -------
        List of prediction dictionaries
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load() first.")
        
        # Validate each feature dict
        for i, features in enumerate(features_list):
            missing = set(self.features) - set(features.keys())
            if missing:
                raise ValueError(f"Row {i}: Missing required features: {missing}")
        
        # Create DataFrame
        X = pd.DataFrame(features_list)[self.features]
        
        # Make predictions
        predictions = self.model.predict(X)
        predictions = np.maximum(0, predictions)  # Ensure non-negative
        
        results = []
        for pred in predictions:
            results.append({
                "predicted_power_kw": round(float(pred), 4),
                "model_name": self.metadata["model_name"],
                "model_version": self.metadata["model_version"],
            })
        
        return results
    
    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        if self.metadata is None:
            raise RuntimeError("Model not loaded. Call _load() first.")
        
        return {
            "model_name": self.metadata["model_name"],
            "model_version": self.metadata["model_version"],
            "features": self.metadata["features"],
            "metrics": self.metadata["metrics"],
            "train_samples": self.metadata["train_samples"],
            "test_samples": self.metadata["test_samples"],
            "train_date_range": self.metadata.get("train_date_range", {}),
            "test_date_range": self.metadata.get("test_date_range", {}),
        }


# Singleton instance for convenience
_predictor_instance: Predictor | None = None


def get_predictor() -> Predictor:
    """
    Get a singleton Predictor instance.
    
    This is useful for FastAPI to load the model once at startup.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = Predictor()
    return _predictor_instance


def reset_predictor() -> None:
    """Reset the singleton predictor (useful for testing)."""
    global _predictor_instance
    _predictor_instance = None
