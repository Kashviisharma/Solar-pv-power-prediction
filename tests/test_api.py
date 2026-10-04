"""
tests/test_api.py
=================
Tests for FastAPI application.

These tests use pytest's httpx async client to test the API endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import MODEL_PATH
from src.predict import Predictor


@pytest.fixture
def client():
    """Create a test client for the FastAPI app with loaded predictor."""
    # Load predictor and set it on the app
    try:
        predictor = Predictor()
        # Monkey-patch the global predictor in the app's module
        import api.main
        api.main.predictor = predictor
    except Exception:
        pass
    return TestClient(app)


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_returns_info(self, client):
        """Root endpoint should return API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_returns_status(self, client):
        """Health endpoint should return status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
    
    def test_health_without_model(self, client):
        """Health endpoint should report unhealthy if model not loaded."""
        # Temporarily unload model
        import api.main
        original_predictor = api.main.predictor
        api.main.predictor = None
        
        try:
            response = client.get("/health")
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False
        finally:
            api.main.predictor = original_predictor


class TestModelInfoEndpoint:
    """Test model info endpoint."""
    
    def test_model_info_returns_metadata(self, client):
        """Model info endpoint should return model metadata."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_name" in data
        assert "model_version" in data
        assert "features" in data
        assert "metrics" in data
    
    def test_model_info_without_model(self, client):
        """Model info endpoint should return 503 if model not loaded."""
        import api.main
        original_predictor = api.main.predictor
        api.main.predictor = None
        
        try:
            response = client.get("/model/info")
            
            assert response.status_code == 503
        finally:
            api.main.predictor = original_predictor


class TestMetricsEndpoint:
    """Test metrics endpoint."""
    
    def test_metrics_returns_evaluation_metrics(self, client):
        """Metrics endpoint should return evaluation metrics."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_name" in data
        assert "model_version" in data
        assert "metrics" in data
        assert "all_metrics" in data
    
    def test_metrics_includes_mae_rmse_r2(self, client):
        """Metrics should include MAE, RMSE, and R²."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        response = client.get("/metrics")
        data = response.json()
        
        metrics = data["metrics"]
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics


class TestPredictEndpoint:
    """Test prediction endpoint."""
    
    def test_predict_with_valid_data(self, client):
        """Predict endpoint should return prediction for valid data."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        request_data = {
            "irradiance": 850.0,
            "ambient_temperature": 25.0,
            "panel_temperature": 43.0,
            "wind_speed": 2.5,
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predicted_power_kw" in data
        assert data["predicted_power_kw"] >= 0
        assert "model_name" in data
        assert "model_version" in data
    
    def test_predict_with_missing_field(self, client):
        """Predict endpoint should return 400 for missing field."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        request_data = {
            "irradiance": 850.0,
            # Missing other fields
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_invalid_irradiance(self, client):
        """Predict endpoint should reject irradiance outside bounds."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        request_data = {
            "irradiance": 2000.0,  # Above max of 1500
            "ambient_temperature": 25.0,
            "panel_temperature": 43.0,
            "wind_speed": 2.5,
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_negative_irradiance(self, client):
        """Predict endpoint should reject negative irradiance."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        request_data = {
            "irradiance": -10.0,
            "ambient_temperature": 25.0,
            "panel_temperature": 43.0,
            "wind_speed": 2.5,
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_without_model(self, client):
        """Predict endpoint should return 503 if model not loaded."""
        import api.main
        original_predictor = api.main.predictor
        api.main.predictor = None
        
        try:
            request_data = {
                "irradiance": 850.0,
                "ambient_temperature": 25.0,
                "panel_temperature": 43.0,
                "wind_speed": 2.5,
            }
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 503
        finally:
            api.main.predictor = original_predictor
    
    def test_predict_nighttime_returns_near_zero(self, client):
        """Predict endpoint should return near-zero power for nighttime conditions."""
        if not MODEL_PATH.exists():
            pytest.skip("Model file not found - run training first")
        
        request_data = {
            "irradiance": 0.0,  # Nighttime
            "ambient_temperature": 15.0,
            "panel_temperature": 15.0,
            "wind_speed": 1.0,
        }
        
        response = client.post("/predict", json=request_data)
        data = response.json()
        
        # Should be very close to zero
        assert data["predicted_power_kw"] < 0.1


class TestPydanticValidation:
    """Test Pydantic schema validation."""
    
    def test_prediction_request_validation(self):
        """Test that PredictionRequest validates input correctly."""
        from api.schemas import PredictionRequest
        
        # Valid data
        valid_data = {
            "irradiance": 850.0,
            "ambient_temperature": 25.0,
            "panel_temperature": 43.0,
            "wind_speed": 2.5,
        }
        request = PredictionRequest(**valid_data)
        assert request.irradiance == 850.0
        
        # Invalid irradiance (too high)
        with pytest.raises(Exception):
            PredictionRequest(
                irradiance=2000.0,
                ambient_temperature=25.0,
                panel_temperature=43.0,
                wind_speed=2.5,
            )
        
        # Negative irradiance
        with pytest.raises(Exception):
            PredictionRequest(
                irradiance=-10.0,
                ambient_temperature=25.0,
                panel_temperature=43.0,
                wind_speed=2.5,
            )
