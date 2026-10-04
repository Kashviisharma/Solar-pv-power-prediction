"""
api/schemas.py
==============
Pydantic schemas for request/response validation in the FastAPI application.

These schemas validate input data against physical bounds derived from the dataset.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, confloat

from src.config import PHYSICAL_BOUNDS


class PredictionRequest(BaseModel):
    """
    Request schema for the /predict endpoint.
    
    All fields are validated against physical bounds from the dataset.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "irradiance": 850.0,
                "ambient_temperature": 25.0,
                "panel_temperature": 43.0,
                "wind_speed": 2.5,
            }
        }
    )
    
    irradiance: confloat(ge=0, le=1500) = Field(
        ...,
        description="Plane-of-array solar irradiance in W/m²",
    )
    ambient_temperature: confloat(ge=-50, le=60) = Field(
        ...,
        description="Ambient air temperature in °C",
    )
    panel_temperature: confloat(ge=-50, le=100) = Field(
        ...,
        description="PV panel/module temperature in °C",
    )
    wind_speed: confloat(ge=0, le=80) = Field(
        ...,
        description="Wind speed in m/s",
    )


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""
    predicted_power_kw: float = Field(
        ...,
        description="Predicted AC power output in kW",
    )
    model_name: str = Field(
        ...,
        description="Name of the model used for prediction",
    )
    model_version: str = Field(
        ...,
        description="Model version",
    )
    features_used: list[str] = Field(
        ...,
        description="List of features used by the model",
    )


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    """Response schema for the /model/info endpoint."""
    model_name: str
    model_version: str
    features: list[str]
    metrics: dict[str, float]
    train_samples: int
    test_samples: int
    train_date_range: dict[str, str]
    test_date_range: dict[str, str]


class MetricsResponse(BaseModel):
    """Response schema for the /metrics endpoint."""
    model_name: str
    model_version: str
    metrics: dict[str, float]
    all_metrics: dict[str, dict[str, float]] = Field(
        ...,
        description="Metrics for all trained models (Linear Regression, Random Forest, XGBoost)",
    )


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Any = Field(None, description="Additional error details")
