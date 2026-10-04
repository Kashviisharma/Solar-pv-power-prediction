"""
api/main.py
===========
FastAPI application for solar PV power prediction.

Endpoints:
- GET /health          : Health check
- GET /model/info      : Model information
- GET /metrics         : Model evaluation metrics
- POST /predict        : Make a prediction
- GET /timeseries      : Historical predictions CSV as JSON (for React charts)

The model is loaded once at startup and reused for all requests.

Usage
-----
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.predict import Predictor, reset_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Global predictor instance
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    Load the model on startup and clean up on shutdown.
    """
    global predictor
    
    log.info("=" * 60)
    log.info("Starting Solar PV Power Prediction API")
    log.info("=" * 60)
    
    # Load model at startup
    try:
        predictor = Predictor()
        log.info("Model loaded successfully")
    except Exception as e:
        log.error("Failed to load model: %s", e)
        # Don't raise — allow app to start but predictions will fail
        predictor = None
    
    yield
    
    # Cleanup on shutdown
    log.info("Shutting down API")
    reset_predictor()


# Create FastAPI app
app = FastAPI(
    title="Solar PV Power Prediction API",
    description="REST API for predicting solar PV power output from environmental conditions",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Streamlit frontend (and any browser client) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/timeseries")
async def timeseries(limit: int = Query(default=1000, le=5000)):
    """
    Return the last `limit` rows of the predictions time-series CSV as JSON.
    Used by the React dashboard to populate charts.
    """
    ts_file = Path("data/processed/predictions_time_series.csv")
    if not ts_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Time-series data not found. Run: python -m scripts.simulate_stream --max-rows 1000",
        )

    df = pd.read_csv(ts_file)          # read as plain strings first
    df.sort_values("timestamp", inplace=True)
    df = df.tail(limit).reset_index(drop=True)

    # Normalise timestamp to a plain HH:MM label the charts can use
    # Handles both "2012-07-01T06:00:00" and "2012-07-01 06:00:00+00:00"
    def _fmt(ts_str: str) -> str:
        ts_str = str(ts_str).split("+")[0].replace(" ", "T")
        return ts_str  # keep full ISO — chart XAxis slices [:16] for display

    df["timestamp"] = df["timestamp"].apply(_fmt)

    # Round floats to 4 dp
    float_cols = ["irradiance", "ambient_temperature", "panel_temperature",
                  "wind_speed", "actual_power_kw", "predicted_power_kw", "prediction_error_kw"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4).fillna(0)

    return df.to_dict(orient="records")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Solar PV Power Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "metrics": "/metrics",
            "predict": "/predict",
        },
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns the API status and whether the model is loaded.
    """
    if predictor is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=None,
            model_version=None,
        )
    
    info = predictor.get_model_info()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=info["model_name"],
        model_version=info["model_version"],
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get model information.
    
    Returns model name, version, features, and training metadata.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return predictor.get_model_info()


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """
    Get model evaluation metrics.
    
    Returns MAE, RMSE, and R² for the best model and all trained models.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = predictor.get_model_info()
    return MetricsResponse(
        model_name=info["model_name"],
        model_version=info["model_version"],
        metrics=info["metrics"],
        all_metrics=info.get("all_metrics", {}),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for given environmental conditions.
    
    Accepts irradiance, temperatures, and wind speed.
    Returns predicted AC power output in kW.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic model to dict
        features = request.model_dump()
        
        # Make prediction
        result = predictor.predict(features)
        
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "message": str(exc)},
    )
