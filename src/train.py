"""
src/train.py
============
Train and compare multiple regression models for solar PV power prediction.

Models trained:
1. Linear Regression (baseline)
2. Random Forest (non-linear baseline)
3. XGBoost (gradient boosting, expected best performer)

All models use a time-aware train/test split to prevent data leakage.
The best model is saved as a joblib file for use in the API.

Usage
-----
    python -m src.train
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import MODEL_METADATA, MODEL_PATH, MODEL_VERSION
from src.data_loader import load_clean
from src.features import get_available_features, split_features_target, time_based_train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def create_linear_regression() -> Pipeline:
    """
    Create a Linear Regression model with StandardScaler.
    
    Linear regression is a good baseline — it assumes a linear relationship
    between irradiance/temperature and power output.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def create_random_forest() -> RandomForestRegressor:
    """
    Create a Random Forest regressor.
    
    Random Forest can capture non-linear relationships and interactions
    between features (e.g., irradiance × temperature effects).
    """
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )


def create_xgboost() -> XGBRegressor:
    """
    Create an XGBoost regressor.
    
    XGBoost (gradient boosting) typically performs best on tabular data
    with non-linear relationships. It handles the irradiance-power relationship
    and temperature effects well.
    """
    return XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> Any:
    """
    Train a model on the training data.
    
    Parameters
    ----------
    model : Scikit-learn or XGBoost model
    X_train : Training features
    y_train : Training target
    model_name : Name for logging
    
    Returns
    -------
    Trained model
    """
    log.info("Training %s...", model_name)
    model.fit(X_train, y_train)
    log.info("  %s training complete", model_name)
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, float]:
    """
    Evaluate a model on test data.
    
    Parameters
    ----------
    model : Trained model
    X_test : Test features
    y_test : Test target
    model_name : Name for logging
    
    Returns
    -------
    Dictionary of metrics: mae, rmse, r2
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    
    # R² score
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    
    log.info("  %s metrics: MAE=%.4f, RMSE=%.4f, R²=%.4f", model_name, mae, rmse, r2)
    
    return metrics


def main() -> int:
    """Train all models and save the best one."""
    log.info("=" * 60)
    log.info("Model Training Pipeline")
    log.info("=" * 60)
    
    # Load cleaned data
    df = load_clean()
    log.info("Loaded cleaned data: %d rows × %d columns", len(df), len(df.columns))
    
    # Time-based train/test split
    train_df, test_df = time_based_train_test_split(df)
    
    # Get available features
    feature_cols = get_available_features(train_df)
    
    # Split features and target
    X_train, y_train = split_features_target(train_df, feature_cols)
    X_test, y_test = split_features_target(test_df, feature_cols)
    
    log.info("")
    log.info("Training models...")
    log.info("")
    
    # Train Linear Regression
    lr_model = create_linear_regression()
    lr_model = train_model(lr_model, X_train, y_train, "Linear Regression")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    
    # Train Random Forest
    rf_model = create_random_forest()
    rf_model = train_model(rf_model, X_train, y_train, "Random Forest")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Train XGBoost
    xgb_model = create_xgboost()
    xgb_model = train_model(xgb_model, X_train, y_train, "XGBoost")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    # Compare models
    log.info("")
    log.info("Model Comparison:")
    log.info("-" * 60)
    log.info("{:<20} {:>10} {:>10} {:>10}".format("Model", "MAE", "RMSE", "R²"))
    log.info("-" * 60)
    log.info("{:<20} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        "Linear Regression", lr_metrics["mae"], lr_metrics["rmse"], lr_metrics["r2"]
    ))
    log.info("{:<20} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        "Random Forest", rf_metrics["mae"], rf_metrics["rmse"], rf_metrics["r2"]
    ))
    log.info("{:<20} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        "XGBoost", xgb_metrics["mae"], xgb_metrics["rmse"], xgb_metrics["r2"]
    ))
    log.info("-" * 60)
    
    # Select best model (lowest RMSE)
    models = {
        "Linear Regression": (lr_model, lr_metrics),
        "Random Forest": (rf_model, rf_metrics),
        "XGBoost": (xgb_model, xgb_metrics),
    }
    
    best_model_name = min(models.keys(), key=lambda k: models[k][1]["rmse"])
    best_model, best_metrics = models[best_model_name]
    
    log.info("")
    log.info("Best model: %s (RMSE=%.4f)", best_model_name, best_metrics["rmse"])
    
    # Save best model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    log.info("Saved model: %s", MODEL_PATH)
    
    # Save model metadata
    metadata = {
        "model_name": best_model_name,
        "model_version": MODEL_VERSION,
        "features": feature_cols,
        "metrics": best_metrics,
        "all_metrics": {
            "Linear Regression": lr_metrics,
            "Random Forest": rf_metrics,
            "XGBoost": xgb_metrics,
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_date_range": {
            "start": str(train_df["timestamp"].min()),
            "end": str(train_df["timestamp"].max()),
        },
        "test_date_range": {
            "start": str(test_df["timestamp"].min()),
            "end": str(test_df["timestamp"].max()),
        },
    }
    
    with open(MODEL_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved metadata: %s", MODEL_METADATA)
    
    log.info("")
    log.info("Next step:")
    log.info("  python -m src.evaluate")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
