"""
src/evaluate.py
================
Detailed evaluation and visualization of model performance.

This module loads the saved model and metadata, evaluates on test data,
and generates visualizations of predictions vs actual values.

Usage
-----
    python -m src.evaluate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import MODEL_METADATA, MODEL_PATH
from src.data_loader import load_clean
from src.features import get_available_features, split_features_target, time_based_train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_model_and_metadata():
    """Load the saved model and its metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    if not MODEL_METADATA.exists():
        raise FileNotFoundError(f"Metadata file not found: {MODEL_METADATA}")
    
    model = joblib.load(MODEL_PATH)
    
    with open(MODEL_METADATA, "r") as f:
        metadata = json.load(f)
    
    log.info("Loaded model: %s", metadata["model_name"])
    log.info("Model version: %s", metadata["model_version"])
    log.info("Features: %s", metadata["features"])
    
    return model, metadata


def evaluate_on_test_data(model, metadata) -> dict[str, float]:
    """Evaluate the model on the test set."""
    df = load_clean()
    
    # Split train/test using the same fraction as during training
    train_df, test_df = time_based_train_test_split(df)
    
    # Get features from metadata
    feature_cols = metadata["features"]
    
    # Split features and target
    X_test, y_test = split_features_target(test_df, feature_cols)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    
    log.info("Test set evaluation:")
    log.info("  MAE:  %.4f kW", mae)
    log.info("  RMSE: %.4f kW", rmse)
    log.info("  R²:   %.4f", r2)
    
    return metrics, y_test, y_pred, test_df


def plot_predictions(y_test, y_pred, output_dir: Path):
    """Generate prediction vs actual plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot 1: Actual vs Predicted scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Power (kW)")
    ax.set_ylabel("Predicted Power (kW)")
    ax.set_title("Actual vs Predicted PV Power")
    ax.grid(True, alpha=0.3)
    
    scatter_path = output_dir / "actual_vs_predicted.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved scatter plot: %s", scatter_path)
    
    # Plot 2: Residuals histogram
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel("Prediction Error (kW)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Prediction Errors")
    ax.grid(True, alpha=0.3)
    
    resid_path = output_dir / "residuals_histogram.png"
    plt.savefig(resid_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved residuals plot: %s", resid_path)
    
    # Plot 3: Time series (sample of test data)
    fig, ax = plt.subplots(figsize=(14, 6))
    sample_size = min(500, len(y_test))
    ax.plot(y_test.iloc[:sample_size].values, label='Actual', alpha=0.7)
    ax.plot(y_pred[:sample_size], label='Predicted', alpha=0.7)
    ax.set_xlabel("Time Step (sample)")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Actual vs Predicted Power Over Time (Sample)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ts_path = output_dir / "time_series_sample.png"
    plt.savefig(ts_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved time series plot: %s", ts_path)


def main() -> int:
    """Run evaluation and generate plots."""
    log.info("=" * 60)
    log.info("Model Evaluation")
    log.info("=" * 60)
    
    # Load model
    model, metadata = load_model_and_metadata()
    
    # Evaluate
    metrics, y_test, y_pred, test_df = evaluate_on_test_data(model, metadata)
    
    # Generate plots
    plots_dir = Path(__file__).parent.parent / "models" / "plots"
    plot_predictions(y_test, y_pred, plots_dir)
    
    # Print comparison with training metrics
    log.info("")
    log.info("Comparison with training metrics:")
    log.info("-" * 60)
    log.info("{:<15} {:>10} {:>10}".format("Metric", "Training", "Current"))
    log.info("-" * 60)
    train_metrics = metadata["metrics"]
    log.info("{:<15} {:>10.4f} {:>10.4f}".format("MAE", train_metrics["mae"], metrics["mae"]))
    log.info("{:<15} {:>10.4f} {:>10.4f}".format("RMSE", train_metrics["rmse"], metrics["rmse"]))
    log.info("{:<15} {:>10.4f} {:>10.4f}".format("R²", train_metrics["r2"], metrics["r2"]))
    log.info("-" * 60)
    
    log.info("")
    log.info("Evaluation complete. Plots saved to: %s", plots_dir)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
