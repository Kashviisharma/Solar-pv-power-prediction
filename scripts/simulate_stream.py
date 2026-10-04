#!/usr/bin/env python3
"""
scripts/simulate_stream.py
===========================
Simulate live sensor data by streaming historical dataset rows to the API.

This script:
1. Loads the cleaned historical dataset
2. Iterates through rows sequentially
3. Sends each row's environmental features to the /predict endpoint
4. Stores timestamp, inputs, actual power, predicted power, and error
5. Saves results to a time-series CSV file for Grafana

This simulates a real deployment where sensors send readings to the API
at regular intervals.

Usage
-----
    python scripts/simulate_stream.py
    
    # With custom interval (seconds between readings)
    python scripts/simulate_stream.py --interval 5
    
    # Limit number of rows
    python scripts/simulate_stream.py --max-rows 1000
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import SIMULATE_INTERVAL, SIMULATE_MAX_ROWS
from src.data_loader import load_clean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# API configuration
API_URL = "http://localhost:8000/predict"
TIME_SERIES_OUTPUT = Path(__file__).parent.parent / "data" / "processed" / "predictions_time_series.csv"


def send_prediction(features: dict[str, float]) -> dict[str, Any]:
    """
    Send a prediction request to the API.
    
    Parameters
    ----------
    features : Dictionary of environmental features
    
    Returns
    -------
    API response as a dictionary
    """
    try:
        response = requests.post(API_URL, json=features, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        log.error("Could not connect to API at %s. Is the API running?", API_URL)
        raise
    except requests.exceptions.Timeout:
        log.error("API request timed out")
        raise
    except requests.exceptions.HTTPError as e:
        log.error("API returned error: %s", e)
        raise


def simulate_stream(
    interval_seconds: float = SIMULATE_INTERVAL,
    max_rows: int = SIMULATE_MAX_ROWS,
) -> None:
    """
    Simulate live sensor data streaming.
    
    Parameters
    ----------
    interval_seconds : Time to wait between simulated readings
    max_rows : Maximum number of rows to process (0 = all)
    """
    log.info("=" * 60)
    log.info("Historical Data Stream Simulation")
    log.info("=" * 60)
    log.info("API URL: %s", API_URL)
    log.info("Interval: %.1f seconds", interval_seconds)
    log.info("Max rows: %s", "all" if max_rows == 0 else max_rows)
    log.info("")
    
    # Load cleaned data
    df = load_clean()
    log.info("Loaded dataset: %d rows", len(df))
    
    # Check API is available
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        health_response.raise_for_status()
        log.info("API health check: OK")
    except Exception as e:
        log.error("API health check failed: %s", e)
        log.error("Please start the API first:")
        log.error("  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Prepare time-series storage
    time_series_data = []
    
    # Feature columns to send to API
    feature_cols = ["irradiance", "ambient_temperature", "panel_temperature", "wind_speed"]
    
    # Check all feature columns exist
    missing = set(feature_cols) - set(df.columns)
    if missing:
        log.error("Missing feature columns in dataset: %s", missing)
        return
    
    # Limit rows if specified
    if max_rows > 0:
        df = df.head(max_rows)
        log.info("Processing first %d rows", max_rows)
    
    log.info("")
    log.info("Starting simulation...")
    log.info("-" * 60)
    
    start_time = time.time()
    success_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        # Extract features
        features = {
            "irradiance": float(row["irradiance"]),
            "ambient_temperature": float(row["ambient_temperature"]),
            "panel_temperature": float(row["panel_temperature"]),
            "wind_speed": float(row["wind_speed"]),
        }
        
        # Get actual power
        actual_power = float(row["power"])
        timestamp = row["timestamp"]
        
        try:
            # Send prediction request
            result = send_prediction(features)
            predicted_power = result["predicted_power_kw"]
            
            # Calculate error
            prediction_error = actual_power - predicted_power
            
            # Store result
            time_series_data.append({
                "timestamp": timestamp,
                "irradiance": features["irradiance"],
                "ambient_temperature": features["ambient_temperature"],
                "panel_temperature": features["panel_temperature"],
                "wind_speed": features["wind_speed"],
                "actual_power_kw": actual_power,
                "predicted_power_kw": predicted_power,
                "prediction_error_kw": prediction_error,
            })
            
            success_count += 1
            
            # Log progress periodically
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                log.info("Processed %d/%d rows (%.1f rows/sec)", idx + 1, len(df), rate)
            
        except Exception as e:
            error_count += 1
            log.error("Error processing row %d: %s", idx, e)
        
        # Wait before next reading (except for last row)
        if idx < len(df) - 1:
            time.sleep(interval_seconds)
    
    # Save time-series data
    if time_series_data:
        ts_df = pd.DataFrame(time_series_data)
        ts_df.to_csv(TIME_SERIES_OUTPUT, index=False)
        log.info("")
        log.info("Saved time-series data: %s (%d rows)", TIME_SERIES_OUTPUT, len(ts_df))
    
    # Print summary
    elapsed = time.time() - start_time
    log.info("")
    log.info("Simulation complete")
    log.info("-" * 60)
    log.info("Total rows processed: %d", len(df))
    log.info("Successful predictions: %d", success_count)
    log.info("Errors: %d", error_count)
    log.info("Total time: %.1f seconds", elapsed)
    log.info("Average rate: %.1f rows/sec", len(df) / elapsed if elapsed > 0 else 0)
    
    # Print error statistics
    if time_series_data:
        ts_df = pd.DataFrame(time_series_data)
        mae = ts_df["prediction_error_kw"].abs().mean()
        rmse = (ts_df["prediction_error_kw"] ** 2).mean() ** 0.5
        log.info("Simulation MAE: %.4f kW", mae)
        log.info("Simulation RMSE: %.4f kW", rmse)


def main() -> int:
    """Main entry point."""
    global API_URL
    
    parser = argparse.ArgumentParser(
        description="Simulate live sensor data streaming to the API"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=SIMULATE_INTERVAL,
        help=f"Interval between readings in seconds (default: {SIMULATE_INTERVAL})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=SIMULATE_MAX_ROWS,
        help=f"Maximum number of rows to process (0 = all, default: {SIMULATE_MAX_ROWS})",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=API_URL,
        help=f"API URL (default: {API_URL})",
    )
    
    args = parser.parse_args()
    
    # Update global API URL if specified
    API_URL = args.api_url
    
    try:
        simulate_stream(args.interval, args.max_rows)
        return 0
    except KeyboardInterrupt:
        log.info("\nSimulation interrupted by user")
        return 0
    except Exception as e:
        log.error("Simulation failed: %s", e)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
