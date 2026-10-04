"""
src/config.py
=============
Centralised configuration loaded from environment variables / .env file.

All other modules import from here — no hard-coded paths or parameters
anywhere else in the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR       = PROJECT_ROOT / os.getenv("DATA_RAW_DIR",       "data/raw")
PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
MODELS_DIR    = PROJECT_ROOT / os.getenv("MODELS_DIR",         "models")

# Ensure directories exist at import time
for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Column mapping: raw PVDAQ names → application names
# ---------------------------------------------------------------------------
# Used by both the download script and data loader — single source of truth
COLUMN_MAP: dict[str, str] = {
    "measdatetime":   "timestamp",
    "ac_power":       "power",               # TARGET — AC power output in kW
    "poa_irradiance": "irradiance",          # Plane-of-array irradiance, W/m²
    "ambient_temp":   "ambient_temperature", # °C
    "module_temp":    "panel_temperature",   # °C
    "wind_speed":     "wind_speed",          # m/s
    "dc_power":       "dc_power",            # kW (auxiliary)
    # Alternate spellings seen in some PVDAQ exports:
    "ac_power_kw":    "power",
    "poa":            "irradiance",
    "t_amb":          "ambient_temperature",
    "t_mod":          "panel_temperature",
    "ws":             "wind_speed",
}

# Minimum required columns (application names) for the ML pipeline
REQUIRED_COLUMNS: set[str] = {"timestamp", "power", "irradiance"}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
PVDAQ_SITE_ID    = int(os.getenv("PVDAQ_SITE_ID",    "4"))
PVDAQ_START_YEAR = int(os.getenv("PVDAQ_START_YEAR", "2011"))
PVDAQ_END_YEAR   = int(os.getenv("PVDAQ_END_YEAR",   "2013"))

# Raw merged CSV (produced by download_data.py)
RAW_MERGED_CSV = RAW_DIR.parent / "processed" / f"pvdaq_site{PVDAQ_SITE_ID}_merged_raw.csv"
# Clean processed CSV (produced by preprocessing.py)
CLEAN_CSV      = PROCESSED_DIR / f"pvdaq_site{PVDAQ_SITE_ID}_clean.csv"

# ---------------------------------------------------------------------------
# Column names (application-level)
# ---------------------------------------------------------------------------
TIMESTAMP_COL = "timestamp"
TARGET_COL    = "power"            # AC power output in kW — regression target

# All possible feature columns (the pipeline uses whichever are actually present)
FEATURE_COLS_ORDERED = [
    "irradiance",           # Plane-of-array irradiance, W/m²  (primary driver)
    "ambient_temperature",  # Ambient air temperature, °C
    "panel_temperature",    # Module/panel temperature, °C
    "wind_speed",           # Wind speed, m/s
]

# ---------------------------------------------------------------------------
# Physical validity ranges (used for data cleaning AND API validation)
# ---------------------------------------------------------------------------
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "irradiance":           (0.0,   1500.0),   # W/m²  (theoretical max ~1361 W/m²)
    "power":                (-0.1,  500.0),    # kW    (small negative ok = inverter standby)
    "dc_power":             (-0.1,  500.0),    # kW
    "ambient_temperature":  (-50.0, 60.0),     # °C    (survivable sensor range)
    "panel_temperature":    (-50.0, 100.0),    # °C    (panels can get very hot)
    "wind_speed":           (0.0,   80.0),     # m/s   (hurricane-force = ~90 m/s)
}

# Irradiance threshold below which a row is considered "nighttime"
NIGHTTIME_IRRADIANCE_THRESHOLD = 5.0   # W/m²

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
# Time-based split — first TRAIN_FRACTION of sorted rows → training
TRAIN_FRACTION = 0.80

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_VERSION     = os.getenv("MODEL_VERSION", "1.0")
MODEL_PATH        = MODELS_DIR / "solar_power_model.joblib"
MODEL_METADATA    = MODELS_DIR / "model_metadata.json"
BEST_MODEL_NAME   = "XGBoost"   # updated after evaluation

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
SIMULATE_INTERVAL = float(os.getenv("SIMULATE_INTERVAL_SECONDS", "2"))
SIMULATE_MAX_ROWS = int(os.getenv("SIMULATE_MAX_ROWS", "0"))
