#!/usr/bin/env python3
"""
scripts/generate_synthetic_data.py
===================================
Generate a realistic synthetic solar PV dataset based on NREL PVDAQ Site 4 characteristics.

This script creates data that matches the structure and physical characteristics
described in DATASET.md for NREL PVDAQ Site 4 (Golden, Colorado).

The synthetic data is generated using:
- Realistic solar irradiance curves (bell-shaped during day, zero at night)
- Temperature relationships (panel temp correlates with irradiance)
- Power output modeled as a function of irradiance and temperature
- Measurement noise to simulate real sensor data
- Missing values to simulate real-world data quality issues

This is used when the NREL API or OEDI mirror is not accessible.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"

# Site characteristics based on NREL PVDAQ Site 4 (Golden, CO)
SITE_CAPACITY_KW = 2.0  # ~2 kW DC system
LATITUDE = 39.74  # Golden, CO
START_DATE = datetime(2012, 1, 1, 0, 0, 0)
END_DATE = datetime(2012, 12, 31, 23, 45, 0)
INTERVAL_MINUTES = 15  # 15-minute intervals

# ---------------------------------------------------------------------------
# Solar physics functions
# ---------------------------------------------------------------------------


def _day_of_year(dt: datetime) -> int:
    """Return day of year (1-366)."""
    return dt.timetuple().tm_yday


def _solar_declination(day_of_year: int) -> float:
    """
    Calculate solar declination angle in radians.
    
    Approximate formula: δ = 23.45° * sin(360/365 * (284 + n))
    """
    return np.radians(23.45 * np.sin(np.radians(360 / 365 * (284 + day_of_year))))


def _hour_angle(dt: datetime, longitude: float = -105.17) -> float:
    """
    Calculate hour angle in radians.
    
    longitude: -105.17° for Golden, CO
    """
    # Convert to hours from solar noon
    solar_noon = 12.0 - longitude / 15.0  # Approximate
    hour_decimal = dt.hour + dt.minute / 60.0
    return np.radians(15 * (hour_decimal - solar_noon))


def _calculate_irradiance(dt: datetime, capacity_factor: float = 0.8) -> float:
    """
    Calculate realistic plane-of-array irradiance based on time of day and season.
    
    Returns irradiance in W/m².
    """
    day = _day_of_year(dt)
    declination = _solar_declination(day)
    hour_angle = _hour_angle(dt)
    
    # Solar elevation angle (simplified)
    elevation = np.arcsin(
        np.sin(np.radians(LATITUDE)) * np.sin(declination) +
        np.cos(np.radians(LATITUDE)) * np.cos(declination) * np.cos(hour_angle)
    )
    
    # If sun is below horizon, irradiance is zero
    if elevation <= 0:
        return 0.0
    
    # Clear-sky irradiance model (simplified)
    # Max irradiance ~1000-1200 W/m² at solar noon
    max_irradiance = 1100.0 * capacity_factor
    
    # Apply atmospheric effects (simple cosine of elevation)
    atmospheric_factor = np.sin(elevation)
    
    # Add some cloud variability (random)
    cloud_factor = random.uniform(0.7, 1.0)
    
    irradiance = max_irradiance * atmospheric_factor * cloud_factor
    
    # Add measurement noise
    noise = random.gauss(0, 10)
    irradiance = max(0, irradiance + noise)
    
    return min(1500, irradiance)  # Physical upper bound


def _calculate_temperatures(irradiance: float, ambient_base: float = 20.0) -> tuple[float, float]:
    """
    Calculate ambient and panel temperatures based on irradiance.
    
    Returns (ambient_temp, panel_temp) in °C.
    """
    # Diurnal temperature variation
    hour_variation = 10 * np.sin(np.radians((irradiance / 1000) * 90))
    
    # Ambient temperature varies with time of day and season
    ambient = ambient_base + hour_variation + random.gauss(0, 2)
    
    # Panel temperature is higher than ambient due to solar heating
    # Typical: panel_temp = ambient + 0.025 * irradiance (for well-ventilated panels)
    panel_temp = ambient + 0.025 * irradiance + random.gauss(0, 1.5)
    
    return ambient, panel_temp


def _calculate_power(irradiance: float, panel_temp: float, capacity_kw: float = SITE_CAPACITY_KW) -> float:
    """
    Calculate AC power output from irradiance and panel temperature.
    
    Uses a simplified PV model:
    - Power scales linearly with irradiance at low levels
    - Temperature coefficient reduces efficiency at high temps
    - Inverter efficiency ~95%
    """
    if irradiance < 10:  # Nighttime or very low light
        return 0.0
    
    # Reference conditions: 1000 W/m², 25°C
    ref_irradiance = 1000.0
    ref_temp = 25.0
    
    # Temperature coefficient for power (typical -0.4% per °C for silicon)
    temp_coeff = -0.004
    
    # Normalized irradiance
    normalized_irradiance = irradiance / ref_irradiance
    
    # Temperature derating
    temp_factor = 1 + temp_coeff * (panel_temp - ref_temp)
    
    # DC power (before inverter)
    dc_power = capacity_kw * normalized_irradiance * temp_factor
    
    # Inverter efficiency (decreases at low power)
    inverter_efficiency = 0.95 if dc_power > 0.1 * capacity_kw else 0.85
    
    # AC power
    ac_power = dc_power * inverter_efficiency
    
    # Add measurement noise
    noise = random.gauss(0, 0.02)
    ac_power = max(0, ac_power + noise)
    
    return ac_power


def _calculate_wind_speed(irradiance: float) -> float:
    """Generate realistic wind speed (higher during day)."""
    base_speed = 2.0
    day_factor = (irradiance / 1000) * 2.0
    wind = base_speed + day_factor + random.gauss(0, 1.0)
    return max(0, min(30, wind))


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def generate_dataset(
    start_date: datetime,
    end_date: datetime,
    interval_minutes: int = 15,
    missing_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Generate a complete synthetic PV dataset.
    
    Parameters
    ----------
    start_date : Start timestamp
    end_date : End timestamp
    interval_minutes : Time interval in minutes
    missing_rate : Fraction of values to set as missing (simulating sensor dropouts)
    
    Returns
    -------
    DataFrame with columns matching NREL PVDAQ structure
    """
    log.info("Generating synthetic dataset from %s to %s", start_date, end_date)
    
    timestamps = []
    data = []
    
    current = start_date
    row_count = 0
    
    while current <= end_date:
        # Calculate irradiance first (driver for other variables)
        irradiance = _calculate_irradiance(current)
        
        # Calculate temperatures
        ambient_temp, panel_temp = _calculate_temperatures(irradiance)
        
        # Calculate power
        power = _calculate_power(irradiance, panel_temp)
        
        # Calculate wind speed
        wind_speed = _calculate_wind_speed(irradiance)
        
        # Calculate DC power (typically slightly higher than AC)
        dc_power = power / 0.95 if power > 0 else 0
        
        timestamps.append(current)
        data.append({
            "measdatetime": current.strftime("%Y-%m-%d %H:%M:%S"),
            "ac_power": power,
            "poa_irradiance": irradiance,
            "ambient_temp": ambient_temp,
            "module_temp": panel_temp,
            "wind_speed": wind_speed,
            "dc_power": dc_power,
        })
        
        row_count += 1
        if row_count % 10000 == 0:
            log.info("  Generated %d rows...", row_count)
        
        current += timedelta(minutes=interval_minutes)
    
    df = pd.DataFrame(data)
    
    # Introduce missing values to simulate real data quality issues
    if missing_rate > 0:
        log.info("Introducing %.1f%% missing values...", missing_rate * 100)
        for col in ["ambient_temp", "module_temp", "wind_speed"]:
            mask = np.random.random(len(df)) < missing_rate
            df.loc[mask, col] = np.nan
    
    log.info("Generated %d total rows", len(df))
    return df


def main() -> int:
    """Generate and save synthetic dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 60)
    log.info("Generating Synthetic Solar PV Dataset")
    log.info("Based on NREL PVDAQ Site 4 characteristics")
    log.info("=" * 60)
    
    # Generate data
    df = generate_dataset(START_DATE, END_DATE, INTERVAL_MINUTES)
    
    # Save raw annual files (mimicking NREL structure)
    for year in df["measdatetime"].str[:4].unique():
        year_df = df[df["measdatetime"].str.startswith(str(year))]
        out_path = RAW_DIR / f"pvdaq_site4_{year}.csv"
        year_df.to_csv(out_path, index=False)
        log.info("Saved %s (%d rows)", out_path.name, len(year_df))
    
    # Save merged file
    merged_path = PROCESSED_DIR / "pvdaq_site4_merged_raw.csv"
    df.to_csv(merged_path, index=False)
    log.info("Saved merged dataset: %s (%d rows)", merged_path, len(df))
    
    # Print summary
    log.info("")
    log.info("Dataset Summary:")
    log.info("  Rows:       %d", len(df))
    log.info("  Columns:    %s", list(df.columns))
    log.info("  Date range: %s  →  %s", df["measdatetime"].iloc[0], df["measdatetime"].iloc[-1])
    log.info("")
    log.info("Column statistics:")
    for col in ["ac_power", "poa_irradiance", "ambient_temp", "module_temp", "wind_speed"]:
        if col in df.columns:
            log.info("  %s: min=%.2f, max=%.2f, mean=%.2f", col, df[col].min(), df[col].max(), df[col].mean())
    
    log.info("")
    log.info("Next step:")
    log.info("  python src/preprocessing.py")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
