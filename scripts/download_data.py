#!/usr/bin/env python3
"""
scripts/download_data.py
========================
Download NREL PVDAQ solar PV measurement data for Site 4 (Golden, CO).

Dataset:  NREL PVDAQ — Site 4 (NREL STR, Golden, Colorado)
Source:   National Renewable Energy Laboratory, U.S. Dept of Energy
API docs: https://developer.nrel.gov/docs/solar/pvdaq-v3/
OEDI:     https://data.openei.org/submissions/4568
License:  U.S. Government Open Data (public domain equivalent)

Usage
-----
    # With NREL API key (recommended):
    python scripts/download_data.py

    # The script reads NREL_API_KEY from .env automatically.
    # Get a free key at: https://developer.nrel.gov/signup/

    # The script will download data year by year (to avoid large requests)
    # and save each year as:
    #   data/raw/pvdaq_site4_YYYY.csv

    # After downloading, run the merge step:
    #   python scripts/download_data.py --merge

    # To download a specific year:
    #   python scripts/download_data.py --year 2012

    # To use the OEDI S3 mirror (no API key required):
    #   python scripts/download_data.py --source oedi

Fallback
--------
If neither NREL API key nor OEDI S3 is accessible, the script will
print clear instructions for manual download and exit gracefully.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")

# ── PVDAQ configuration ────────────────────────────────────────────────────
NREL_API_KEY = os.getenv("NREL_API_KEY", "")
PVDAQ_SITE_ID = int(os.getenv("PVDAQ_SITE_ID", "4"))
PVDAQ_START_YEAR = int(os.getenv("PVDAQ_START_YEAR", "2011"))
PVDAQ_END_YEAR = int(os.getenv("PVDAQ_END_YEAR", "2013"))

# NREL PVDAQ REST API base URL
PVDAQ_API_BASE = "https://developer.nrel.gov/api/pvdaq/v3"

# OEDI S3 mirror — NREL makes some PVDAQ data available here without a key.
# These are the known public CSV files for PVDAQ site 4.
OEDI_BASE = (
    "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems"
)

# Mapping: PVDAQ column names → application column names
# (Full mapping; only columns present in the actual data are used)
COLUMN_MAP = {
    "measdatetime": "timestamp",
    "ac_power":     "power",           # TARGET — AC power output in kW
    "poa_irradiance": "irradiance",    # Plane-of-array irradiance, W/m²
    "ambient_temp": "ambient_temperature",  # °C
    "module_temp":  "panel_temperature",    # °C
    "wind_speed":   "wind_speed",           # m/s
    "dc_power":     "dc_power",             # kW (auxiliary)
    # Some PVDAQ exports use slightly different names:
    "ac_power_kw":  "power",
    "poa":          "irradiance",
    "t_amb":        "ambient_temperature",
    "t_mod":        "panel_temperature",
    "ws":           "wind_speed",
}

# Minimum required columns (application names) for the ML pipeline
REQUIRED_COLUMNS = {"timestamp", "power", "irradiance"}


# ---------------------------------------------------------------------------
# NREL API download
# ---------------------------------------------------------------------------

def _build_api_url(year: int, site_id: int, api_key: str) -> str:
    """Build PVDAQ data request URL for a single calendar year."""
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    return (
        f"{PVDAQ_API_BASE}/data.csv"
        f"?api_key={api_key}"
        f"&system_id={site_id}"
        f"&start_date={start}"
        f"&end_date={end}"
        f"&resolution=15"           # 15-minute intervals
        f"&measurement_type=all"    # request all sensors
    )


def download_via_api(year: int, site_id: int, api_key: str, out_dir: Path) -> Path | None:
    """
    Download one year of PVDAQ data via NREL REST API.

    Returns the path to the saved CSV, or None on failure.
    """
    url = _build_api_url(year, site_id, api_key)
    out_path = out_dir / f"pvdaq_site{site_id}_{year}.csv"

    if out_path.exists():
        log.info("Already exists, skipping: %s", out_path.name)
        return out_path

    log.info("Downloading %d via NREL API …", year)
    log.debug("URL: %s", url.replace(api_key, "***"))

    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        log.error("HTTP error downloading %d: %s", year, exc)
        if resp.status_code == 403:
            log.error(
                "403 Forbidden — your API key may be invalid or the site ID may "
                "not be accessible. Check https://developer.nrel.gov/signup/"
            )
        return None
    except requests.exceptions.RequestException as exc:
        log.error("Network error downloading %d: %s", year, exc)
        return None

    # Check that we actually got CSV (not an error JSON response)
    content_type = resp.headers.get("Content-Type", "")
    if "json" in content_type:
        log.error(
            "API returned JSON instead of CSV for year %d. "
            "Response: %s",
            year,
            resp.text[:500],
        )
        return None

    out_path.write_bytes(resp.content)
    size_kb = out_path.stat().st_size / 1024
    log.info("Saved %s (%.1f KB)", out_path.name, size_kb)
    return out_path


# ---------------------------------------------------------------------------
# OEDI S3 mirror download (no API key needed)
# ---------------------------------------------------------------------------

def _oedi_url(year: int, site_id: int) -> str:
    """
    Construct OEDI S3 URL for a PVDAQ site-year CSV.

    OEDI organises PVDAQ data as:
      s3://oedi-data-lake/pvdaq/csv/systems/{site_id}/{site_id}_{year}.csv
    The public HTTPS endpoint mirrors this path.
    """
    return f"{OEDI_BASE}/{site_id}/{site_id}_{year}.csv"


def download_via_oedi(year: int, site_id: int, out_dir: Path) -> Path | None:
    """
    Download one year of PVDAQ data from the OEDI S3 open-data mirror.

    Returns the path to the saved CSV, or None on failure.
    """
    url = _oedi_url(year, site_id)
    out_path = out_dir / f"pvdaq_site{site_id}_{year}.csv"

    if out_path.exists():
        log.info("Already exists, skipping: %s", out_path.name)
        return out_path

    log.info("Downloading %d via OEDI mirror: %s", year, url)

    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        log.warning("OEDI mirror HTTP error for year %d: %s", year, exc)
        return None
    except requests.exceptions.RequestException as exc:
        log.warning("OEDI mirror network error for year %d: %s", year, exc)
        return None

    out_path.write_bytes(resp.content)
    size_kb = out_path.stat().st_size / 1024
    log.info("Saved %s (%.1f KB)", out_path.name, size_kb)
    return out_path


# ---------------------------------------------------------------------------
# Data inspection and column normalisation
# ---------------------------------------------------------------------------

def inspect_raw_file(csv_path: Path) -> None:
    """
    Load a raw CSV and print a summary of its structure.
    Helps the user verify columns are as expected.
    """
    try:
        df = pd.read_csv(csv_path, nrows=5)
    except Exception as exc:
        log.error("Could not read %s: %s", csv_path.name, exc)
        return

    log.info(
        "\n--- %s ---\n"
        "Columns (%d):  %s\n"
        "First row:\n%s",
        csv_path.name,
        len(df.columns),
        list(df.columns),
        df.head(2).to_string(),
    )


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw PVDAQ columns to application column names using COLUMN_MAP.

    Handles case-insensitive matching and whitespace in column names.
    Only renames columns that exist; unknown columns are kept as-is.
    """
    # Normalise column names to lowercase+stripped for matching
    rename = {}
    for col in df.columns:
        normalised = col.strip().lower()
        if normalised in COLUMN_MAP:
            rename[col] = COLUMN_MAP[normalised]
    if rename:
        log.debug("Renaming columns: %s", rename)
    return df.rename(columns=rename)


def validate_columns(df: pd.DataFrame, source: str) -> bool:
    """Check that the minimum required columns are present after normalisation."""
    present = set(df.columns)
    missing = REQUIRED_COLUMNS - present
    if missing:
        log.error(
            "File %s is missing required columns after normalisation: %s\n"
            "Available columns: %s",
            source,
            missing,
            list(present),
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Merge annual files
# ---------------------------------------------------------------------------

def merge_annual_files(raw_dir: Path, site_id: int) -> pd.DataFrame | None:
    """
    Concatenate all annual CSVs for a site into a single DataFrame.

    Returns the merged DataFrame, or None if no files found.
    """
    pattern = f"pvdaq_site{site_id}_*.csv"
    files = sorted(raw_dir.glob(pattern))

    if not files:
        log.error("No annual CSV files found in %s matching %s", raw_dir, pattern)
        return None

    log.info("Merging %d annual files …", len(files))
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as exc:
            log.warning("Could not read %s: %s", f.name, exc)
            continue

        df = normalise_columns(df)
        if not validate_columns(df, f.name):
            log.warning("Skipping %s due to missing columns.", f.name)
            continue

        frames.append(df)
        log.info("  Loaded %s: %d rows", f.name, len(df))

    if not frames:
        log.error("No valid files to merge.")
        return None

    merged = pd.concat(frames, ignore_index=True)
    log.info("Merged total: %d rows", len(merged))
    return merged


def save_merged(merged: pd.DataFrame, processed_dir: Path, site_id: int) -> Path:
    """Save the merged DataFrame to the processed directory."""
    out_path = processed_dir / f"pvdaq_site{site_id}_merged_raw.csv"
    merged.to_csv(out_path, index=False)
    log.info("Merged raw data saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Manual download instructions
# ---------------------------------------------------------------------------

MANUAL_INSTRUCTIONS = """
================================================================================
MANUAL DOWNLOAD INSTRUCTIONS
================================================================================

Automatic download failed. You can obtain the dataset manually:

METHOD 1 — NREL Developer API
-------------------------------
1. Register for a FREE API key (takes ~1 minute):
   https://developer.nrel.gov/signup/

2. Add your key to .env:
   NREL_API_KEY=your_actual_key_here

3. Re-run this script:
   python scripts/download_data.py

METHOD 2 — OEDI Open Data (no key required)
--------------------------------------------
1. Visit:
   https://data.openei.org/submissions/4568

2. Download the PVDAQ CSV files for Site 4.

3. Place them in:
   data/raw/pvdaq_site4_YYYY.csv

   (where YYYY is the year, e.g., pvdaq_site4_2011.csv)

4. Run the merge step:
   python scripts/download_data.py --merge

METHOD 3 — PVDAQ Web Interface
--------------------------------
1. Visit: https://pvdaq.nrel.gov/
2. Select System ID: 4
3. Choose date range and export as CSV.
4. Place files in data/raw/ with naming convention above.

================================================================================
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download NREL PVDAQ solar PV data for Site 4."
    )
    p.add_argument(
        "--source",
        choices=["api", "oedi", "auto"],
        default="auto",
        help=(
            "Download source. "
            "'auto' tries OEDI first (no key needed), then API. "
            "Default: auto"
        ),
    )
    p.add_argument(
        "--year",
        type=int,
        default=None,
        help="Download only this specific year (overrides .env range).",
    )
    p.add_argument(
        "--site-id",
        type=int,
        default=PVDAQ_SITE_ID,
        help=f"PVDAQ site ID (default: {PVDAQ_SITE_ID})",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="Merge all downloaded annual CSV files and save to data/processed/.",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Print column names and sample rows of downloaded files.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Ensure directories exist ───────────────────────────────────────────
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)

    site_id = args.site_id
    years = [args.year] if args.year else list(range(PVDAQ_START_YEAR, PVDAQ_END_YEAR + 1))

    log.info("=" * 60)
    log.info("NREL PVDAQ Download — Site %d", site_id)
    log.info("Years: %s", years)
    log.info("Output dir: %s", RAW_DIR)
    log.info("=" * 60)

    # ── If only merging, skip download ────────────────────────────────────
    if not args.merge:
        downloaded: list[Path] = []
        failed_years: list[int] = []

        for year in years:
            path: Path | None = None

            # --- Try OEDI first (no key needed) ---
            if args.source in ("oedi", "auto"):
                path = download_via_oedi(year, site_id, RAW_DIR)

            # --- Fall back to NREL API ---
            if path is None and args.source in ("api", "auto"):
                if not NREL_API_KEY or NREL_API_KEY == "your_api_key_here":
                    log.warning(
                        "NREL_API_KEY not set. "
                        "Skipping API download for year %d.",
                        year,
                    )
                else:
                    path = download_via_api(year, site_id, NREL_API_KEY, RAW_DIR)

            if path:
                downloaded.append(path)
                if args.inspect:
                    inspect_raw_file(path)
                time.sleep(0.5)   # be polite to NREL servers
            else:
                failed_years.append(year)

        log.info("")
        log.info("Download summary:")
        log.info("  ✓ Succeeded: %d years", len(downloaded))
        log.info("  ✗ Failed:    %d years  %s", len(failed_years), failed_years)

        if not downloaded:
            print(MANUAL_INSTRUCTIONS)
            return 1

        if failed_years:
            log.warning(
                "Some years failed. The pipeline will work with partial data "
                "but coverage may be limited."
            )

    # ── Merge step ────────────────────────────────────────────────────────
    if args.merge or not args.merge:
        # Always attempt merge after downloading
        log.info("")
        log.info("Merging annual files …")
        merged = merge_annual_files(RAW_DIR, site_id)
        if merged is not None:
            out = save_merged(merged, PROCESSED_DIR, site_id)

            # Print quick dataset summary
            log.info("")
            log.info("Dataset summary:")
            log.info("  Rows:       %d", len(merged))
            log.info("  Columns:    %s", list(merged.columns))
            if "timestamp" in merged.columns:
                ts = pd.to_datetime(merged["timestamp"], errors="coerce")
                log.info("  Date range: %s  →  %s", ts.min(), ts.max())
            log.info("")
            log.info("Next step:")
            log.info("  python src/preprocessing.py")
            log.info("  — or open notebooks/01_data_exploration.ipynb")
            return 0
        else:
            log.error("Merge failed. See errors above.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
