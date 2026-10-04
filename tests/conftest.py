"""
tests/conftest.py
=================
Shared pytest fixtures used across all test modules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal synthetic DataFrame that mimics the structure of PVDAQ data
# (used for unit tests that must NOT depend on the real downloaded dataset)
# ---------------------------------------------------------------------------

SAMPLE_DATA = {
    "timestamp": [
        "2012-06-01 06:00:00+00:00",
        "2012-06-01 06:15:00+00:00",
        "2012-06-01 12:00:00+00:00",
        "2012-06-01 12:15:00+00:00",
        "2012-06-01 20:00:00+00:00",  # nighttime
    ],
    "power":               [0.0, 0.3, 1.8, 1.75, 0.0],
    "irradiance":          [10.0, 120.0, 850.0, 820.0, 0.0],
    "ambient_temperature": [18.0, 20.0, 32.0, 31.5, 15.0],
    "panel_temperature":   [18.5, 25.0, 52.0, 51.0, 15.2],
    "wind_speed":          [1.5, 2.0, 3.5, 3.0, 0.8],
}


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a small DataFrame that mimics cleaned PVDAQ data."""
    df = pd.DataFrame(SAMPLE_DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent
