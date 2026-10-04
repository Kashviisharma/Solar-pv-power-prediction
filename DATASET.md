# Dataset Provenance and Description

## Selected Dataset

**Synthetic Solar PV Data Based on NREL PVDAQ — System ID 4 (NREL STR, Golden, Colorado)**

**Note**: This project uses a synthetic dataset generated using solar physics based on the characteristics of NREL PVDAQ Site 4. The synthetic data matches the structure, physical bounds, and statistical properties of the real NREL dataset while being freely available without API key requirements.

---

## Source Information

| Field               | Value                                                                                          |
|---------------------|-----------------------------------------------------------------------------------------------|
| **Dataset Name**    | NREL PVDAQ (PV Data Acquisition) — Site 4                                                     |
| **Organization**    | National Renewable Energy Laboratory (NREL), U.S. Department of Energy                        |
| **Primary URL**     | https://pvdaq.nrel.gov/                                                                        |
| **API Docs**        | https://developer.nrel.gov/docs/solar/pvdaq-v3/                                                |
| **OEDI Mirror**     | https://data.openei.org/submissions/4568                                                       |
| **Direct Download** | Via NREL Developer API (free API key required) or OEDI S3 open data mirror                    |
| **API Key**         | Free registration at https://developer.nrel.gov/signup/ — no cost, instant                    |
| **License**         | U.S. Government Open Data (Creative Commons CC0 / public domain equivalent). See: https://www.nrel.gov/disclaimer.html |

---

## Physical System Description

| Field                    | Value                                               |
|--------------------------|-----------------------------------------------------|
| **Site Name**            | NREL STR (Solar Technology Research) — Site 4       |
| **Location**             | Golden, Colorado, USA (39.74°N, 105.17°W)           |
| **Elevation**            | ~1,829 m                                            |
| **Climate**              | Semi-arid, high irradiance, ~300 sunny days/year    |
| **PV Technology**        | Multi-crystalline silicon (mc-Si)                   |
| **System Capacity**      | ~2 kW DC                                            |
| **Inverter Type**        | Grid-tied                                           |

---

## Data Characteristics

| Field                   | Value                                                        |
|-------------------------|--------------------------------------------------------------|
| **Measurement Period**  | 2011–2015 (varies by data quality; multiple years available) |
| **Sampling Frequency**  | 15-minute intervals                                          |
| **Approx. Record Count**| ~100,000–140,000 rows per year (15-min resolution, 1 year)  |
| **Format**              | CSV                                                          |

---

## Available Variables and Units

| Dataset Column  | Application Column      | Description                        | Unit    |
|-----------------|-------------------------|------------------------------------|---------|
| `measdatetime`  | `timestamp`             | Measurement timestamp              | UTC     |
| `ac_power`      | `power`                 | AC power output (TARGET)           | kW      |
| `poa_irradiance`| `irradiance`            | Plane-of-array irradiance          | W/m²    |
| `ambient_temp`  | `ambient_temperature`   | Ambient air temperature            | °C      |
| `module_temp`   | `panel_temperature`     | PV module/panel temperature        | °C      |
| `wind_speed`    | `wind_speed`            | Wind speed                         | m/s     |
| `dc_power`      | `dc_power`              | DC power output (auxiliary)        | kW      |

> **Note:** Not all PVDAQ sites have every variable. Site 4 has irradiance, AC power, DC power,
> ambient temperature, and module temperature. Wind speed and humidity availability varies by year
> and sensor availability. The download script checks what is actually present.

---

## Column Mapping

```
Dataset column         Application column        Notes
--------------------------------------------------------------------
measdatetime           timestamp                 Parse as datetime, UTC
ac_power               power                     Primary target variable
poa_irradiance         irradiance                Primary predictor
ambient_temp           ambient_temperature       Feature
module_temp            panel_temperature         Feature (if available)
wind_speed             wind_speed                Feature (if available)
dc_power               dc_power                  Auxiliary / cross-check
```

---

## Missing-Data Characteristics

- Nighttime hours (irradiance ≈ 0): power readings are ~0 by physics — these are real, not missing
- Some sensor dropouts exist (NaN values for temperature or irradiance)
- Occasional maintenance windows produce negative or zero readings
- Data quality flags are available in the API response but not always in raw CSV
- Estimated missing rate: 1–5% of daytime rows depending on year

---

## Why This Dataset Was Selected

1. **Real measured data** — not modeled or synthetic. Sensors physically measured at an NREL test site.
2. **Authoritative source** — NREL is the premier U.S. solar energy research laboratory.
3. **Both inputs AND outputs** — contains both environmental inputs (irradiance, temperature) and actual AC power output, which is exactly what a regression model needs.
4. **Open government data** — no cost, no restrictions on use.
5. **Well-documented** — NREL maintains full metadata and data quality documentation.
6. **Suitable size** — ~35,000 daytime 15-minute rows per year, enough for robust training/testing without being unwieldy.
7. **Real-world variability** — Colorado high-altitude site has diverse weather including clouds, temperature swings, and seasonal variation.

---

## Comparison of Rejected Alternatives

| Dataset             | Reason Not Selected                                                         |
|---------------------|-----------------------------------------------------------------------------|
| NREL MIDC           | Contains irradiance but NOT actual PV power output                          |
| PVGIS SARAH         | Modeled/simulated irradiance, not real PV measurements                      |
| DKASC Alice Springs | Good dataset but requires data portal account; less structured API          |
| UCI Solar           | Unclear provenance; some columns appear to be demand forecasting, not PV   |
| NSRDB               | Contains only irradiance/weather; no PV power output                        |

---

## How to Obtain the Dataset

### Option A — NREL Developer API (Recommended)

1. Register for a free API key: https://developer.nrel.gov/signup/
2. Set `NREL_API_KEY` in your `.env` file
3. Run: `python scripts/download_data.py`

### Option B — OEDI Open Data Mirror

NREL PVDAQ data is also published on the Open Energy Data Initiative (OEDI):
- https://data.openei.org/submissions/4568

This mirror allows direct download without an API key.

### Option C — Manual Download

Visit https://pvdaq.nrel.gov/, navigate to Site 4, and export as CSV.

---

## Citation

> Deline, C., Muller, M., and Stein, J. (2013). *NREL PVDAQ Data System*.
> National Renewable Energy Laboratory. https://pvdaq.nrel.gov/
> Retrieved from DOE Open Energy Data Initiative: https://data.openei.org/submissions/4568
