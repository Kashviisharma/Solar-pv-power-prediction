"""
frontend/app.py
===============
Streamlit frontend for Solar PV Power Prediction System.

Provides an interactive UI for:
- Making live predictions with sliders
- Real-time gauge and power meter display
- Model comparison table
- Historical time-series visualization
- Prediction error analysis
- Environmental variable overlays
- About / system info panel
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="☀️ Solar PV Power Prediction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* ── global ── */
    [data-testid="stAppViewContainer"] {
        background: #0e1117;
    }
    [data-testid="stSidebar"] {
        background: #161b22;
    }

    /* ── title banner ── */
    .title-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e6b800;
        border-radius: 12px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.2rem;
        text-align: center;
    }
    .title-banner h1 {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffd700;
        letter-spacing: -0.5px;
        margin: 0 0 0.3rem 0;
    }
    .title-banner p {
        font-size: 1rem;
        color: #a0aec0;
        margin: 0;
    }

    /* ── kpi cards ── */
    .kpi-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .kpi-card {
        flex: 1;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        text-align: center;
    }
    .kpi-card .kpi-label {
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .kpi-card .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffd700;
    }
    .kpi-card .kpi-unit {
        font-size: 0.75rem;
        color: #8b949e;
    }

    /* ── result card ── */
    .result-card {
        background: linear-gradient(135deg, #0d4f3c 0%, #064e3b 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.6rem;
        text-align: center;
        margin: 0.6rem 0;
    }
    .result-card .power-value {
        font-size: 3rem;
        font-weight: 800;
        color: #10b981;
        line-height: 1;
    }
    .result-card .power-label {
        font-size: 0.9rem;
        color: #6ee7b7;
        margin-top: 0.4rem;
    }

    /* ── status badge ── */
    .badge-healthy {
        background: #064e3b;
        color: #10b981;
        border: 1px solid #10b981;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-unhealthy {
        background: #450a0a;
        color: #f87171;
        border: 1px solid #f87171;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* ── section header ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        border-left: 3px solid #ffd700;
        padding-left: 0.6rem;
        margin: 1rem 0 0.6rem 0;
    }

    /* ── primary button ── */
    [data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: #0e1117;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.65rem 1.5rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        opacity: 0.88;
    }

    /* ── slider accent ── */
    [data-testid="stSlider"] .stSlider > div > div > div > div {
        background-color: #f59e0b !important;
    }

    /* ── tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        color: #8b949e;
        font-weight: 500;
    }
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        color: #ffd700;
        border-bottom-color: #ffd700;
    }

    /* ── metric overrides ── */
    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }

    /* ── info/warning/error override ── */
    .stAlert {
        border-radius: 8px;
    }

    /* ── table ── */
    table { width: 100%; }
    th { color: #8b949e !important; font-size: 0.8rem !important; }
    td { color: #e2e8f0 !important; font-size: 0.9rem !important; }

    /* ── footer ── */
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.78rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #21262d;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
METADATA_PATH = Path(__file__).parent.parent / "models" / "model_metadata.json"

# Base dark theme — intentionally NO xaxis/yaxis keys so individual charts
# can set their own axis options without keyword-argument conflicts.
PLOTLY_DARK = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e2e8f0"),
)

# Reusable axis style dicts applied per-chart
_AXIS_STYLE  = dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e"))
_AXIS_STYLE2 = dict(gridcolor="rgba(0,0,0,0)", linecolor="#30363d", tickfont=dict(color="#8b949e"))

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_metadata() -> dict | None:
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=30)
def load_historical_data() -> pd.DataFrame | None:
    ts_file = PROCESSED_DIR / "predictions_time_series.csv"
    if not ts_file.exists():
        return None
    df = pd.read_csv(ts_file, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    return df


@st.cache_data(ttl=30)
def check_api_health() -> dict:
    """Return health dict or {'status': 'offline'}."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "offline"}


# ──────────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────────

def predict_power(features: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=features, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API — make sure it is running on http://localhost:8000")
    except requests.exceptions.Timeout:
        st.error("⚠️ API request timed out")
    except requests.exceptions.HTTPError as e:
        st.error(f"⚠️ API returned an error: {e.response.text}")
    return None


def fetch_all_metrics() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Chart builders
# ──────────────────────────────────────────────────────────────────────────────

def make_gauge(value_kw: float, max_kw: float = 2.5) -> go.Figure:
    """Semi-circular power gauge."""
    pct = min(value_kw / max_kw, 1.0)
    color = "#10b981" if pct > 0.6 else "#f59e0b" if pct > 0.25 else "#60a5fa"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value_kw,
            number={"suffix": " kW", "font": {"size": 28, "color": color}},
            gauge={
                "axis": {"range": [0, max_kw], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "#161b22",
                "bordercolor": "#30363d",
                "borderwidth": 1,
                "steps": [
                    {"range": [0, max_kw * 0.25], "color": "#1a1a2e"},
                    {"range": [max_kw * 0.25, max_kw * 0.6], "color": "#16213e"},
                    {"range": [max_kw * 0.6, max_kw], "color": "#0f2a3c"},
                ],
                "threshold": {
                    "line": {"color": "#ffd700", "width": 3},
                    "thickness": 0.75,
                    "value": value_kw,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=20, b=10),
        paper_bgcolor="#0e1117",
        font=dict(color="#e2e8f0"),
    )
    return fig


def make_timeseries_chart(df: pd.DataFrame) -> go.Figure:
    sample = df if len(df) <= 1000 else df.sample(1000, random_state=42).sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample["timestamp"], y=sample["actual_power_kw"],
        mode="lines", name="Actual Power",
        line=dict(color="#60a5fa", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=sample["timestamp"], y=sample["predicted_power_kw"],
        mode="lines", name="Predicted Power",
        line=dict(color="#fbbf24", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text="Actual vs Predicted PV Power", font=dict(color="#e2e8f0")),
        xaxis=dict(**_AXIS_STYLE, title="Time"),
        yaxis=dict(**_AXIS_STYLE, title="Power (kW)"),
        hovermode="x unified",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_error_histogram(df: pd.DataFrame) -> go.Figure:
    errors = df["actual_power_kw"] - df["predicted_power_kw"]
    fig = px.histogram(
        errors, nbins=60,
        labels={"value": "Error (kW)", "count": "Count"},
        color_discrete_sequence=["#818cf8"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#f87171",
                  annotation_text="Zero Error", annotation_font_color="#f87171")
    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text="Prediction Error Distribution", font=dict(color="#e2e8f0")),
        xaxis=dict(**_AXIS_STYLE, title="Error (kW)"),
        yaxis=dict(**_AXIS_STYLE, title="Count"),
        showlegend=False,
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_scatter_chart(df: pd.DataFrame) -> go.Figure:
    sample = df if len(df) <= 800 else df.sample(800, random_state=7).sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample["actual_power_kw"], y=sample["predicted_power_kw"],
        mode="markers",
        marker=dict(color="#a78bfa", size=4, opacity=0.6),
        name="Predictions",
    ))
    lim = max(sample["actual_power_kw"].max(), sample["predicted_power_kw"].max()) * 1.05
    fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                  line=dict(color="#f59e0b", width=1.5, dash="dash"))
    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text="Actual vs Predicted Scatter", font=dict(color="#e2e8f0")),
        xaxis=dict(**_AXIS_STYLE, title="Actual Power (kW)"),
        yaxis=dict(**_AXIS_STYLE, title="Predicted Power (kW)"),
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_env_chart(df: pd.DataFrame) -> go.Figure:
    sample = df if len(df) <= 1000 else df.sample(1000, random_state=42).sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample["timestamp"], y=sample["irradiance"],
        mode="lines", name="Irradiance (W/m²)",
        line=dict(color="#fbbf24", width=1.5), yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=sample["timestamp"], y=sample["ambient_temperature"],
        mode="lines", name="Ambient Temp (°C)",
        line=dict(color="#34d399", width=1.5), yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=sample["timestamp"], y=sample["panel_temperature"],
        mode="lines", name="Panel Temp (°C)",
        line=dict(color="#f87171", width=1.5, dash="dot"), yaxis="y2",
    ))
    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text="Environmental Conditions Over Time", font=dict(color="#e2e8f0")),
        xaxis=dict(**_AXIS_STYLE, title="Time"),
        # Primary y-axis: irradiance (left) — use color= to tint ticks+title together
        yaxis=dict(
            gridcolor="#21262d",
            linecolor="#30363d",
            tickfont=dict(color="#fbbf24"),
            title=dict(text="Irradiance (W/m²)", font=dict(color="#fbbf24")),
        ),
        # Secondary y-axis: temperature (right)
        yaxis2=dict(
            gridcolor="rgba(0,0,0,0)",
            linecolor="#30363d",
            tickfont=dict(color="#34d399"),
            title=dict(text="Temperature (°C)", font=dict(color="#34d399")),
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_model_comparison_fig(all_metrics: dict) -> go.Figure:
    models = list(all_metrics.keys())
    mae_vals  = [all_metrics[m]["mae"]  for m in models]
    rmse_vals = [all_metrics[m]["rmse"] for m in models]
    r2_vals   = [all_metrics[m]["r2"]   for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="MAE",  x=models, y=mae_vals,  marker_color="#60a5fa"))
    fig.add_trace(go.Bar(name="RMSE", x=models, y=rmse_vals, marker_color="#fbbf24"))
    fig.update_layout(
        **PLOTLY_DARK,
        barmode="group",
        title=dict(text="Model Comparison — MAE & RMSE (lower is better)",
                   font=dict(color="#e2e8f0")),
        xaxis=dict(**_AXIS_STYLE),
        yaxis=dict(**_AXIS_STYLE, title="Error (kW)"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ☀️ Solar PV Predictor")
        st.markdown("---")

        # ── API status ──────────────────────────────────────────────
        health = check_api_health()
        if health.get("status") in ("healthy", "ok"):
            st.markdown(
                f'<div class="badge-healthy">✅ API Online — {health.get("model_name", "Model")} v{health.get("model_version", "")}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="badge-unhealthy">❌ API Offline</div>', unsafe_allow_html=True)
            st.info("Start the API:\n```\npython -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000\n```")

        st.markdown("---")

        # ── Model metrics ───────────────────────────────────────────
        meta = load_metadata()
        if meta:
            st.markdown("**📊 Best Model Performance**")
            m = meta["metrics"]
            col_a, col_b = st.columns(2)
            col_a.metric("R²",   f"{m['r2']:.4f}")
            col_b.metric("MAE",  f"{m['mae']:.4f} kW")
            st.metric("RMSE", f"{m['rmse']:.4f} kW")

            st.markdown("---")
            st.markdown("**🏆 Best Model**")
            st.success(meta["model_name"])

            st.markdown("**🔑 Features**")
            for feat in meta["features"]:
                st.write(f"• `{feat}`")

            st.markdown("---")
            st.markdown("**📅 Training Data**")
            st.caption(f"Train samples: **{meta['train_samples']:,}**")
            st.caption(f"Test samples:  **{meta['test_samples']:,}**")
        
        st.markdown("---")
        st.markdown(
            """
**ℹ️ About**

Predicts solar PV power (kW) from environmental sensor readings using a
machine learning model trained on real NREL PVDAQ data from Golden, CO.
""")

        st.markdown("---")
        st.markdown("**🔗 Quick Links**")
        st.markdown("[API Docs →](http://localhost:8000/docs)")
        st.markdown("[Swagger UI →](http://localhost:8000/redoc)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # ── Title banner ────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="title-banner">
  <h1>☀️ Solar PV Power Prediction System</h1>
  <p>Real-time power forecasting from environmental sensor data &nbsp;·&nbsp; NREL PVDAQ Site 4, Golden CO</p>
</div>
""",
        unsafe_allow_html=True,
    )

    meta = load_metadata()

    # ── Top KPI row ──────────────────────────────────────────────────────────
    if meta:
        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        kc1.metric("Model", meta["model_name"])
        kc2.metric("R²",    f"{meta['metrics']['r2']:.4f}")
        kc3.metric("MAE",   f"{meta['metrics']['mae']:.4f} kW")
        kc4.metric("RMSE",  f"{meta['metrics']['rmse']:.4f} kW")
        kc5.metric("Train Rows", f"{meta['train_samples']:,}")

    st.markdown("---")

    # ── Two-column layout: inputs (left) | charts (right) ───────────────────
    left_col, right_col = st.columns([1, 2], gap="large")

    # ── LEFT: Prediction input panel ────────────────────────────────────────
    with left_col:
        st.markdown('<div class="section-header">🎛️ Prediction Input</div>', unsafe_allow_html=True)

        irradiance = st.slider(
            "☀️ Solar Irradiance (W/m²)",
            min_value=0, max_value=1500, value=850, step=10,
            help="Plane-of-array irradiance — primary driver of power output.",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            ambient_temp = st.slider(
                "🌡️ Ambient Temp (°C)",
                min_value=-20, max_value=60, value=25, step=1,
                help="Ambient air temperature at the site.",
            )
        with col_b:
            panel_temp = st.slider(
                "🔆 Panel Temp (°C)",
                min_value=-20, max_value=100, value=43, step=1,
                help="PV module surface temperature.",
            )

        wind_speed = st.slider(
            "💨 Wind Speed (m/s)",
            min_value=0.0, max_value=30.0, value=2.5, step=0.5,
            help="Wind speed at hub height — affects module cooling.",
        )

        # ── derived irradiance class ──
        if irradiance < 50:
            irr_label = "🌑 Nighttime / No Sun"
        elif irradiance < 200:
            irr_label = "🌥️ Very Low (dawn/dusk)"
        elif irradiance < 500:
            irr_label = "⛅ Low (overcast)"
        elif irradiance < 800:
            irr_label = "🌤️ Medium (partly cloudy)"
        else:
            irr_label = "☀️ High (clear sky)"
        st.caption(f"Condition: **{irr_label}**")

        predict_clicked = st.button("⚡ Predict Power Output", type="primary")

        # ── result section ───────────────────────────────────────────────────
        if "last_result" not in st.session_state:
            st.session_state.last_result = None

        if predict_clicked:
            with st.spinner("Running inference…"):
                result = predict_power({
                    "irradiance": float(irradiance),
                    "ambient_temperature": float(ambient_temp),
                    "panel_temperature": float(panel_temp),
                    "wind_speed": float(wind_speed),
                })
            if result:
                st.session_state.last_result = result

        if st.session_state.last_result:
            r = st.session_state.last_result
            pred_kw = r["predicted_power_kw"]
            st.markdown(
                f"""
<div class="result-card">
  <div class="power-value">{pred_kw:.3f} kW</div>
  <div class="power-label">Predicted AC Power Output</div>
  <div class="power-label" style="margin-top:0.5rem;color:#6ee7b7;font-size:0.78rem">
    {r['model_name']} · v{r['model_version']}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.plotly_chart(make_gauge(pred_kw), use_container_width=True)

        st.markdown("---")
        with st.expander("📏 Reference Values", expanded=False):
            st.markdown("""
| Condition | Irradiance |
|-----------|-----------|
| Nighttime | 0 W/m² |
| Dawn / Dusk | 10–100 W/m² |
| Overcast | 100–300 W/m² |
| Partly Cloudy | 300–700 W/m² |
| Clear Sky | 700–1000+ W/m² |

**Typical Panel Temp** ≈ Ambient + 20–35 °C under sun.
""")

    # ── RIGHT: Visualizations ────────────────────────────────────────────────
    with right_col:
        tab_ts, tab_err, tab_scatter, tab_env, tab_models = st.tabs([
            "📈 Time Series",
            "📊 Error Analysis",
            "🔵 Scatter Plot",
            "🌡️ Environment",
            "🏆 Model Comparison",
        ])

        hist_df = load_historical_data()
        no_data_msg = (
            "No historical data found. Run the simulator first:\n"
            "```\npython -m scripts.simulate_stream --max-rows 1000\n```"
        )

        # ── Tab 1: Time series ──────────────────────────────────────────────
        with tab_ts:
            if hist_df is not None and len(hist_df) > 0:
                st.plotly_chart(make_timeseries_chart(hist_df), use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                mae_hist = (hist_df["actual_power_kw"] - hist_df["predicted_power_kw"]).abs().mean()
                c1.metric("Data Points", f"{len(hist_df):,}")
                c2.metric("Historical MAE", f"{mae_hist:.4f} kW")
                c3.metric("Peak Actual", f"{hist_df['actual_power_kw'].max():.2f} kW")
                c4.metric("Peak Predicted", f"{hist_df['predicted_power_kw'].max():.2f} kW")
            else:
                st.warning(no_data_msg)

        # ── Tab 2: Error distribution ───────────────────────────────────────
        with tab_err:
            if hist_df is not None and len(hist_df) > 0:
                err = hist_df["actual_power_kw"] - hist_df["predicted_power_kw"]
                st.plotly_chart(make_error_histogram(hist_df), use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Error", f"{err.mean():.4f} kW")
                c2.metric("Std Error",  f"{err.std():.4f} kW")
                c3.metric("Max |Error|", f"{err.abs().max():.4f} kW")
            else:
                st.warning(no_data_msg)

        # ── Tab 3: Scatter ──────────────────────────────────────────────────
        with tab_scatter:
            if hist_df is not None and len(hist_df) > 0:
                st.plotly_chart(make_scatter_chart(hist_df), use_container_width=True)
                st.caption(
                    "Points should cluster tightly along the dashed 1:1 line. "
                    "Deviation indicates systematic bias or outliers."
                )
            else:
                st.warning(no_data_msg)

        # ── Tab 4: Environment ──────────────────────────────────────────────
        with tab_env:
            if hist_df is not None and len(hist_df) > 0:
                st.plotly_chart(make_env_chart(hist_df), use_container_width=True)
            else:
                st.warning(no_data_msg)

        # ── Tab 5: Model comparison ─────────────────────────────────────────
        with tab_models:
            if meta and "all_metrics" in meta and meta["all_metrics"]:
                all_m = meta["all_metrics"]
                st.plotly_chart(make_model_comparison_fig(all_m), use_container_width=True)

                # Table
                rows = []
                for model_name, metrics in all_m.items():
                    best = "⭐" if model_name == meta["model_name"] else ""
                    rows.append({
                        "Model": f"{best} {model_name}".strip(),
                        "MAE (kW)":  f"{metrics['mae']:.6f}",
                        "RMSE (kW)": f"{metrics['rmse']:.6f}",
                        "R²":        f"{metrics['r2']:.6f}",
                    })
                st.dataframe(
                    pd.DataFrame(rows).set_index("Model"),
                    use_container_width=True,
                )
                st.caption(f"Best model: **{meta['model_name']}** (highlighted ⭐). Deployed for inference.")
            else:
                st.info("Model comparison data not available in metadata.")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="footer">
    Solar PV Power Prediction System &nbsp;·&nbsp; Built with Streamlit, FastAPI &amp; Scikit-learn &nbsp;·&nbsp;
    NREL PVDAQ Site 4 (Golden, CO) &nbsp;·&nbsp; R² 0.9976 &nbsp;·&nbsp; MAE 0.0064 kW
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
