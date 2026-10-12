"""
AI-Based Electricity Demand Projection - Delhi Power System
Interactive Stage 11 Web Application with Transparent Input-Output Console.
"""

from __future__ import annotations

import sys
from datetime import datetime, time, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import importlib
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import utils.data
import utils.features
importlib.reload(utils.data)
importlib.reload(utils.features)

from utils.data import (
    ROOT,
    find_model,
    is_model_available,
    load_history,
    load_json,
    load_model,
    load_predictions,
)
try:
    from utils.data import HISTORICAL_HIGHLIGHTS
except ImportError:
    HISTORICAL_HIGHLIGHTS = [
        {
            "title": "🔥 Delhi All-Time Record Peak (8,631.53 MW)",
            "timestamp": "2024-06-19 15:05:00",
            "description": "Historical maximum power demand in Delhi during the historic June 2024 heatwave (44.5°C).",
            "category": "Record Peak",
        },
        {
            "title": "🌙 Summer Midnight AC Peak (7,842.10 MW)",
            "timestamp": "2024-06-25 23:30:00",
            "description": "Intense late-night domestic air conditioning surge with temperature at 36.5°C.",
            "category": "Night Cooling",
        },
        {
            "title": "🌧️ Sultry Monsoon Afternoon (6,415.80 MW)",
            "timestamp": "2024-08-12 14:15:00",
            "description": "Heavy post-monsoon humidity (85% RH) causing sustained compressor loads.",
            "category": "Monsoon Peak",
        },
        {
            "title": "❄️ Winter Morning Peak (4,912.40 MW)",
            "timestamp": "2024-01-15 08:35:00",
            "description": "Peak morning domestic heating and water geyser surge during dense winter cold spell (8.2°C).",
            "category": "Winter Heating",
        },
        {
            "title": "🍃 Pleasant Spring Afternoon (3,520.15 MW)",
            "timestamp": "2024-03-10 12:45:00",
            "description": "Mild, comfortable weather (24.2°C) requiring neither space cooling nor heating.",
            "category": "Baseload",
        },
    ]

from utils.features import (
    FEATURE_COLUMNS,
    FEATURE_INFO,
    LAGS,
    SCENARIO_PRESETS,
    WEATHER_COLUMNS,
    calculate_heat_index,
    classify_demand_regime,
    estimate_lags,
    get_wind_direction_label,
    make_custom_feature_row,
    make_feature_row,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Aesthetics
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Delhi Power System | AI Demand Projection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Modern typography and styling */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Custom Header Banner */
.app-header {
    background: linear-gradient(135deg, #091e3a 0%, #2f80ed 50%, #2d9cdb 100%);
    padding: 24px 32px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 50, 120, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.app-header h1 {
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
    color: #ffffff !important;
}

.app-header p {
    font-size: 1.05rem;
    margin: 6px 0 0 0;
    opacity: 0.9;
    color: #e0f2fe;
}

/* Card Containers */
.custom-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 24px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    margin-bottom: 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.dark-mode .custom-card {
    background: #1e293b;
    border-color: #334155;
}

/* Metric Display Card */
.metric-box {
    background: linear-gradient(145deg, #f8fafc 0%, #edf2f7 100%);
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.metric-box .metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0f172a;
    font-family: 'JetBrains Mono', monospace;
}

.metric-box .metric-lbl {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #64748b;
    margin-top: 4px;
}

/* Prediction Output Banner */
.prediction-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    border-radius: 16px;
    padding: 26px 30px;
    border: 1px solid #334155;
    box-shadow: 0 12px 36px rgba(15, 23, 42, 0.2);
    margin-bottom: 24px;
}

.prediction-val {
    font-size: 3.2rem;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}

/* Badges */
.badge-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

/* Section Subheaders */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Feature input card */
.feature-input-group {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

.feature-group-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #334155;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
}

/* Callout Note */
.info-callout {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    color: #1e40af;
    font-size: 0.9rem;
    margin-bottom: 16px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def demand_unit() -> str:
    return "MW"


# -----------------------------------------------------------------------------
# Chart Helpers
# -----------------------------------------------------------------------------
def line_chart(frame: pd.DataFrame, title: str, y_columns: list[str], labels: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    palette = ["#0284c7", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    for i, column in enumerate(y_columns):
        fig.add_scatter(
            x=frame["datetime"],
            y=frame[column],
            mode="lines",
            name=labels.get(column, column),
            line=dict(width=2, color=palette[i % len(palette)]),
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, weight="bold")),
        xaxis=dict(title="Datetime", gridcolor="rgba(226, 232, 240, 0.6)"),
        yaxis=dict(title=f"Demand ({demand_unit()})", gridcolor="rgba(226, 232, 240, 0.6)"),
        height=440,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def gauge_chart(predicted_mw: float) -> go.Figure:
    """Renders a high-resolution semicircular gauge of Delhi grid capacity."""
    regime = classify_demand_regime(predicted_mw)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=predicted_mw,
            domain={"x": [0, 1], "y": [0, 1]},
            delta={"reference": 4413.68, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
            title={"text": "<b>Delhi Grid Load Level (MW)</b><br><span style='font-size:0.75em;color:#64748b'>Mean: 4,414 MW | Historical Record Peak: 8,631.5 MW</span>", "font": {"size": 15}},
            number={"suffix": " MW", "font": {"size": 25, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [1000, 10000], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": regime["color"], "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [1000, 5589.38], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [5589.38, 6905.92], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [6905.92, 7654.84], "color": "rgba(249, 115, 22, 0.25)"},
                    {"range": [7654.84, 10000], "color": "rgba(239, 68, 68, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 4},
                    "thickness": 0.8,
                    "value": 8631.53,  # Historical all-time high
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=25, r=25, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# -----------------------------------------------------------------------------
# Data Loading & Initialization
# -----------------------------------------------------------------------------
try:
    history = load_history()
except Exception as exc:
    st.error(f"The cleaned dataset could not be loaded: {exc}")
    st.stop()


def latest_lag_complete_timestamp(df: pd.DataFrame) -> pd.Timestamp:
    """Finds the latest timestamp for which all historical lag deltas exist."""
    idx = pd.DatetimeIndex(df["datetime"])
    valid = pd.Series(True, index=idx)
    for delta in LAGS.values():
        valid &= (idx - delta).isin(idx)
    valid_matches = idx[valid.to_numpy()]
    return valid_matches[-1] if len(valid_matches) > 0 else df["datetime"].iloc[-1]


def predict_from_features(model_key: str, feature_df: pd.DataFrame) -> float:
    """Runs inference directly on a prepared 17-feature DataFrame."""
    model = load_model(model_key)
    missing = [c for c in FEATURE_COLUMNS if c not in feature_df.columns or feature_df[c].isna().any()]
    if missing:
        raise ValueError(f"Missing required input features: {', '.join(missing)}")
    return float(model.predict(feature_df[FEATURE_COLUMNS])[0])


def predict_at(
    model_key: str,
    timestamp: pd.Timestamp,
    df_history: pd.DataFrame,
    recursive: dict[pd.Timestamp, float] | None = None,
) -> float:
    """Predict one historical/future timestamp using exact causal lags."""
    base = df_history.set_index("datetime")
    demand = base["Power demand"].copy()
    if recursive:
        demand = pd.concat([demand, pd.Series(recursive)]).sort_index()
        demand = demand[~demand.index.duplicated(keep="last")]
    weather = base.loc[timestamp, WEATHER_COLUMNS] if timestamp in base.index else base.iloc[-1][WEATHER_COLUMNS]
    features = make_feature_row(timestamp, demand, weather)
    return predict_from_features(model_key, features)


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Health
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 2.4rem;">⚡</div>
            <h2 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #0284c7;">Delhi SLDC AI</h2>
            <p style="margin: 0; font-size: 0.8rem; color: #64748b;">5-Min Power Demand Forecasting</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation Menu",
        [
            "⚡ Demand Prediction & What-If",
            "🏠 Project Overview",
            "📊 Data Explorer",
            "🎯 Model Performance",
            "🔮 Future Demand Forecast",
            "🏔️ Peak Demand Analysis",
            "📖 About & Methodology",
        ],
        index=0,  # Prioritize the interactive prediction page requested by user
    )

    st.markdown("---")
    st.markdown("### 🤖 Loaded AI Models")
    lr_ok = is_model_available("linear_regression")
    rf_ok = is_model_available("random_forest")

    c_lr, c_rf = st.columns(2)
    with c_lr:
        st.markdown(f"**Linear Reg.**<br>{'✅ Ready' if lr_ok else '❌ Missing'}", unsafe_allow_html=True)
    with c_rf:
        st.markdown(f"**Random Forest**<br>{'✅ Ready' if rf_ok else '❌ Missing'}", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Delhi Power System | 5-Minute Resolution\nDataset Span: 2021 – 2024\nTarget: Total Power Demand (MW)")


# -----------------------------------------------------------------------------
# PAGE 1: DEMAND PREDICTION & WHAT-IF CONSOLE (CORE FOCUS)
# -----------------------------------------------------------------------------
def page_prediction_console(df_history: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>⚡ AI Electricity Demand Prediction & What-If Simulator</h1>
            <p>Complete transparency into the 17 input parameters driving Delhi's 5-minute electricity forecast.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-callout">
            💡 <b>How it works:</b> Choose an <b>Interactive What-If Scenario</b> to manually set or adjust 
            weather and demand lags, or load a <b>Historical Real-World Record</b> from Delhi's 2021–2024 dataset. 
            All 17 input features fed into the AI models are displayed in real time.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Control Bar: Model Selector and Input Mode
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        model_choice = st.selectbox(
            "Primary Forecasting Model",
            ["Linear Regression", "Random Forest"],
            help="Select which trained ML model to run as primary forecaster. Both models are loaded.",
        )
        primary_key = "linear_regression" if model_choice == "Linear Regression" else "random_forest"
    with col_ctrl2:
        input_mode = st.radio(
            "Data Input Mode",
            [
                "🎛️ Interactive Scenario Builder (What-If Analysis & Custom Inputs)",
                "📂 Historical Record Inspection (Ingest Actual Delhi Observations)",
            ],
            horizontal=True,
        )

    st.markdown("---")

    # Storage for active feature dictionary
    active_features: dict[str, float] = {}
    actual_historical_demand: float | None = None
    input_source_desc = ""

    # =========================================================================
    # MODE 1: INTERACTIVE SCENARIO BUILDER (WHAT-IF)
    # =========================================================================
    if "Interactive" in input_mode:
        st.markdown('<div class="section-title">🎛️ Step 1: Configure Your 17 Input Parameters</div>', unsafe_allow_html=True)

        # Quick Preset Selector
        preset_cols = st.columns([2, 1])
        with preset_cols[0]:
            chosen_preset = st.selectbox(
                "⚡ Load Quick Scenario Preset (Pre-fills all 17 features with realistic Delhi conditions)",
                ["-- Custom Manual Input --"] + list(SCENARIO_PRESETS.keys()),
            )
        with preset_cols[1]:
            auto_lags = st.checkbox(
                "⚡ Auto-estimate 4 Demand Lags",
                value=True,
                help="Automatically computes realistic 5-min, 1-hr, 24-hr, and 7-day lags based on your chosen date, time, and weather.",
            )

        preset_data = SCENARIO_PRESETS.get(chosen_preset, {})
        if chosen_preset != "-- Custom Manual Input --":
            st.info(f"**Preset Context:** {preset_data.get('desc', '')}")

        # Three logical feature groups: Temporal, Weather, Lags
        col_temp, col_weath, col_lags = st.columns(3)

        # ----------------- Group 1: Temporal & Calendar -----------------
        with col_temp:
            st.markdown('<div class="feature-group-title">📅 1. Date & Time Factors (7 Features)</div>', unsafe_allow_html=True)
            default_date = date(
                preset_data.get("year", 2024),
                preset_data.get("month", 6),
                preset_data.get("day", 19),
            )
            input_date = st.date_input("Date", value=default_date)

            default_time = time(preset_data.get("hour", 15), preset_data.get("minute", 5))
            input_time = st.time_input("Dispatch Time", value=default_time, step=300)

            # Extract calendar features
            f_year = float(input_date.year)
            f_month = float(input_date.month)
            f_day = float(input_date.day)
            f_hour = float(input_time.hour)
            f_minute = float(input_time.minute - (input_time.minute % 5))  # Align to 5-min grid
            f_day_of_week = float(input_date.weekday())
            f_is_weekend = float(1 if f_day_of_week in [5, 6] else 0)

            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            st.caption(
                f"**Computed Calendar Features:**<br>"
                f"• Day: **{day_names[int(f_day_of_week)]}** (`day_of_week={int(f_day_of_week)}`)<br>"
                f"• Type: **{'Weekend' if f_is_weekend else 'Weekday'}** (`is_weekend={int(f_is_weekend)}`)<br>"
                f"• Dispatch Slot: **{int(f_hour):02d}:{int(f_minute):02d}** (5-minute interval)",
                unsafe_allow_html=True,
            )

        # ----------------- Group 2: Atmospheric Weather -----------------
        with col_weath:
            st.markdown('<div class="feature-group-title">🌡️ 2. Weather Conditions (6 Features)</div>', unsafe_allow_html=True)

            f_temp = st.slider(
                "Temperature (°C)",
                min_value=0.0,
                max_value=50.0,
                value=float(preset_data.get("temp", 44.5)),
                step=0.5,
                help=FEATURE_INFO["temp"]["desc"],
            )

            f_dwpt = st.slider(
                "Dew Point (°C)",
                min_value=-10.0,
                max_value=35.0,
                value=float(preset_data.get("dwpt", 19.5)),
                step=0.5,
                help=FEATURE_INFO["dwpt"]["desc"],
            )

            f_rhum = st.slider(
                "Relative Humidity (%)",
                min_value=5.0,
                max_value=100.0,
                value=float(preset_data.get("rhum", 32.0)),
                step=1.0,
                help=FEATURE_INFO["rhum"]["desc"],
            )

            heat_idx = calculate_heat_index(f_temp, f_rhum)

            f_wspd = st.number_input(
                "Wind Speed (km/h)",
                min_value=0.0,
                max_value=75.0,
                value=float(preset_data.get("wspd", 14.0)),
                step=1.0,
                help=FEATURE_INFO["wspd"]["desc"],
            )

            f_wdir = st.number_input(
                "Wind Direction (°)",
                min_value=0.0,
                max_value=360.0,
                value=float(preset_data.get("wdir", 290.0)),
                step=5.0,
                help=FEATURE_INFO["wdir"]["desc"],
            )

            f_pres = st.number_input(
                "Surface Pressure (hPa)",
                min_value=970.0,
                max_value=1035.0,
                value=float(preset_data.get("pres", 998.0)),
                step=0.5,
                help=FEATURE_INFO["pres"]["desc"],
            )

            st.caption(
                f"**Atmospheric Context:**<br>"
                f"• Apparent Heat Index: **{heat_idx}°C**<br>"
                f"• Wind Cardinal: **{get_wind_direction_label(f_wdir)}**",
                unsafe_allow_html=True,
            )

        # ----------------- Group 3: Grid Demand Lags -----------------
        with col_lags:
            st.markdown('<div class="feature-group-title">⚡ 3. Historical Demand Lags (4 Features)</div>', unsafe_allow_html=True)

            if auto_lags:
                # If preset specified lags, use them; otherwise estimate from time/temp
                if chosen_preset != "-- Custom Manual Input --":
                    estimated = {
                        "demand_lag_5min": preset_data["demand_lag_5min"],
                        "demand_lag_1hour": preset_data["demand_lag_1hour"],
                        "demand_lag_24hour": preset_data["demand_lag_24hour"],
                        "demand_lag_7day": preset_data["demand_lag_7day"],
                    }
                else:
                    estimated = estimate_lags(int(f_hour), int(f_month), int(f_is_weekend))

                f_lag_5m = estimated["demand_lag_5min"]
                f_lag_1h = estimated["demand_lag_1hour"]
                f_lag_24h = estimated["demand_lag_24hour"]
                f_lag_7d = estimated["demand_lag_7day"]

                st.success("⚡ Lags auto-estimated for this time & season:")
                lag_c1, lag_c2 = st.columns(2)
                with lag_c1:
                    st.metric("5-Min Lag", f"{f_lag_5m:,.0f} MW", help="Short-term grid inertia (t-5m)")
                    st.metric("24-Hr Lag", f"{f_lag_24h:,.0f} MW", help="Yesterday's diurnal rhythm (t-24h)")
                with lag_c2:
                    st.metric("1-Hour Lag", f"{f_lag_1h:,.0f} MW", help="Hourly continuation ramp (t-1h)")
                    st.metric("7-Day Lag", f"{f_lag_7d:,.0f} MW", help="Weekly periodic cycle (t-7d)")
            else:
                f_lag_5m = st.number_input(
                    "Demand 5-Min Ago (MW)",
                    min_value=1000.0,
                    max_value=9500.0,
                    value=float(preset_data.get("demand_lag_5min", 8500.0)),
                    step=10.0,
                    help="Immediate grid momentum (t − 5 min)",
                )
                f_lag_1h = st.number_input(
                    "Demand 1-Hour Ago (MW)",
                    min_value=1000.0,
                    max_value=9500.0,
                    value=float(preset_data.get("demand_lag_1hour", 8300.0)),
                    step=10.0,
                    help="Hourly demand level (t − 1 hr)",
                )
                f_lag_24h = st.number_input(
                    "Demand 24-Hours Ago (MW)",
                    min_value=1000.0,
                    max_value=9500.0,
                    value=float(preset_data.get("demand_lag_24hour", 8100.0)),
                    step=10.0,
                    help="Yesterday's demand at this exact time (t − 24 hr)",
                )
                f_lag_7d = st.number_input(
                    "Demand 7-Days Ago (MW)",
                    min_value=1000.0,
                    max_value=9500.0,
                    value=float(preset_data.get("demand_lag_7day", 7800.0)),
                    step=10.0,
                    help="Last week's demand at this exact time (t − 7 days)",
                )

        # Assemble full 17 features
        active_features = {
            "year": f_year,
            "month": f_month,
            "day": f_day,
            "hour": f_hour,
            "minute": f_minute,
            "day_of_week": f_day_of_week,
            "is_weekend": f_is_weekend,
            "demand_lag_5min": f_lag_5m,
            "demand_lag_1hour": f_lag_1h,
            "demand_lag_24hour": f_lag_24h,
            "demand_lag_7day": f_lag_7d,
            "temp": f_temp,
            "dwpt": f_dwpt,
            "rhum": f_rhum,
            "wdir": f_wdir,
            "wspd": f_wspd,
            "pres": f_pres,
        }
        input_source_desc = f"Custom What-If Scenario ({chosen_preset})"

    # =========================================================================
    # MODE 2: HISTORICAL RECORD INSPECTION
    # =========================================================================
    else:
        st.markdown('<div class="section-title">📂 Step 1: Pick an Actual Historical Observation from Delhi Dataset</div>', unsafe_allow_html=True)

        sample_choice = st.selectbox(
            "Select a Notable Historical Delhi Event or Custom Date:",
            ["-- Choose from Curated Highlights --"] + [h["title"] for h in HISTORICAL_HIGHLIGHTS] + ["📅 Custom Date & Time Picker"],
        )

        selected_dt: pd.Timestamp | None = None

        if sample_choice == "📅 Custom Date & Time Picker":
            default_ts = latest_lag_complete_timestamp(df_history)
            c_d, c_t = st.columns(2)
            sel_date = c_d.date_input(
                "Historical Date",
                value=default_ts.date(),
                min_value=df_history["datetime"].min().date(),
                max_value=df_history["datetime"].max().date(),
            )
            sel_time = c_t.time_input("Historical Time", value=time(default_ts.hour, default_ts.minute), step=300)
            selected_dt = pd.Timestamp(datetime.combine(sel_date, sel_time))
        elif sample_choice != "-- Choose from Curated Highlights --":
            match_highlight = next(h for h in HISTORICAL_HIGHLIGHTS if h["title"] == sample_choice)
            selected_dt = pd.Timestamp(match_highlight["timestamp"])
            st.info(f"**Event Detail:** {match_highlight['description']}")
        else:
            selected_dt = pd.Timestamp("2024-06-19 15:05:00")  # Default to all-time peak

        # Look up in history
        history_idx = df_history.set_index("datetime")
        if selected_dt in history_idx.index:
            row_hist = history_idx.loc[selected_dt]
            actual_historical_demand = float(row_hist["Power demand"])
            weather_hist = row_hist[WEATHER_COLUMNS]
            feat_df = make_feature_row(selected_dt, history_idx["Power demand"], weather_hist)
            active_features = {col: float(feat_df[col].iloc[0]) for col in FEATURE_COLUMNS}
            input_source_desc = f"Historical Record: {selected_dt.strftime('%Y-%m-%d %H:%M:%S')}"
            st.success(f"✅ Ingested exact historical observation for **{selected_dt:%d %b %Y, %H:%M}**. Actual Recorded Demand: **{actual_historical_demand:,.2f} MW**.")
        else:
            st.warning(f"Timestamp {selected_dt} does not exist in the 5-minute historical dataset. Falling back to nearest valid timestamp.")
            nearest_dt = df_history.iloc[(df_history["datetime"] - selected_dt).abs().argsort()[:1]]["datetime"].iloc[0]
            row_hist = history_idx.loc[nearest_dt]
            actual_historical_demand = float(row_hist["Power demand"])
            feat_df = make_feature_row(nearest_dt, history_idx["Power demand"], row_hist[WEATHER_COLUMNS])
            active_features = {col: float(feat_df[col].iloc[0]) for col in FEATURE_COLUMNS}
            input_source_desc = f"Nearest Historical Record: {nearest_dt.strftime('%Y-%m-%d %H:%M:%S')}"

    # =========================================================================
    # STEP 2: PREDICTION INFERENCE & OUTPUT CONSOLE
    # =========================================================================
    st.markdown("---")
    st.markdown('<div class="section-title">⚡ Step 2: Compute AI Forecast for the Configured Inputs</div>', unsafe_allow_html=True)

    predict_btn = st.button("🚀 Predict Electricity Demand Now", type="primary", use_container_width=True)

    # Convert active features to model DataFrame
    feature_row_df = make_custom_feature_row(active_features)

    # Run inference for both models to provide comparative clarity
    pred_primary: float | None = None
    pred_linear: float | None = None
    pred_rf: float | None = None

    try:
        if is_model_available("linear_regression"):
            pred_linear = predict_from_features("linear_regression", feature_row_df)
        if is_model_available("random_forest"):
            pred_rf = predict_from_features("random_forest", feature_row_df)

        pred_primary = pred_linear if primary_key == "linear_regression" else (pred_rf or pred_linear)
    except Exception as exc:
        st.error(f"Inference error: {exc}")
        return

    # =========================================================================
    # STEP 3: RICH MULTI-DIMENSIONAL OUTPUT DISPLAY
    # =========================================================================
    st.markdown('<div class="section-title">📊 Step 3: Forecast Output & Grid Stress Analysis</div>', unsafe_allow_html=True)

    if pred_primary is not None:
        regime = classify_demand_regime(pred_primary)

        # Primary Result Hero Display
        st.markdown(
            f"""
            <div class="prediction-banner">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <div style="text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1px; color: #94a3b8; font-weight: 700;">
                            Forecast Output ({model_choice})
                        </div>
                        <div class="prediction-val">
                            {pred_primary:,.2f} <span style="font-size: 1.8rem; color: #e2e8f0;">MW</span>
                        </div>
                        <div style="font-size: 1.15rem; color: #cbd5e1; margin-top: 4px;">
                            Equivalent to <b>{pred_primary / 1000.0:.2f} GW</b> of electrical load
                        </div>
                    </div>
                    <div style="text-align: right; margin-top: 10px;">
                        <span class="badge-chip" style="background-color: {regime['color']}22; color: {regime['color']}; border: 1px solid {regime['color']}; font-size: 0.95rem; padding: 6px 16px;">
                            {regime['badge']}
                        </span>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 8px;">
                            Grid Stress Level: <b style="color: {regime['color']}">{regime['stress']}</b>
                        </div>
                    </div>
                </div>
                <hr style="border-color: #334155; margin: 18px 0 14px 0;" />
                <div style="font-size: 0.95rem; color: #e2e8f0;">
                    🛡️ <b>Operational Advisory:</b> {regime['advice']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gauge & Multi-Model Comparative Output
        out_c1, out_c2 = st.columns([1.1, 1.4])

        with out_c1:
            st.plotly_chart(gauge_chart(pred_primary), use_container_width=True)

        with out_c2:
            st.markdown("### 🤖 Multi-Model Comparison & Benchmarks")

            bench_c1, bench_c2 = st.columns(2)
            with bench_c1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-val" style="color: #0284c7;">{pred_linear:,.1f} MW</div>
                        <div class="metric-lbl">Linear Regression</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with bench_c2:
                rf_display = f"{pred_rf:,.1f} MW" if pred_rf is not None else "Unavailable"
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-val" style="color: #10b981;">{rf_display}</div>
                        <div class="metric-lbl">Random Forest</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Difference between models
            if pred_linear is not None and pred_rf is not None:
                diff_mw = pred_rf - pred_linear
                diff_pct = (diff_mw / pred_linear) * 100.0
                st.caption(
                    f"**Model Divergence:** Random Forest is **{diff_mw:+,.1f} MW ({diff_pct:+.2f}%)** relative to Linear Regression."
                )

            # Historical Actual vs Prediction Comparison (if ground truth available)
            if actual_historical_demand is not None:
                err_mw = pred_primary - actual_historical_demand
                err_pct = (err_mw / actual_historical_demand) * 100.0
                acc_pct = max(0.0, 100.0 - abs(err_pct))
                st.markdown("#### 🎯 Ground-Truth Validation (Actual vs Predicted)")
                v_c1, v_c2, v_c3 = st.columns(3)
                v_c1.metric("Actual Recorded Demand", f"{actual_historical_demand:,.2f} MW")
                v_c2.metric("Prediction Error", f"{err_mw:+,.2f} MW", delta=f"{err_pct:+.2f}%", delta_color="inverse")
                v_c3.metric("Prediction Accuracy", f"{acc_pct:.2f}%")

            # Reference to Delhi Record Peak
            delta_to_record = 8631.53 - pred_primary
            st.info(
                f"Delhi All-Time Record Peak is **8,631.53 MW** (June 19, 2024). "
                f"This forecast is **{abs(delta_to_record):,.1f} MW {'below' if delta_to_record >= 0 else 'above'}** the historical record peak."
            )

    # =========================================================================
    # STEP 4: COMPLETE 17-FEATURE INPUT VERIFICATION TABLE
    # =========================================================================
    st.markdown("---")
    st.markdown('<div class="section-title">📋 Step 4: Complete Input Data Verification Table (All 17 Features)</div>', unsafe_allow_html=True)
    st.write(
        "Below is the exact list of 17 input variables passed into the model to produce the forecast above. "
        "Every single input is fully visible and verifiable."
    )

    table_rows = []
    for col in FEATURE_COLUMNS:
        info = FEATURE_INFO[col]
        val = active_features[col]
        # Format display value nicely
        if col in ["year", "month", "day", "hour", "minute", "day_of_week", "is_weekend"]:
            disp_val = f"{int(val)}"
        else:
            disp_val = f"{val:,.2f}"

        table_rows.append({
            "Feature Key": col,
            "Human Name": info["label"],
            "Category": info["category"],
            "Configured Value": disp_val,
            "Unit": info["unit"],
            "Operational Purpose": info["desc"],
        })

    input_df = pd.DataFrame(table_rows)

    # Group by category in tabs for clarity
    t_cat1, t_cat2, t_cat3, t_all = st.tabs([
        "🌡️ Atmospheric Weather",
        "⚡ Demand Lags (Momentum)",
        "📅 Temporal / Calendar",
        "📄 All 17 Features",
    ])

    with t_cat1:
        st.dataframe(
            input_df[input_df["Category"] == "Atmospheric Weather"][["Human Name", "Configured Value", "Unit", "Operational Purpose"]],
            use_container_width=True,
            hide_index=True,
        )
    with t_cat2:
        st.dataframe(
            input_df[input_df["Category"] == "Grid Demand Momentum (Lags)"][["Human Name", "Configured Value", "Unit", "Operational Purpose"]],
            use_container_width=True,
            hide_index=True,
        )
    with t_cat3:
        st.dataframe(
            input_df[input_df["Category"] == "Temporal / Calendar"][["Human Name", "Configured Value", "Unit", "Operational Purpose"]],
            use_container_width=True,
            hide_index=True,
        )
    with t_all:
        st.dataframe(input_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 2: PROJECT OVERVIEW (HOME)
# -----------------------------------------------------------------------------
def page_home(df_history: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>⚡ Delhi State Power System | AI Demand Projection</h1>
            <p>5-Minute Resolution Electricity Demand Forecasting & Peak Load Scenario Modeling (2021–2024)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    start, end = df_history.datetime.min(), df_history.datetime.max()
    interval = df_history.datetime.diff().dropna().mode().iloc[0]

    # Executive KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Dataset Time Span", f"{start:%b %Y} – {end:%b %Y}")
    kpi2.metric("Total 5-Min Records", f"{len(df_history):,}")
    kpi3.metric("Sampling Interval", str(interval))
    kpi4.metric("Average Demand", f"{df_history['Power demand'].mean():,.1f} MW")
    kpi5.metric("Delhi All-Time Peak", f"{df_history['Power demand'].max():,.2f} MW")

    st.markdown("---")

    # Workflow cards
    st.markdown("### 🛠️ End-to-End AI Forecasting Architecture")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="custom-card">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">🧹</div>
                <h4>1. Data Ingestion & Cleaning</h4>
                <p style="font-size: 0.85rem; color: #64748b;">
                    5-minute interval power demand merged with weather stations. Missing timestamps chronologically addressed without leakage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="custom-card">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">⚙️</div>
                <h4>2. Causal Feature Engineering</h4>
                <p style="font-size: 0.85rem; color: #64748b;">
                    17 causal features: calendar indicators, exact timestamp lags (5m, 1h, 24h, 7d), and 6 atmospheric variables.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="custom-card">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">🤖</div>
                <h4>3. Dual Model Training</h4>
                <p style="font-size: 0.85rem; color: #64748b;">
                    Linear Regression baseline and 100-tree Random Forest Regressor trained on chronological split (80% train, 20% test).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="custom-card">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">🎯</div>
                <h4>4. Peak Load Projection</h4>
                <p style="font-size: 0.85rem; color: #64748b;">
                    High-demand regime evaluation (>95th & >99th percentile), extreme heatwave simulation, and recursive future forecasting.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Recent Trend Overview
    st.markdown("### 📈 Recent Delhi Electricity Demand Trajectory")
    recent = df_history.tail(2016)  # Last ~7 days
    st.plotly_chart(
        line_chart(recent, "Delhi Electricity Demand (Latest 7 Days)", ["Power demand"], {"Power demand": "Demand"}),
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# PAGE 3: DATA EXPLORER
# -----------------------------------------------------------------------------
def page_explorer(df_history: pd.DataFrame) -> None:
    st.markdown('<div class="app-header"><h1>📊 Historical Data Explorer</h1><p>Analyze Delhi electricity load profiles, diurnal curves, and weather dependencies.</p></div>', unsafe_allow_html=True)

    min_d, max_d = df_history.datetime.min().date(), df_history.datetime.max().date()
    dates = st.date_input("Filter Date Range", value=(date(2024, 6, 1), date(2024, 6, 30)), min_value=min_d, max_value=max_d)

    if len(dates) != 2:
        st.info("Please select both a start and end date.")
        return

    selected = df_history[df_history.datetime.dt.date.between(*dates)]
    if selected.empty:
        st.warning("No records found in this range.")
        return

    st.plotly_chart(
        line_chart(selected, f"Electricity Demand Profile ({dates[0]:%d %b %Y} to {dates[1]:%d %b %Y})", ["Power demand"], {"Power demand": "Demand"}),
        use_container_width=True,
    )

    # 3 Summary Charts: Hourly, Day of Week, Monthly
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(
        px.bar(
            selected.groupby(selected.datetime.dt.hour)["Power demand"].mean().reset_index(name="Demand"),
            x="datetime", y="Demand",
            title="<b>Average Load by Hour of Day</b>",
            labels={"datetime": "Hour", "Demand": f"Demand ({demand_unit()})"},
            color="Demand", color_continuous_scale="Viridis",
        ),
        use_container_width=True,
    )
    c2.plotly_chart(
        px.bar(
            selected.groupby(selected.datetime.dt.dayofweek)["Power demand"].mean().reset_index(name="Demand"),
            x="datetime", y="Demand",
            title="<b>Average Load by Day of Week</b>",
            labels={"datetime": "Day (0=Mon, 6=Sun)", "Demand": f"Demand ({demand_unit()})"},
            color="Demand", color_continuous_scale="Teal",
        ),
        use_container_width=True,
    )
    c3.plotly_chart(
        px.bar(
            df_history.groupby(df_history.datetime.dt.month)["Power demand"].mean().reset_index(name="Demand"),
            x="datetime", y="Demand",
            title="<b>All-Time Average Load by Month</b>",
            labels={"datetime": "Month", "Demand": f"Demand ({demand_unit()})"},
            color="Demand", color_continuous_scale="Plasma",
        ),
        use_container_width=True,
    )

    # Temperature vs Demand U-Curve
    st.markdown("### 🌡️ Temperature vs Electricity Demand (Cooling & Heating U-Curve)")
    sample_corr = selected.sample(min(len(selected), 4000), random_state=42)
    fig_temp = px.scatter(
        sample_corr,
        x="temp",
        y="Power demand",
        color="hour" if "hour" in sample_corr.columns else None,
        title="<b>Temperature (°C) vs Power Demand (MW)</b>",
        labels={"temp": "Temperature (°C)", "Power demand": "Demand (MW)"},
        opacity=0.6,
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # Top Historical Peaks
    peaks = selected.nlargest(10, "Power demand")[["datetime", "Power demand", "temp", "rhum"]].copy()
    peaks["date"] = peaks.datetime.dt.date
    peaks["time"] = peaks.datetime.dt.time
    st.subheader("Top 10 Peak Demand Records in Selected Period")
    st.dataframe(peaks[["date", "time", "Power demand", "temp", "rhum"]], use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
def page_performance() -> None:
    st.markdown('<div class="app-header"><h1>🎯 Machine Learning Model Performance</h1><p>Evaluation on chronological hold-out test set (March – December 2024, 70,825 intervals).</p></div>', unsafe_allow_html=True)

    lr_meta = load_json("outputs/linear_regression/linear_regression_metrics.json")
    rf_meta = load_json("outputs/random_forest/random_forest_metrics.json")
    comp_meta = load_json("outputs/model_comparison/model_comparison_summary.json")

    # Metrics Summary Table
    table = pd.DataFrame([
        {"Model": "Linear Regression (Baseline)", **lr_meta["test_metrics"]},
        {"Model": "Random Forest Regressor (100 Trees)", **rf_meta["test_metrics"]},
    ]).rename(columns={"R2": "R²", "MAE": "MAE (MW)", "RMSE": "RMSE (MW)"})

    st.dataframe(table, hide_index=True, use_container_width=True)

    # Regime Analysis from Summary JSON
    st.markdown("### ⚡ Performance Breakdown Across Demand Regimes")
    regime_data = comp_meta.get("regime_analysis", {})
    if regime_data:
        r_rows = []
        for reg_key, reg_info in regime_data.items():
            r_rows.append({
                "Demand Regime": reg_key.replace("_", " ").title(),
                "Threshold Definition": reg_info["definition"],
                "Test Records": f"{reg_info['observation_count']:,} ({reg_info['percentage_of_test_set']}%)",
                "Linear Regression MAE": f"{reg_info['linear_regression_mae_mw']:.2f} MW",
                "Random Forest MAE": f"{reg_info['random_forest_mae_mw']:.2f} MW",
            })
        st.dataframe(pd.DataFrame(r_rows), hide_index=True, use_container_width=True)

    # Actual vs Predicted Test Curves
    model_sel = st.selectbox("Inspect Test Predictions For Model:", ["linear_regression", "random_forest"])
    pred = load_predictions(model_sel)
    pred["residual"] = pred.actual_demand - pred.predicted_demand

    st.plotly_chart(
        line_chart(
            pred.iloc[1000:2500],
            f"Actual vs Predicted Demand — {model_sel.replace('_', ' ').title()} (Zoomed Test Segment)",
            ["actual_demand", "predicted_demand"],
            {"actual_demand": "Actual", "predicted_demand": "Predicted"},
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        px.histogram(
            pred,
            x="residual",
            nbins=100,
            title=f"<b>Residual Error Distribution — {model_sel.replace('_', ' ').title()}</b>",
            labels={"residual": f"Actual − Predicted Error ({demand_unit()})"},
            color_discrete_sequence=["#0284c7"],
        ),
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# PAGE 5: FUTURE DEMAND FORECAST
# -----------------------------------------------------------------------------
def run_forecast(key: str, df_history: pd.DataFrame, steps: int) -> pd.DataFrame:
    interval = df_history.datetime.diff().dropna().mode().iloc[0]
    start = df_history.datetime.max() + interval
    recursive: dict[pd.Timestamp, float] = {}
    rows = []
    for timestamp in pd.date_range(start, periods=steps, freq=interval):
        value = predict_at(key, timestamp, df_history, recursive)
        recursive[timestamp] = value
        rows.append((timestamp, value))
    return pd.DataFrame(rows, columns=["datetime", "predicted_demand"])


def page_forecast(df_history: pd.DataFrame) -> None:
    st.markdown('<div class="app-header"><h1>🔮 Recursive Future Demand Forecasting</h1><p>Multi-step forward projection using recursive lag propagation.</p></div>', unsafe_allow_html=True)

    choice = st.selectbox("Forecasting Engine", ["Linear Regression", "Random Forest"])
    key = "linear_regression" if choice == "Linear Regression" else "random_forest"
    interval = df_history.datetime.diff().dropna().mode().iloc[0]

    horizons = {
        "Next 24 Hours (288 steps)": int(pd.Timedelta(days=1) / interval),
        "Next 7 Days (2,016 steps)": int(pd.Timedelta(days=7) / interval),
        "Next 30 Days (8,640 steps)": int(pd.Timedelta(days=30) / interval),
    }
    horizon_label = st.selectbox("Forecast Horizon", list(horizons.keys()))

    st.warning("⚠️ **Recursive Forecast Notice:** Past predictions are fed recursively into future lag features (t-5m, t-1h, t-24h, t-7d). Weather persistence assumes the latest recorded atmospheric row.")

    if st.button("🚀 Generate Recursive Forecast", type="primary"):
        with st.spinner("Propagating recursive predictions across future dispatch intervals..."):
            forecast_df = run_forecast(key, df_history, horizons[horizon_label])

        context = df_history.tail(min(len(df_history), int(pd.Timedelta(days=7) / interval)))
        fig = line_chart(context, f"Historical Context and Future Forecast ({horizon_label})", ["Power demand"], {"Power demand": "Historical Actual"})
        fig.add_scatter(x=forecast_df.datetime, y=forecast_df.predicted_demand, mode="lines", name="AI Forecast", line=dict(color="#f59e0b", width=2.2))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "📥 Download Forecast Results (CSV)",
            forecast_df.to_csv(index=False).encode(),
            "delhi_power_demand_forecast.csv",
            "text/csv",
        )
        st.session_state["forecast"] = forecast_df
        st.session_state["forecast_model"] = choice


# -----------------------------------------------------------------------------
# PAGE 6: PEAK DEMAND ANALYSIS
# -----------------------------------------------------------------------------
def page_peaks(df_history: pd.DataFrame) -> None:
    st.markdown('<div class="app-header"><h1>🏔️ Peak Demand & Critical Event Analysis</h1><p>Deep-dive into Delhi power system historical records and forecast-window peaks.</p></div>', unsafe_allow_html=True)

    actual_max_row = df_history.loc[df_history["Power demand"].idxmax()]
    st.subheader("🔥 Delhi Historical Record Peak")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Demand", f"{actual_max_row['Power demand']:,.2f} MW")
    c2.metric("Date", str(actual_max_row.datetime.date()))
    c3.metric("Time", str(actual_max_row.datetime.time()))
    c4.metric("Ambient Temperature", f"{actual_max_row['temp']:.1f}°C")

    # Peak hour analysis
    peak_hr_csv = ROOT / "outputs" / "peak_analysis" / "peak_hour_analysis.csv"
    if peak_hr_csv.exists():
        st.markdown("### 🕒 Peak Hour Occurrence Distribution")
        pk_df = pd.read_csv(peak_hr_csv)
        st.plotly_chart(
            px.bar(
                pk_df,
                x="hour",
                y="actual_peak_count",
                title="<b>Frequency of Daily Peak Load by Hour of Day</b>",
                labels={"hour": "Hour of Day (0–23)", "actual_peak_count": "Peak Occurrences"},
                color="actual_peak_count",
                color_continuous_scale="Reds",
            ),
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# PAGE 7: ABOUT & METHODOLOGY
# -----------------------------------------------------------------------------
def page_about() -> None:
    st.markdown('<div class="app-header"><h1>📖 Methodology, Features & Validation</h1><p>Comprehensive technical documentation of the Delhi AI power forecasting pipeline.</p></div>', unsafe_allow_html=True)

    st.markdown("### 📐 All 17 Input Features & Mathematical Designations")
    rows = []
    for f in FEATURE_COLUMNS:
        info = FEATURE_INFO[f]
        rows.append({
            "Feature": f,
            "Name": info["label"],
            "Category": info["category"],
            "Unit": info["unit"],
            "Delhi Range": f"{info['min']} to {info['max']}",
            "Physical Grid Meaning": info["desc"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("### 🛡️ Strictly Causal Lag Construction (Zero Data Leakage)")
    st.write(
        "All 4 demand lags are constructed using exact temporal timestamps (`t - 5m`, `t - 1h`, `t - 24h`, `t - 7d`). "
        "The pipeline strictly avoids simple position-based index shifting (`shift(n)`), preventing corruption across irregular gaps. "
        "No future information or non-causal moving averages are used."
    )


# -----------------------------------------------------------------------------
# MAIN ROUTER
# -----------------------------------------------------------------------------
if page == "⚡ Demand Prediction & What-If":
    page_prediction_console(history)
elif page == "🏠 Project Overview":
    page_home(history)
elif page == "📊 Data Explorer":
    page_explorer(history)
elif page == "🎯 Model Performance":
    page_performance()
elif page == "🔮 Future Demand Forecast":
    page_forecast(history)
elif page == "🏔️ Peak Demand Analysis":
    page_peaks(history)
else:
    page_about()
