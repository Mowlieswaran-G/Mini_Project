"""
AI-Based Electricity Demand Projection — Delhi Power System
Power Predict AI | Multi-Model Electricity Demand & Peak Forecasting System
"""

from __future__ import annotations

import sys
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
for p in (str(ROOT_DIR), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.utils.data import (
        ROOT, find_model, is_model_available, is_lstm_file_present, is_torch_installed,
        load_history, load_json, load_model, load_predictions, HISTORICAL_HIGHLIGHTS
    )
    from app.utils.features import (
        FEATURE_COLUMNS, FEATURE_INFO, LAGS, SCENARIO_PRESETS, PHYSICAL_BOUNDS,
        WEATHER_COLUMNS, calculate_heat_index, classify_demand_regime,
        estimate_lags, get_wind_direction_label, validate_weather_inputs,
        fetch_optional_delhi_weather, make_custom_feature_row, make_feature_row
    )
except ImportError:
    from utils.data import (
        ROOT, find_model, is_model_available, is_lstm_file_present, is_torch_installed,
        load_history, load_json, load_model, load_predictions, HISTORICAL_HIGHLIGHTS
    )
    from utils.features import (
        FEATURE_COLUMNS, FEATURE_INFO, LAGS, SCENARIO_PRESETS, PHYSICAL_BOUNDS,
        WEATHER_COLUMNS, calculate_heat_index, classify_demand_regime,
        estimate_lags, get_wind_direction_label, validate_weather_inputs,
        fetch_optional_delhi_weather, make_custom_feature_row, make_feature_row
    )


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Power Predict AI | Delhi Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body {
    background-color: #0f172a !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

[data-testid="stAppViewContainer"] { background-color: #0f172a !important; }

header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 10 !important;
    height: 48px !important;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: none !important;
    z-index: 20 !important;
}

section[data-testid="stSidebar"] > div:first-child {
    background: #ffffff !important;
    border-radius: 20px !important;
    margin: 12px !important;
    height: calc(100vh - 24px) !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
    padding: 20px 16px !important;
    display: flex !important;
    flex-direction: column !important;
    overflow-y: auto !important;
}

.block-container {
    background: #ffffff !important;
    border-radius: 24px !important;
    padding: 28px 36px 36px 36px !important;
    margin: 12px 12px 12px 0px !important;
    min-height: calc(100vh - 24px) !important;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25) !important;
}

.ne-dash-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 14px;
    margin-bottom: 18px;
    border-bottom: 1.5px solid #f1f5f9;
}
.ne-dash-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.30rem;
    font-weight: 800;
    color: #0f172a;
}
.ne-dash-meta {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
}

/* KPI Cards */
.ne-kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.ne-kpi-card {
    background: #f8fafc;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 14px 16px;
    text-align: left;
}
.ne-kpi-pill {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    color: #ffffff !important;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 0.70rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 6px;
}
.ne-kpi-val {
    font-size: 1.50rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #0f172a;
    line-height: 1.1;
}
.ne-kpi-val span {
    font-size: 0.80rem;
    font-weight: 600;
    color: #64748b;
}
.ne-kpi-sub {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 4px;
}

/* Section Header */
.st2 {
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: #1e3a8a !important;
    border-left: 4px solid #2563eb !important;
    padding-left: 10px !important;
    margin: 22px 0 12px !important;
}

/* Callout Box */
.cb {
    background: #eff6ff !important;
    border-left: 4px solid #2563eb !important;
    border-radius: 0 10px 10px 0 !important;
    padding: 12px 16px !important;
    margin-bottom: 16px !important;
    color: #1e3a8a !important;
    font-size: 0.88rem !important;
    line-height: 1.5 !important;
}

/* Banner */
.ne-result-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%) !important;
    border-radius: 16px !important;
    padding: 22px 26px !important;
    color: #ffffff !important;
    margin-bottom: 20px !important;
    box-shadow: 0 6px 20px rgba(30, 58, 138, 0.25) !important;
}
.ne-result-banner * { color: #ffffff !important; }
.ne-rb-val {
    font-size: 2.4rem;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
}
.ne-rb-val span { font-size: 1.2rem; color: #38bdf8 !important; }

/* Alert Box */
.alert-warning-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 12px 0;
    font-size: 0.86rem;
    color: #92400e;
}
.alert-danger-box {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 12px 0;
    font-size: 0.86rem;
    color: #991b1b;
}

[data-testid="stPlotlyChart"] { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ── Chart Layout Helpers ──────────────────────────────────────────────────────
PALETTE = ["#2563eb", "#ea580c", "#7c3aed", "#059669", "#0891b2", "#d97706"]

def _wl(fig, title: str, h: int = 380):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0, y=0.98, xanchor="left", yanchor="top",
                   font=dict(size=14, color="#1e3a8a")),
        paper_bgcolor="#ffffff", plot_bgcolor="#fafcfb",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#0f172a"),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", title_font=dict(color="#475569")),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", title="Demand (MW)", title_font=dict(color="#475569")),
        height=h, margin=dict(l=14, r=14, t=72, b=22),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ── Load History ──────────────────────────────────────────────────────────────
try:
    history = load_history()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

# ── Top Dashboard Header Bar ──────────────────────────────────────────────────
def render_dash_bar(subtitle: str = "Delhi State Grid Analysis Center"):
    st.markdown(f"""
    <div class="ne-dash-bar">
      <div class="ne-dash-title">
        <span>⚡</span> Power Predict AI
        <span style="font-size:0.75rem;font-weight:600;color:#2563eb;margin-left:8px;background:#eff6ff;padding:3px 10px;border-radius:9999px;">{subtitle}</span>
      </div>
      <div class="ne-dash-meta">
        SLDC Telemetry • 5-Min Intervals • 2021–2024
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 6px 14px;">
      <div style="width:36px;height:36px;background:#2563eb;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#ffffff;font-size:1.1rem;font-weight:bold;">⚡</div>
      <div>
        <div style="font-size:1.05rem;font-weight:800;color:#0f172a;line-height:1.1;">Power Predict AI</div>
        <div style="font-size:0.70rem;color:#2563eb;font-weight:700;">DELHI GRID FORECASTING</div>
      </div>
    </div>
    <div style="font-size:0.70rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;padding:0 6px 8px;">SECTIONS</div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation Menu", [
        "1. 🏠 Project Overview",
        "2. 🧹 Dataset & Data Quality",
        "3. 📊 Exploratory Data Analysis",
        "4. 🤖 Model Comparison",
        "5. 🔮 Demand Forecasting",
        "6. 🏔️ Peak-Demand Analysis",
        "7. 🌦️ Weather Scenario Analysis",
        "8. 🚨 Forecast Alerts & Regimes",
        "9. 🎯 Model Evaluation & Residuals",
        "10. 📥 Downloadable Reports"
    ], index=0, label_visibility="collapsed")

    st.markdown("<hr style='margin:14px 0 10px 0;'>", unsafe_allow_html=True)
    st.markdown("**🤖 Validated AI Models**")
    lr_ok = is_model_available("linear_regression")
    rf_ok = is_model_available("random_forest")
    lstm_ready = is_model_available("lstm")
    lstm_file = is_lstm_file_present()

    st.markdown(f"- **Linear Regression:** {'✅ Ready' if lr_ok else '❌ Missing'}")
    st.markdown(f"- **Random Forest:** {'✅ Ready' if rf_ok else '❌ Missing'}")
    if lstm_ready:
        st.markdown("- **PyTorch LSTM:** ✅ Ready (CPU/GPU)")
    elif lstm_file:
        st.markdown("- **PyTorch LSTM:** ⚠️ Model Ready (PyTorch needed for live inference)")
    else:
        st.markdown("- **PyTorch LSTM:** ❌ Checkpoint Missing")

    st.caption("Delhi State Load Despatch Centre • 70,825 Hold-out Test Observations")

# ==============================================================================
# SECTION 1: PROJECT OVERVIEW
# ==============================================================================
if page.startswith("1."):
    render_dash_bar("Project Overview & System Architecture")

    st.markdown("""
    <div class="ne-kpi-row">
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">⚡ System Goal</div>
        <div class="ne-kpi-val">5-Min <span>Loads</span></div>
        <div class="ne-kpi-sub">Delhi State Grid Telemetry</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">📊 Dataset Records</div>
        <div class="ne-kpi-val">393,440</div>
        <div class="ne-kpi-sub">2021-01 to 2024-12 (4 Years)</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🔥 Highest Test Peak</div>
        <div class="ne-kpi-val" style="color:#ea580c">8,631.5 <span>MW</span></div>
        <div class="ne-kpi-sub">19 June 2024 at 15:05 IST</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🎯 Baseline Test MAE</div>
        <div class="ne-kpi-val">25.24 <span>MW</span></div>
        <div class="ne-kpi-sub">Linear Regression (0.9990 R²)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="st2">Project Executive Summary</div>', unsafe_allow_html=True)
    st.markdown("""
    This project provides a scientifically validated electricity demand forecasting and peak load projection system
    for the National Capital Territory of Delhi using 4 full years (2021–2024) of 5-minute dispatch telemetry.
    
    Delhi experiences severe seasonal load variations:
    - **Summer/Monsoon (May–July):** Sustained heatwaves exceeding **44 °C** and high humidity cause massive air-conditioning loads, driving grid demand past **8,600 MW**.
    - **Winter (Dec–Jan):** Cold spells down to **4 °C** drive domestic space heating and water geyser spikes during early mornings.
    """)

    st.markdown('<div class="st2">Three Validated Models Evaluated</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **1. Linear Regression (Baseline)**
        - Ordinary Least Squares formulation on 17 causal features.
        - Sub-second fit (~0.42 s) with unbounded linear extrapolation.
        - **Test MAE:** 25.24 MW | **R²:** 0.9990 | **Peak Error:** -71.12 MW (-0.82%).
        """)
    with c2:
        st.markdown("""
        **2. Random Forest Regressor**
        - 100 Decision Trees (`max_depth=20`, `min_samples_leaf=2`).
        - Captures non-linear meteorological and humidity interactions.
        - **Test MAE:** 28.17 MW | **R²:** 0.9983 | **Peak Error:** -954.21 MW (-11.05%).
        - *Limitation:* Capped by training leaf upper bounds (~7,746 MW).
        """)
    with c3:
        st.markdown("""
        **3. PyTorch Deep LSTM Network**
        - 2-layer stacked LSTM with sequence lookback window (6 steps = 30 mins).
        - Sequential memory for intra-day inertia and ramp dynamics.
        - **Test MAE:** 27.63 MW | **R²:** 0.9987 | **Peak Error:** -451.86 MW (-5.23%).
        - Cuts Random Forest's peak under-prediction error by more than 50%.
        """)

    st.markdown('<div class="st2">Hardware & Production Environment</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cb">
      💻 <b>Engineered for Resource-Constrained Hardware:</b> The application uses intelligent memory caching
      (<code>@st.cache_resource</code> and <code>@st.cache_data</code>) to ensure fast, responsive performance
      on standard laptop hardware (such as an Intel Core i3 with 8 GB RAM) without repeated model loads.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: DATASET & DATA QUALITY
# ==============================================================================
elif page.startswith("2."):
    render_dash_bar("Dataset Characteristics & Data Quality Verification")

    st.markdown('<div class="st2">Data Pipeline Audit</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cb">
      🛡️ <b>Strict Zero-Leakage Standards:</b>
      1. Contemporaneous moving averages (<code>moving_avg_3</code>) in raw telemetry were dropped to prevent target leakage.
      2. 540 missing wind direction (<code>wdir</code>) readings were imputed strictly via causal forward-fill (<code>ffill</code>).
      3. All 4 autoregressive lags are mapped using <b>exact timestamp lookup</b> (<code>t − 5m</code>, <code>t − 1h</code>, <code>t − 24h</code>, <code>t − 7d</code>). Missing historical slots remain <code>NaN</code> without synthetic interpolation.
      4. 80/20 train-test split was executed chronologically without random shuffling.
    </div>
    """, unsafe_allow_html=True)

    dq_file = ROOT / "outputs" / "data_quality_report.json"
    if dq_file.exists():
        dq = load_json("outputs/data_quality_report.json")
        checks = dq.get("checks", {})
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Observations", f"{checks.get('raw_dataset', {}).get('row_count', 393440):,}")
        c2.metric("Cleaned Observations", f"{checks.get('cleaned_dataset', {}).get('row_count', 393440):,}")
        c3.metric("Train Set (80%)", f"{checks.get('train_test_split', {}).get('train_rows', 283298):,}")
        c4.metric("Test Set (20%)", f"{checks.get('train_test_split', {}).get('test_rows', 70825):,}")

    st.markdown('<div class="st2">Historical Telemetry Head & Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(history.head(10), width="stretch")

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.markdown("**Demand Summary (MW)**")
        st.dataframe(history[["Power demand"]].describe(), width="stretch")
    with c_s2:
        st.markdown("**Weather Variables Summary**")
        st.dataframe(history[WEATHER_COLUMNS].describe(), width="stretch")

# ==============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# ==============================================================================
elif page.startswith("3."):
    render_dash_bar("Exploratory Data Analysis & Empirical Grid Patterns")

    mn, mx = history.datetime.min().date(), history.datetime.max().date()
    dates = st.date_input("Select Historical Range for Inspection", value=(date(2024, 6, 1), date(2024, 6, 30)), min_value=mn, max_value=mx)
    if len(dates) == 2:
        st_ts, end_ts = pd.Timestamp(dates[0]), pd.Timestamp(dates[1]) + pd.Timedelta(days=1)
        sel = history[(history["datetime"] >= st_ts) & (history["datetime"] < end_ts)]
        if not sel.empty:
            step = max(1, len(sel) // 800)
            chart_data = sel.iloc[::step]
            
            fig = px.line(chart_data, x="datetime", y="Power demand", title="Electricity Demand Profile (MW)",
                          labels={"datetime": "Date & Time", "Power demand": "Demand (MW)"}, color_discrete_sequence=["#2563eb"])
            _wl(fig, f"Delhi Power Demand: {dates[0]} to {dates[1]}")
            st.plotly_chart(fig, width="stretch")

            c1, c2, c3 = st.columns(3)
            with c1:
                hr_avg = sel.groupby(pd.to_datetime(sel["datetime"]).dt.hour)["Power demand"].mean().reset_index()
                hr_avg.columns = ["Hour", "Average Demand (MW)"]
                f_hr = px.bar(hr_avg, x="Hour", y="Average Demand (MW)", title="Diurnal Average Hourly Demand (MW)",
                              labels={"Hour": "Hour of Day (0–23)", "Average Demand (MW)": "Demand (MW)"}, color_discrete_sequence=["#0891b2"])
                _wl(f_hr, "Diurnal Cycle (MW)", 300)
                st.plotly_chart(f_hr, width="stretch")
            with c2:
                dow_avg = sel.groupby(pd.to_datetime(sel["datetime"]).dt.dayofweek)["Power demand"].mean().reset_index()
                dow_avg.columns = ["Day_Num", "Average Demand (MW)"]
                dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                dow_avg["Day of Week"] = dow_avg["Day_Num"].map(lambda d: dow_names[d] if 0 <= d < 7 else str(d))
                f_dow = px.bar(dow_avg, x="Day of Week", y="Average Demand (MW)", title="Day of Week Average Demand (MW)",
                               labels={"Day of Week": "Day of Week", "Average Demand (MW)": "Demand (MW)"}, color_discrete_sequence=["#d97706"])
                _wl(f_dow, "Day of Week (MW)", 300)
                st.plotly_chart(f_dow, width="stretch")
            with c3:
                smp = sel.sample(min(len(sel), 2000), random_state=42)
                f_scat = px.scatter(smp, x="temp", y="Power demand", color="rhum", title="Temperature vs Demand (MW)",
                                    labels={"temp": "Temp (°C)", "Power demand": "Demand (MW)", "rhum": "RH (%)"}, color_continuous_scale="YlOrRd")
                _wl(f_scat, "Temperature vs Demand", 300)
                st.plotly_chart(f_scat, width="stretch")
        else:
            st.warning("⚠️ No historical telemetry records found within the chosen date range.")
    elif len(dates) == 1:
        st.info("ℹ️ Please select an end date in the date picker to display the historical demand profile.")
    else:
        st.warning("⚠️ Please select a valid date range to inspect historical telemetry.")

# ==============================================================================
# SECTION 4: MODEL COMPARISON
# ==============================================================================
elif page.startswith("4."):
    render_dash_bar("Multi-Model Benchmark & Comparative Evaluation")

    st.markdown('<div class="st2">Hold-Out Test Benchmark (70,825 Observations • March to December 2024)</div>', unsafe_allow_html=True)
    comp_file = ROOT / "model_comparison_delhi_powerdemand.csv"
    if comp_file.exists():
        comp_df = pd.read_csv(comp_file, encoding="utf-8-sig")
        st.dataframe(comp_df, width="stretch", hide_index=True)

    st.markdown('<div class="st2">Detailed Model Error Distribution (Residuals)</div>', unsafe_allow_html=True)
    models_to_inspect = ["linear_regression", "random_forest"]
    if (ROOT / "lstm_test_predictions.csv").exists():
        models_to_inspect.append("lstm")

    m_pick = st.selectbox("Select model to inspect error distribution:", models_to_inspect,
                          format_func=lambda x: x.replace("_", " ").title())
    
    preds_df = load_predictions(m_pick)
    residuals = preds_df["actual_demand"] - preds_df["predicted_demand"]
    smp_res = residuals.sample(min(15000, len(residuals)), random_state=42)

    fig_res = px.histogram(smp_res, x=smp_res.values, nbins=100, title=f"Residual Error Distribution (Actual − Predicted MW) — {m_pick.replace('_', ' ').title()}",
                           labels={"x": "Residual Error (MW)"}, color_discrete_sequence=["#2563eb"])
    _wl(fig_res, f"Residual Distribution: {m_pick.replace('_', ' ').title()}")
    fig_res.update_layout(yaxis=dict(title="Frequency"))
    st.plotly_chart(fig_res, width="stretch")

# ==============================================================================
# SECTION 5: DEMAND FORECASTING
# ==============================================================================
elif page.startswith("5."):
    render_dash_bar("Recursive Forward Forecasting Engine")

    st.markdown('<div class="st2">Multi-Step Forecasting Controls</div>', unsafe_allow_html=True)
    col_e, col_h = st.columns(2)
    with col_e:
        engine_opts = ["Linear Regression (Ultra-Fast Linear Extrapolation)", "Random Forest (100 Trees Ensemble)"]
        if is_model_available("lstm"):
            engine_opts.append("PyTorch LSTM (Deep Recurrent Sequence Model)")
        elif is_lstm_file_present():
            st.caption("ℹ️ PyTorch LSTM model file exists on disk. Install `torch` to enable live LSTM inference.")
        eng_choice = st.selectbox("Select Forecasting Model Engine:", engine_opts)
        eng_key = "linear_regression" if "Linear" in eng_choice else ("random_forest" if "Random" in eng_choice else "lstm")

    with col_h:
        horizons = {
            "Next 1 Hour (12 steps of 5-min intervals)": 12,
            "Next 24 Hours (288 steps of 5-min intervals)": 288,
            "Next 7 Days (2,016 steps of 5-min intervals)": 2016
        }
        hz_label = st.selectbox("Select Forecast Horizon:", list(horizons.keys()), index=1)
        steps_count = horizons[hz_label]

    st.markdown("""
    <div class="alert-warning-box">
      ⚠️ <b>Forecast Methodology & Uncertainty Disclaimer:</b>
      Multi-step forward forecasting propagates demand projections recursively: each predicted value is appended to the autoregressive lag buffer.
      <b>Error Accumulation:</b> Forecast prediction intervals widen over longer horizons (e.g. 7 days).
      <b>Weather Assumption:</b> Unless updated with live meteorological NWP data, future weather variables are assumed to persist at the last observed conditions.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Generate Recursive Forecast", type="primary"):
        from time import perf_counter
        t0 = perf_counter()

        with st.spinner(f"Generating recursive forecast for {steps_count} steps..."):
            iv = pd.Timedelta(history["datetime"].diff().dropna().mode().iloc[0])
            st_dt = history["datetime"].max() + iv
            future_dates = pd.date_range(st_dt, periods=steps_count, freq=iv)

            recent_demands = list(history["Power demand"].tail(2016).to_numpy(dtype=float))
            last_weather = history.iloc[-1][WEATHER_COLUMNS].to_numpy(dtype=float)

            model = load_model(eng_key)
            preds = []

            if eng_key == "linear_regression":
                coef = np.asarray(model.coef_, dtype=np.float64)
                intercept = float(model.intercept_)
                f_buf = np.empty(17, dtype=np.float64)
                f_buf[11:17] = last_weather
                for ts in future_dates:
                    f_buf[0] = ts.year; f_buf[1] = ts.month; f_buf[2] = ts.day
                    f_buf[3] = ts.hour; f_buf[4] = ts.minute; f_buf[5] = ts.dayofweek
                    f_buf[6] = 1.0 if ts.dayofweek in (5, 6) else 0.0
                    f_buf[7] = recent_demands[-1]
                    f_buf[8] = recent_demands[-12]
                    f_buf[9] = recent_demands[-288]
                    f_buf[10] = recent_demands[-2016]
                    p = float(np.dot(f_buf, coef) + intercept)
                    preds.append(p)
                    recent_demands.append(p)
            elif eng_key == "random_forest":
                trees = [e.tree_ for e in model.estimators_]
                n_trees = len(trees)
                f_buf = np.empty((1, 17), dtype=np.float32)
                f_buf[0, 11:17] = last_weather
                for ts in future_dates:
                    f_buf[0, 0] = ts.year; f_buf[0, 1] = ts.month; f_buf[0, 2] = ts.day
                    f_buf[0, 3] = ts.hour; f_buf[0, 4] = ts.minute; f_buf[0, 5] = ts.dayofweek
                    f_buf[0, 6] = 1.0 if ts.dayofweek in (5, 6) else 0.0
                    f_buf[0, 7] = recent_demands[-1]
                    f_buf[0, 8] = recent_demands[-12]
                    f_buf[0, 9] = recent_demands[-288]
                    f_buf[0, 10] = recent_demands[-2016]
                    p = float(sum(t.predict(f_buf)[0, 0] for t in trees) / n_trees)
                    preds.append(p)
                    recent_demands.append(p)
            else:
                # Build rolling 6-step sequential buffer for PyTorch LSTM from trailing historical intervals
                seq_buffer = []
                for idx in range(-6, 0):
                    row_hist = history.iloc[idx]
                    ts_h = row_hist["datetime"]
                    f_vec = np.empty(17, dtype=np.float64)
                    f_vec[0] = ts_h.year; f_vec[1] = ts_h.month; f_vec[2] = ts_h.day
                    f_vec[3] = ts_h.hour; f_vec[4] = ts_h.minute; f_vec[5] = ts_h.dayofweek
                    f_vec[6] = 1.0 if ts_h.dayofweek in (5, 6) else 0.0
                    pos = len(recent_demands) + idx
                    f_vec[7] = recent_demands[pos - 1]
                    f_vec[8] = recent_demands[pos - 12]
                    f_vec[9] = recent_demands[pos - 288]
                    f_vec[10] = recent_demands[pos - 2016]
                    f_vec[11:17] = row_hist[WEATHER_COLUMNS].to_numpy(dtype=float)
                    seq_buffer.append(f_vec)

                for ts in future_dates:
                    input_3d = np.array(seq_buffer, dtype=np.float64).reshape(1, 6, 17)
                    p = float(model.predict(input_3d)[0])
                    preds.append(p)
                    recent_demands.append(p)

                    # Construct next step feature vector and causally roll 6-step buffer
                    f_new = np.empty(17, dtype=np.float64)
                    f_new[0] = ts.year; f_new[1] = ts.month; f_new[2] = ts.day
                    f_new[3] = ts.hour; f_new[4] = ts.minute; f_new[5] = ts.dayofweek
                    f_new[6] = 1.0 if ts.dayofweek in (5, 6) else 0.0
                    f_new[7] = recent_demands[-1]
                    f_new[8] = recent_demands[-12]
                    f_new[9] = recent_demands[-288]
                    f_new[10] = recent_demands[-2016]
                    f_new[11:17] = last_weather

                    seq_buffer.pop(0)
                    seq_buffer.append(f_new)

            fc_df = pd.DataFrame({"datetime": future_dates, "predicted_demand_mw": preds})
            dur = perf_counter() - t0
            st.session_state["active_forecast"] = {
                "df": fc_df,
                "model": eng_choice,
                "horizon": hz_label,
                "duration": dur,
                "weather_assumed": f"Persistence of last observed conditions (Temp={last_weather[0]:.1f}°C, RH={last_weather[2]:.0f}%, Pres={last_weather[5]:.1f} hPa)"
            }

    if "active_forecast" in st.session_state:
        fc_res = st.session_state["active_forecast"]
        df_fc = fc_res["df"]

        st.success(f"✅ Generated {len(df_fc):,} forward predictions in {fc_res['duration']:.3f} s using {fc_res['model']}.")
        st.info(f"ℹ️ **Weather Assumption:** {fc_res['weather_assumed']}")

        # Forecast Metrics
        c_f1, c_f2, c_f3, c_f4 = st.columns(4)
        c_f1.metric("Forecast Mean", f"{df_fc['predicted_demand_mw'].mean():,.1f} MW")
        c_f2.metric("Forecast Peak", f"{df_fc['predicted_demand_mw'].max():,.1f} MW")
        c_f3.metric("Forecast Min", f"{df_fc['predicted_demand_mw'].min():,.1f} MW")
        c_f4.metric("Forecast Steps", f"{len(df_fc):,}")

        # Forecast Plot
        ctx_len = min(288, len(history))
        ctx_df = history.tail(ctx_len)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=ctx_df["datetime"], y=ctx_df["Power demand"], mode="lines", name="Historical Demand (MW)", line=dict(color="#2563eb", width=2)))
        fig_fc.add_trace(go.Scatter(x=df_fc["datetime"], y=df_fc["predicted_demand_mw"], mode="lines", name=f"Forecast Demand ({fc_res['model'].split(' ')[0]}) (MW)", line=dict(color="#ea580c", width=2.5, dash="dash")))
        _wl(fig_fc, f"Delhi Power Demand: Historical Context & {fc_res['horizon']}")
        st.plotly_chart(fig_fc, width="stretch")

        st.download_button(
            "📥 Download Forecast CSV",
            df_fc.to_csv(index=False).encode(),
            "delhi_power_demand_forecast.csv",
            "text/csv"
        )

# ==============================================================================
# SECTION 6: PEAK-DEMAND ANALYSIS
# ==============================================================================
elif page.startswith("6."):
    render_dash_bar("Dedicated Peak-Demand Analysis (All 3 Models)")

    st.markdown('<div class="st2">Test Period Peak Observations (272 Daily Peaks)</div>', unsafe_allow_html=True)
    peak_perf_path = ROOT / "outputs" / "peak_analysis" / "peak_model_performance.csv"
    if peak_perf_path.exists():
        st.markdown("**Model Peak Performance Comparison (MW and Timing):**")
        st.dataframe(pd.read_csv(peak_perf_path), width="stretch", hide_index=True)

    daily_peak_path = ROOT / "outputs" / "peak_analysis" / "daily_peak_analysis.csv"
    if daily_peak_path.exists():
        dp_df = pd.read_csv(daily_peak_path)
        st.markdown('<div class="st2">Top 10 Daily Peaks in Test Period</div>', unsafe_allow_html=True)
        top_10 = dp_df.nlargest(10, "actual_daily_peak_mw")
        st.dataframe(top_10, width="stretch", hide_index=True)

        st.markdown('<div class="st2">Actual Daily Peak vs. Predictions Over Time</div>', unsafe_allow_html=True)
        fig_pk = go.Figure()
        dp_dt = pd.to_datetime(dp_df["actual_peak_time"])
        fig_pk.add_trace(go.Scatter(x=dp_dt, y=dp_df["actual_daily_peak_mw"], mode="lines+markers", name="Actual Daily Peak (MW)", line=dict(color="#0d47a1", width=2)))
        fig_pk.add_trace(go.Scatter(x=dp_dt, y=dp_df["lr_pred_at_actual_peak_mw"], mode="lines", name="Linear Regression (MW)", line=dict(color="#dc2626", width=1.3, dash="dash")))
        fig_pk.add_trace(go.Scatter(x=dp_dt, y=dp_df["rf_pred_at_actual_peak_mw"], mode="lines", name="Random Forest (MW)", line=dict(color="#16a34a", width=1.3, dash="dot")))
        if "lstm_pred_at_actual_peak_mw" in dp_df.columns:
            fig_pk.add_trace(go.Scatter(x=dp_dt, y=dp_df["lstm_pred_at_actual_peak_mw"], mode="lines", name="PyTorch LSTM (MW)", line=dict(color="#9333ea", width=1.5, dash="dashdot")))
        
        _wl(fig_pk, "Daily Peak Demand Over Time (Test Period)")
        st.plotly_chart(fig_pk, width="stretch")

# ==============================================================================
# SECTION 7: WEATHER SCENARIO ANALYSIS
# ==============================================================================
elif page.startswith("7."):
    render_dash_bar("Interactive Weather Scenario & What-If Simulator")

    st.markdown('<div class="st2">Step 1 — Atmospheric Scenario Presets & Live Weather</div>', unsafe_allow_html=True)
    c_p1, c_p2 = st.columns([1.5, 1.0])
    with c_p1:
        preset_name = st.selectbox("Select Scenario Preset:", list(SCENARIO_PRESETS.keys()))
        preset_vals = SCENARIO_PRESETS[preset_name]
        st.caption(preset_vals["desc"])
    with c_p2:
        st.markdown("**Optional Meteorological API Query**")
        if st.button("🌐 Fetch Current New Delhi Weather (Open-Meteo)"):
            with st.spinner("Querying free Open-Meteo API for New Delhi (28.61° N, 77.21° E)..."):
                live_w = fetch_optional_delhi_weather(timeout_secs=3.0)
                if live_w:
                    st.session_state["live_weather"] = live_w
                    st.success(f"Fetched live weather ({live_w['temp']}°C, {live_w['rhum']}% RH, {live_w['pres']} hPa).")
                else:
                    st.warning("Weather API connection timed out or network is offline. Presets and manual inputs remain fully usable.")

    # Populate default inputs
    w_src = st.session_state.get("live_weather", preset_vals)

    st.markdown('<div class="st2">Step 2 — Dispatch Timestamp & Meteorological Inputs</div>', unsafe_allow_html=True)

    def_year = int(preset_vals.get("year", 2024))
    def_month = int(preset_vals.get("month", 6))
    def_day = int(preset_vals.get("day", 19))
    def_hour = int(preset_vals.get("hour", 15))
    def_minute = int(preset_vals.get("minute", 5))

    c_t1, c_t2 = st.columns(2)
    with c_t1:
        sim_date = st.date_input(
            "Simulation Calendar Date",
            value=date(def_year, def_month, def_day),
            min_value=date(2021, 1, 1),
            max_value=date(2030, 12, 31)
        )
    with c_t2:
        sim_time = st.time_input(
            "Simulation Dispatch Time (5-Min Grid Steps)",
            value=time(def_hour, def_minute),
            step=300
        )

    sim_dt = datetime.combine(sim_date, sim_time)

    cw1, cw2, cw3 = st.columns(3)
    with cw1:
        temp_val = st.number_input("Ambient Temperature (°C)", min_value=-5.0, max_value=55.0, value=float(w_src.get("temp", 35.0)), step=0.5)
        dwpt_val = st.number_input("Dew Point (°C)", min_value=-20.0, max_value=35.0, value=float(w_src.get("dwpt", 20.0)), step=0.5)
    with cw2:
        rhum_val = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=float(w_src.get("rhum", 45.0)), step=1.0)
        pres_val = st.number_input("Surface Pressure (hPa)", min_value=950.0, max_value=1050.0, value=float(w_src.get("pres", 998.0)), step=0.5)
    with cw3:
        wspd_val = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=float(w_src.get("wspd", 12.0)), step=0.5)
        wdir_val = st.number_input("Wind Direction (°)", min_value=0.0, max_value=360.0, value=float(w_src.get("wdir", 270.0)), step=5.0)

    weather_dict = {"temp": temp_val, "dwpt": dwpt_val, "rhum": rhum_val, "pres": pres_val, "wspd": wspd_val, "wdir": wdir_val}
    is_valid, val_warnings = validate_weather_inputs(weather_dict)

    if not is_valid:
        for w in val_warnings:
            st.error(w)
    elif val_warnings:
        for w in val_warnings:
            st.warning(w)

    heat_idx = calculate_heat_index(temp_val, rhum_val)
    st.info(f"🌡️ **Calculated Apparent Heat Index:** **{heat_idx:.1f} °C** (Wind: {get_wind_direction_label(wdir_val)})")

    st.markdown('<div class="st2">Step 3 — Run Model Projections Under This Weather Scenario</div>', unsafe_allow_html=True)
    m_choice = st.selectbox("Select Model for Primary Projection:", ["Linear Regression (Baseline)", "Random Forest (100 Trees)"] + (["PyTorch LSTM (Deep Network)"] if is_model_available("lstm") else []))
    m_k = "linear_regression" if "Linear" in m_choice else ("random_forest" if "Random" in m_choice else "lstm")

    if is_valid and st.button("⚡ Project Demand Under Scenario", type="primary"):
        sim_is_weekend = 1.0 if sim_dt.weekday() in (5, 6) else 0.0
        feat_dict = {
            "year": float(sim_dt.year),
            "month": float(sim_dt.month),
            "day": float(sim_dt.day),
            "hour": float(sim_dt.hour),
            "minute": float(sim_dt.minute),
            "day_of_week": float(sim_dt.weekday()),
            "is_weekend": sim_is_weekend,
            "demand_lag_5min": float(preset_vals.get("demand_lag_5min", 8500.0)),
            "demand_lag_1hour": float(preset_vals.get("demand_lag_1hour", 8200.0)),
            "demand_lag_24hour": float(preset_vals.get("demand_lag_24hour", 8100.0)),
            "demand_lag_7day": float(preset_vals.get("demand_lag_7day", 7800.0)),
            **weather_dict
        }
        f_row = make_custom_feature_row(feat_dict)
        model = load_model(m_k)
        proj_mw = float(model.predict(f_row)[0])
        rg = classify_demand_regime(proj_mw)

        st.markdown(f"""
        <div class="ne-result-banner">
          <div style="font-size:0.80rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#38bdf8;">⚡ Model-Based Scenario Demand Estimate (Simulation Only)</div>
          <div class="ne-rb-val">{proj_mw:,.2f} <span>MW</span></div>
          <div style="font-size:0.90rem;margin-top:6px;">
            Simulated Dispatch: <b>{sim_dt.strftime('%A, %d %B %Y at %H:%M IST')}</b> &bull; Operating Tier: <b>{rg['badge']}</b> &bull; Historical Stress: <b>{rg['stress']}</b>
          </div>
          <div style="font-size:0.82rem;color:#94a3b8;margin-top:4px;">
            ℹ️ <i>Notice: This value is a hypothetical model-based scenario projection, not actual real-time grid dispatch telemetry.</i>
          </div>
          <div style="font-size:0.85rem;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.2);">🛡️ <b>Operational Advisory:</b> {rg['advice']}</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 8: FORECAST ALERTS & REGIMES
# ==============================================================================
elif page.startswith("8."):
    render_dash_bar("Demand Regimes & Operational Alert Thresholds")

    st.markdown('<div class="st2">Delhi Power System Operational Demand Tiers</div>', unsafe_allow_html=True)
    st.markdown(r"""
    The Delhi grid load distribution (2021–2024) is partitioned into 4 quantile-based operational tiers:
    1. **Normal Demand ($\le 5,589$ MW, $\le 75$th percentile):** Baseload operations; optimal stability.
    2. **High Demand ($5,589$ to $6,906$ MW, 75th–95th percentile):** Substantial HVAC loads active.
    3. **Extreme Peak Demand ($6,906$ to $7,655$ MW, 95th–99th percentile):** Peaking gas units and spinning reserves required.
    4. **Super Extreme Peak ($> 7,655$ MW, Top 1%):** Severe grid stress approaching historical records.
    """)

    st.markdown('<div class="st2">User-Configurable Operational Alert Threshold</div>', unsafe_allow_html=True)
    thresh = st.slider("Set Simulation Alert Threshold (MW):", min_value=5000.0, max_value=8600.0, value=7500.0, step=50.0)

    st.markdown("""
    <div class="alert-warning-box">
      ℹ️ <b>Threshold Disclaimer:</b> This alert threshold is user-configured for simulation and study purposes.
      It does not constitute an official SLDC emergency limit or statutory grid protocol.
    </div>
    """, unsafe_allow_html=True)

    # Count days in test period exceeding threshold
    dp_file = ROOT / "outputs" / "peak_analysis" / "daily_peak_analysis.csv"
    if dp_file.exists():
        dp_data = pd.read_csv(dp_file)
        exceed_count = (dp_data["actual_daily_peak_mw"] >= thresh).sum()
        pct_exceed = (exceed_count / len(dp_data)) * 100
        st.metric(f"Test Days Exceeding {thresh:,.0f} MW", f"{exceed_count} of {len(dp_data)} Days", f"{pct_exceed:.1f}%")

# ==============================================================================
# SECTION 9: MODEL EVALUATION & RESIDUALS
# ==============================================================================
elif page.startswith("9."):
    render_dash_bar("Model Evaluation & Hold-Out Test Residuals")

    st.markdown('<div class="st2">Hold-Out Test Error Residual Inspection</div>', unsafe_allow_html=True)
    m_choice = st.selectbox("Select Model:", ["linear_regression", "random_forest"] + (["lstm"] if (ROOT / "lstm_test_predictions.csv").exists() else []),
                            format_func=lambda x: x.replace("_", " ").title())
    
    df_p = load_predictions(m_choice)
    df_p["error"] = df_p["actual_demand"] - df_p["predicted_demand"]
    df_p["abs_error"] = df_p["error"].abs()

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Test MAE", f"{df_p['abs_error'].mean():.2f} MW")
    c_m2.metric("Test RMSE", f"{np.sqrt((df_p['error']**2).mean()):.2f} MW")
    c_m3.metric("Max Absolute Error", f"{df_p['abs_error'].max():.2f} MW")

    # Zoomed comparison section with period selection
    st.markdown('<div class="st2">Hold-Out Actual vs. Predicted Demand Inspection</div>', unsafe_allow_html=True)
    p_min_dt = df_p["datetime"].min().date()
    p_max_dt = df_p["datetime"].max().date()

    period_mode = st.radio(
        "Select Evaluation Inspection Period:",
        ["🔥 Summer Peak Period (18–20 June 2024)", "🌱 Typical Spring Period (22–27 March 2024)", "📅 Custom Hold-Out Date Range"],
        horizontal=True
    )

    if period_mode.startswith("🔥"):
        start_dt = pd.Timestamp("2024-06-18 00:00:00")
        end_dt = pd.Timestamp("2024-06-20 23:55:00")
        sample_view = df_p[(df_p["datetime"] >= start_dt) & (df_p["datetime"] <= end_dt)]
        period_label = "Summer Peak Period (18–20 June 2024)"
    elif period_mode.startswith("🌱"):
        sample_view = df_p.iloc[5000:6500]
        period_label = "Typical Spring Period (22–27 March 2024)"
    else:
        ev_dates = st.date_input(
            "Select Date Range within Hold-Out Period (March to December 2024):",
            value=(date(2024, 6, 15), date(2024, 6, 22)),
            min_value=p_min_dt,
            max_value=p_max_dt
        )
        if len(ev_dates) == 2:
            start_dt = pd.Timestamp(ev_dates[0])
            end_dt = pd.Timestamp(ev_dates[1]) + pd.Timedelta(days=1)
            sample_view = df_p[(df_p["datetime"] >= start_dt) & (df_p["datetime"] < end_dt)]
            period_label = f"Custom Range: {ev_dates[0]} to {ev_dates[1]}"
        elif len(ev_dates) == 1:
            st.info("ℹ️ Please select an end date to view the custom hold-out comparison.")
            sample_view = pd.DataFrame()
            period_label = ""
        else:
            sample_view = pd.DataFrame()
            period_label = ""

    if not sample_view.empty:
        step_v = max(1, len(sample_view) // 1000)
        plot_df = sample_view.iloc[::step_v]

        fig_zoom = go.Figure()
        fig_zoom.add_trace(go.Scatter(x=plot_df["datetime"], y=plot_df["actual_demand"], mode="lines", name="Actual Demand (MW)", line=dict(color="#0d47a1", width=2)))
        fig_zoom.add_trace(go.Scatter(x=plot_df["datetime"], y=plot_df["predicted_demand"], mode="lines", name=f"Predicted ({m_choice.replace('_', ' ').title()}) (MW)", line=dict(color="#ea580c", width=1.8, dash="dash")))
        _wl(fig_zoom, f"Actual vs Predicted Load — {period_label} ({m_choice.replace('_', ' ').title()})")
        st.plotly_chart(fig_zoom, width="stretch")
    elif len(period_mode) > 0 and not period_mode.startswith("📅"):
        st.warning("⚠️ No hold-out predictions found for the selected period.")

# ==============================================================================
# SECTION 10: DOWNLOADABLE REPORTS
# ==============================================================================
elif page.startswith("10."):
    render_dash_bar("Downloadable Evaluation Reports & Data Artifacts")

    st.markdown('<div class="st2">Download Verified Project CSV & HTML Artifacts</div>', unsafe_allow_html=True)
    c_d1, c_d2 = st.columns(2)

    with c_d1:
        st.markdown("**1. Model Comparison Summary CSV**")
        comp_path = ROOT / "model_comparison_delhi_powerdemand.csv"
        if comp_path.exists():
            with open(comp_path, "rb") as f:
                st.download_button("📥 Download Model Comparison CSV", f.read(), "model_comparison_delhi_powerdemand.csv", "text/csv")
        else:
            st.warning("⚠️ model_comparison_delhi_powerdemand.csv not found on disk.")

        st.markdown("**2. Daily Peak Analysis CSV (272 Days)**")
        daily_path = ROOT / "outputs" / "peak_analysis" / "daily_peak_analysis.csv"
        if daily_path.exists():
            with open(daily_path, "rb") as f:
                st.download_button("📥 Download Daily Peak Analysis CSV", f.read(), "daily_peak_analysis.csv", "text/csv")
        else:
            st.warning("⚠️ daily_peak_analysis.csv not found on disk.")

    with c_d2:
        st.markdown("**3. Peak Model Performance CSV**")
        perf_path = ROOT / "outputs" / "peak_analysis" / "peak_model_performance.csv"
        if perf_path.exists():
            with open(perf_path, "rb") as f:
                st.download_button("📥 Download Peak Performance CSV", f.read(), "peak_model_performance.csv", "text/csv")
        else:
            st.warning("⚠️ peak_model_performance.csv not found on disk.")

        st.markdown("**4. Project Summary Markdown Report**")
        summary_path = ROOT / "outputs" / "project_dashboard" / "project_summary.md"
        if summary_path.exists():
            with open(summary_path, "rb") as f:
                st.download_button("📥 Download Project Summary (Markdown)", f.read(), "project_summary.md", "text/markdown")
        else:
            st.warning("⚠️ project_summary.md not found on disk.")

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("**5. Standalone Interactive HTML Dashboard**")
    html_dash_path = ROOT / "outputs" / "project_dashboard" / "project_dashboard.html"
    if html_dash_path.exists():
        with open(html_dash_path, "rb") as f:
            st.download_button(
                "🌐 Download Standalone HTML Dashboard",
                f.read(),
                "project_dashboard.html",
                "text/html"
            )
        st.caption("Self-contained visual report with embedded charts. Can be opened offline in any standard web browser without a Python server.")
    else:
        st.warning("⚠️ Standalone HTML dashboard (project_dashboard.html) not found. Run python project_results_dashboard.py to generate it.")

