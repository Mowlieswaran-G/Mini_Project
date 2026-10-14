"""
Chase | Delhi Power Grid AI Management Dashboard
State Load Despatch Centre (SLDC) Electricity Demand Forecasting & Grid Operations Platform
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, time
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths & Imports ───────────────────────────────────────────────────────────
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
    page_title="Chase | Delhi Power Grid Management",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Chase Design System & CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --chase-bg: #F4F7FC;
    --chase-card: #FFFFFF;
    --chase-border: #E8ECF2;
    --chase-primary: #4F46E5;
    --chase-primary-light: #EEF2FF;
    --chase-text-main: #1E293B;
    --chase-text-muted: #64748B;
    --chase-text-sub: #94A3B8;
    
    /* Signature Pastel Card Tokens from reference UI */
    --kpi-mint-bg: #E8F8F2;
    --kpi-mint-text: #059669;
    --kpi-mint-pill: #A7F3D0;
    
    --kpi-coral-bg: #FFEAEA;
    --kpi-coral-text: #E11D48;
    --kpi-coral-pill: #FECDD3;
    
    --kpi-peach-bg: #FFF3E6;
    --kpi-peach-text: #D97706;
    --kpi-peach-pill: #FED7AA;
    
    --kpi-purple-bg: #F3E8FF;
    --kpi-purple-text: #7C3AED;
    --kpi-purple-pill: #DDD6FE;
    
    --kpi-cyan-bg: #E0F2FE;
    --kpi-cyan-text: #0284C7;
    --kpi-cyan-pill: #BAE6FD;
    
    --kpi-yellow-bg: #FEF9C3;
    --kpi-yellow-text: #CA8A04;
    --kpi-yellow-pill: #FEF08A;
}

/* Base Body and Streamlit Container Overrides */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--chase-bg) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--chase-text-main) !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 40px !important;
    z-index: 10 !important;
}

/* Streamlit Sidebar Styling - Floating Pristine Card */
section[data-testid="stSidebar"] {
    background-color: var(--chase-bg) !important;
    border-right: none !important;
    padding: 12px 6px 12px 14px !important;
}

section[data-testid="stSidebar"] > div:first-child {
    background: #FFFFFF !important;
    border-radius: 20px !important;
    border: 1px solid var(--chase-border) !important;
    padding: 24px 16px !important;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04) !important;
    height: calc(100vh - 24px) !important;
}

/* Main Content Area */
.main .block-container {
    max-width: 100% !important;
    padding: 12px 24px 32px 12px !important;
}

/* Top App Bar */
.chase-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #FFFFFF;
    border: 1px solid var(--chase-border);
    border-radius: 18px;
    padding: 14px 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(15, 23, 42, 0.03);
}

.chase-top-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--chase-text-main);
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 12px;
}

.chase-top-right {
    display: flex;
    align-items: center;
    gap: 16px;
}

.chase-date-badge {
    background: #F8FAFC;
    border: 1px solid var(--chase-border);
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 0.80rem;
    font-weight: 600;
    color: var(--chase-text-muted);
    display: flex;
    align-items: center;
    gap: 8px;
}

.chase-user-card {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #F8FAFC;
    border: 1px solid var(--chase-border);
    border-radius: 9999px;
    padding: 4px 14px 4px 6px;
    cursor: default;
}

.chase-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
}

/* Card Component */
.chase-card {
    background: #FFFFFF;
    border: 1px solid var(--chase-border);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.03);
    margin-bottom: 18px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.chase-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}

.chase-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--chase-text-main);
    letter-spacing: -0.01em;
}

.chase-card-subtitle {
    font-size: 0.78rem;
    color: var(--chase-text-muted);
    font-weight: 500;
}

/* Pastel KPI Metric Cards (From Chase Design) */
.kpi-pastel-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}

.kpi-pastel-card {
    border-radius: 16px;
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 100px;
    position: relative;
    border: 1px solid transparent;
}

.kpi-pastel-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.kpi-pastel-icon {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
}

.kpi-pastel-val {
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 4px;
}

.kpi-pastel-label {
    font-size: 0.76rem;
    font-weight: 600;
    opacity: 0.9;
}

/* Colors for KPI cards */
.kpi-mint { background-color: var(--kpi-mint-bg); color: var(--kpi-mint-text); border-color: #D1FAE5; }
.kpi-mint .kpi-pastel-icon { background-color: #FFFFFF; color: var(--kpi-mint-text); box-shadow: 0 2px 6px rgba(5,150,105,0.12); }

.kpi-coral { background-color: var(--kpi-coral-bg); color: var(--kpi-coral-text); border-color: #FFE4E6; }
.kpi-coral .kpi-pastel-icon { background-color: #FFFFFF; color: var(--kpi-coral-text); box-shadow: 0 2px 6px rgba(225,29,72,0.12); }

.kpi-peach { background-color: var(--kpi-peach-bg); color: var(--kpi-peach-text); border-color: #FEF3C7; }
.kpi-peach .kpi-pastel-icon { background-color: #FFFFFF; color: var(--kpi-peach-text); box-shadow: 0 2px 6px rgba(217,119,6,0.12); }

.kpi-purple { background-color: var(--kpi-purple-bg); color: var(--kpi-purple-text); border-color: #EDE9FE; }
.kpi-purple .kpi-pastel-icon { background-color: #FFFFFF; color: var(--kpi-purple-text); box-shadow: 0 2px 6px rgba(124,58,237,0.12); }

.kpi-cyan { background-color: var(--kpi-cyan-bg); color: var(--kpi-cyan-text); border-color: #E0F2FE; }
.kpi-cyan .kpi-pastel-icon { background-color: #FFFFFF; color: var(--kpi-cyan-text); box-shadow: 0 2px 6px rgba(2,132,199,0.12); }

.kpi-yellow { background-color: var(--kpi-yellow-bg); color: var(--kpi-yellow-text); border-color: #FEF08A; }
.kpi-yellow .kpi-pastel-icon { background-color: #FFFFFF; color: var(--kpi-yellow-text); box-shadow: 0 2px 6px rgba(202,138,4,0.12); }

/* Load Shedding Banner Card */
.load-shedding-box {
    background: #FEF9C3;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    margin-bottom: 14px;
    border: 1px solid #FEF08A;
}
.load-shedding-number {
    font-size: 2.1rem;
    font-weight: 800;
    color: #854D0E;
    line-height: 1;
}
.load-shedding-caption {
    font-size: 0.78rem;
    font-weight: 700;
    color: #A16207;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* Clean List Table for Load Shedding */
.chase-rank-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.chase-rank-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.84rem;
}
.chase-rank-name {
    font-weight: 600;
    color: var(--chase-text-main);
}
.chase-rank-value {
    font-weight: 700;
    color: var(--chase-text-muted);
}

/* Revenue Pill Grid */
.rev-pill-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.rev-pill {
    padding: 12px 14px;
    border-radius: 12px;
    text-align: center;
}
.rev-pill-label {
    font-size: 0.70rem;
    font-weight: 600;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.rev-pill-val {
    font-size: 1.15rem;
    font-weight: 800;
}

/* Modern Consumer Data Table */
.chase-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.85rem;
}
.chase-table th {
    background: #F8FAFC;
    color: var(--chase-text-muted);
    font-weight: 700;
    text-align: left;
    padding: 12px 14px;
    border-bottom: 1.5px solid var(--chase-border);
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.chase-table td {
    padding: 12px 14px;
    border-bottom: 1px solid #F1F5F9;
    color: var(--chase-text-main);
    font-weight: 500;
}
.chase-table tr:hover td {
    background-color: #F8FAFC;
}
.badge-status-active {
    background: #E8F8F2;
    color: #059669;
    padding: 3px 8px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    display: inline-block;
}
.badge-status-warning {
    background: #FFEAEA;
    color: #E11D48;
    padding: 3px 8px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    display: inline-block;
}
.table-action-btn {
    color: #4F46E5 !important;
    font-weight: 700;
    font-size: 0.80rem;
    text-decoration: none;
    cursor: pointer;
}

/* Sidebar Radio Navigation Pill Overrides */
div[data-testid="stRadio"] > div {
    gap: 6px !important;
}
div[data-testid="stRadio"] label {
    background: transparent !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    margin: 0 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: var(--chase-text-muted) !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stRadio"] label:hover {
    background: #F1F5F9 !important;
    color: var(--chase-text-main) !important;
}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {
    background: #EEF2FF !important;
    color: #4F46E5 !important;
    font-weight: 700 !important;
}

/* Status Model Cards in Sidebar */
.sidebar-model-badge {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px;
    background: #F8FAFC;
    border: 1px solid var(--chase-border);
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}
.sidebar-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.dot-green { background: #10B981; }
.dot-yellow { background: #F59E0B; }
.dot-red { background: #EF4444; }

/* Custom Forecast Result Banner */
.forecast-result-card {
    background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%);
    border-radius: 18px;
    padding: 24px;
    color: #FFFFFF;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.3);
}
</style>
""", unsafe_allow_html=True)

# ── Clean Plotly Layout Function (Spline, Minimalist, White Canvas) ────────────
def build_chase_chart(fig, title: str = "", height: int = 340, y_title: str = "Demand (MW)"):
    """Styles Plotly figures with smooth spline curves and Chase light-mode aesthetics."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            x=0, y=0.98, xanchor="left", yanchor="top",
            font=dict(size=14, color="#1E293B", family="Plus Jakarta Sans")
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#64748B"),
        xaxis=dict(
            gridcolor="#F1F5F9",
            linecolor="#E2E8F0",
            zeroline=False,
            tickfont=dict(size=11, color="#64748B")
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            linecolor="#E2E8F0",
            zeroline=False,
            title=y_title,
            title_font=dict(size=11, color="#64748B"),
            tickfont=dict(size=11, color="#64748B")
        ),
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#475569")
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Plus Jakarta Sans",
            font_color="#FFFFFF"
        )
    )
    return fig

# ── Load Dataset History ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_cached_history():
    return load_history()

try:
    history = get_cached_history()
except Exception as exc:
    st.error(f"Could not load historical Delhi telemetry dataset: {exc}")
    st.stop()

# ── Top Bar Renderer ──────────────────────────────────────────────────────────
def render_topbar(page_name: str, date_range_text: str = "Jan 5, 2024 – Dec 31, 2024"):
    st.markdown(f"""
    <div class="chase-topbar">
      <div class="chase-top-title">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="9.5" stroke="#4F46E5" stroke-width="2.2"/>
          <circle cx="12" cy="12" r="4.2" fill="#4F46E5"/>
        </svg>
        <span>{page_name}</span>
      </div>
      <div class="chase-top-right">
        <div class="chase-date-badge">
          <span>📅</span> {date_range_text}
        </div>
        <div class="chase-user-card">
          <div class="chase-avatar">M</div>
          <div style="text-align:left;">
            <div style="font-size:0.80rem;font-weight:700;color:#1E293B;line-height:1.1;">Mowlieswaran</div>
            <div style="font-size:0.68rem;color:#64748B;font-weight:600;">Grid Manager</div>
          </div>
          <span style="font-size:0.75rem;color:#94A3B8;margin-left:4px;">▾</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:4px 6px 18px;border-bottom:1px solid #F1F5F9;margin-bottom:16px;">
      <svg width="34" height="34" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="#4F46E5" stroke-width="2.4"/>
        <circle cx="12" cy="12" r="4.5" fill="#4F46E5"/>
      </svg>
      <div>
        <div style="font-size:1.20rem;font-weight:800;color:#1E293B;line-height:1.1;letter-spacing:-0.02em;">Chase</div>
        <div style="font-size:0.70rem;color:#4F46E5;font-weight:700;letter-spacing:0.5px;">DELHI POWER SYSTEM</div>
      </div>
    </div>
    <div style="font-size:0.68rem;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;padding:0 8px 8px;">MENU</div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "📈 Performance",
            "🗺️ Distribution",
            "👥 Consumer & Telemetry",
            "🔮 Forecast Studio",
            "🌦️ Weather Scenarios",
            "🏔️ Peak Analysis",
            "📑 Reports & Audit"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<hr style='margin:18px 0 14px 0;border-color:#F1F5F9;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;padding:0 8px 8px;'>AI MODEL ENGINES</div>", unsafe_allow_html=True)

    lr_avail = is_model_available("linear_regression")
    rf_avail = is_model_available("random_forest")
    lstm_ready = is_model_available("lstm")
    lstm_exists = is_lstm_file_present()

    st.markdown(f"""
    <div class="sidebar-model-badge">
      <span><span class="sidebar-dot {'dot-green' if lr_avail else 'dot-red'}"></span>Linear Regression</span>
      <span style="color:{'#059669' if lr_avail else '#EF4444'};">{'Active' if lr_avail else 'Offline'}</span>
    </div>
    <div class="sidebar-model-badge">
      <span><span class="sidebar-dot {'dot-green' if rf_avail else 'dot-red'}"></span>Random Forest</span>
      <span style="color:{'#059669' if rf_avail else '#EF4444'};">{'Active (100T)' if rf_avail else 'Offline'}</span>
    </div>
    <div class="sidebar-model-badge">
      <span><span class="sidebar-dot {'dot-green' if lstm_ready else ('dot-yellow' if lstm_exists else 'dot-red')}"></span>PyTorch LSTM</span>
      <span style="color:{'#059669' if lstm_ready else ('#D97706' if lstm_exists else '#EF4444')};">{'Active' if lstm_ready else ('Ready' if lstm_exists else 'Offline')}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.70rem;color:#94A3B8;padding:12px 6px 0;text-align:center;">
      Delhi SLDC • 393,440 Telemetry Steps
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 1: DASHBOARD (Top-Left Screen in Reference UI)
# ==============================================================================
if nav_selection.startswith("📊"):
    render_topbar("Dashboard", "Jan 1, 2024 – Dec 31, 2024")

    # ── Row 1: Sudden Interruption | Collection & Due | Revenue ──────────────────
    c1, c2, c3 = st.columns([1.3, 1.2, 1.1])

    with c1:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Sudden Interruption</div>
              <div class="chase-card-subtitle">Grid load dips & rapid ramp-rate variations</div>
            </div>
            <div style="font-size:0.75rem;font-weight:700;color:#0284C7;background:#E0F2FE;padding:3px 8px;border-radius:6px;">Real-Time</div>
          </div>
        """, unsafe_allow_html=True)
        
        # 14-day sample for sudden variations
        sample_slice = history.tail(288 * 14).iloc[::12].copy()
        fig_interruption = go.Figure()
        fig_interruption.add_trace(go.Scatter(
            x=sample_slice["datetime"],
            y=sample_slice["Power demand"],
            mode="lines",
            line=dict(color="#06B6D4", width=2, shape="spline", smoothing=1.2),
            name="Load Telemetry"
        ))
        build_chase_chart(fig_interruption, height=160, y_title="MW")
        fig_interruption.update_layout(margin=dict(l=5, r=5, t=10, b=10))
        st.plotly_chart(fig_interruption, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Collection & Due</div>
              <div class="chase-card-subtitle">Delhi DISCOM Power Tariff & Capacity Reserve</div>
            </div>
          </div>
          <div style="display:flex;flex-direction:column;gap:12px;margin-top:6px;">
            <div style="background:#E8F8F2;border:1px solid #D1FAE5;border-radius:14px;padding:14px 18px;display:flex;justify-content:space-between;align-items:center;">
              <div>
                <div style="font-size:0.70rem;font-weight:700;color:#059669;text-transform:uppercase;letter-spacing:0.5px;">Summer Peak Demand</div>
                <div style="font-size:1.65rem;font-weight:800;color:#059669;margin-top:2px;">₹ 1,02,487 <span style="font-size:0.80rem;font-weight:600;color:#047857;">(8,631 MW)</span></div>
              </div>
              <span style="background:#A7F3D0;color:#065F46;font-size:0.72rem;font-weight:700;padding:4px 8px;border-radius:6px;">Peak Safe</span>
            </div>
            <div style="background:#FFEAEA;border:1px solid #FFE4E6;border-radius:14px;padding:14px 18px;display:flex;justify-content:space-between;align-items:center;">
              <div>
                <div style="font-size:0.70rem;font-weight:700;color:#E11D48;text-transform:uppercase;letter-spacing:0.5px;">Spinning Reserve Margin</div>
                <div style="font-size:1.65rem;font-weight:800;color:#E11D48;margin-top:2px;">₹ 14,523 <span style="font-size:0.80rem;font-weight:600;color:#BE123C;">(420 MW)</span></div>
              </div>
              <span style="background:#FECDD3;color:#9F1239;font-size:0.72rem;font-weight:700;padding:4px 8px;border-radius:6px;">Critical</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Revenue & Tariffs</div>
              <div class="chase-card-subtitle">Quarterly Delhi Grid Dispatch</div>
            </div>
          </div>
          <div class="rev-pill-grid">
            <div class="rev-pill" style="background:#E8F8F2;color:#059669;">
              <div class="rev-pill-label">Last Month</div>
              <div class="rev-pill-val">₹ 1,53,255</div>
            </div>
            <div class="rev-pill" style="background:#FFF3E6;color:#D97706;">
              <div class="rev-pill-label">October</div>
              <div class="rev-pill-val">₹ 3,13,254</div>
            </div>
            <div class="rev-pill" style="background:#E0F2FE;color:#0284C7;">
              <div class="rev-pill-label">September</div>
              <div class="rev-pill-val">₹ 1,43,625</div>
            </div>
            <div class="rev-pill" style="background:#F3E8FF;color:#7C3AED;">
              <div class="rev-pill-label">August</div>
              <div class="rev-pill-val">₹ 1,32,540</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Row 2: Generation vs Consumption | Load Shedding ────────────────────────
    c_m1, c_m2 = st.columns([2.1, 1.1])

    with c_m1:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Generation vs Consumption</div>
              <div class="chase-card-subtitle">Actual Grid Demand vs. Scheduled Generation Dispatch</div>
            </div>
            <div style="display:flex;gap:12px;font-size:0.75rem;font-weight:700;">
              <span style="color:#06B6D4;">● Generation</span>
              <span style="color:#8B5CF6;">● Consumption</span>
            </div>
          </div>
        """, unsafe_allow_html=True)

        # Build signature 2-spline curve with gradient fills and peak pin
        recent_chunk = history.tail(288 * 30).iloc[::24].copy()
        gen_sim = recent_chunk["Power demand"] * 1.035 + 25.0

        fig_gvc = go.Figure()
        fig_gvc.add_trace(go.Scatter(
            x=recent_chunk["datetime"],
            y=gen_sim,
            mode="lines",
            name="Generation (MW)",
            line=dict(color="#06B6D4", width=2.5, shape="spline", smoothing=1.3),
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.08)"
        ))
        fig_gvc.add_trace(go.Scatter(
            x=recent_chunk["datetime"],
            y=recent_chunk["Power demand"],
            mode="lines",
            name="Consumption (MW)",
            line=dict(color="#8B5CF6", width=2.5, shape="spline", smoothing=1.3),
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.08)"
        ))

        # Peak callout annotation
        max_idx = recent_chunk["Power demand"].idxmax()
        peak_row = recent_chunk.loc[max_idx]
        fig_gvc.add_annotation(
            x=peak_row["datetime"],
            y=peak_row["Power demand"],
            text=f"<b>{peak_row['Power demand']:,.0f} MW</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#4F46E5",
            ax=0,
            ay=-35,
            bgcolor="#4F46E5",
            font=dict(color="#ffffff", size=11, family="Plus Jakarta Sans"),
            borderpad=4
        )

        build_chase_chart(fig_gvc, height=270, y_title="Megawatts (MW)")
        st.plotly_chart(fig_gvc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_m2:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Load Shedding</div>
              <div class="chase-card-subtitle">Grid Stability & Feeder Outage Prevention</div>
            </div>
          </div>
          <div class="load-shedding-box">
            <div class="load-shedding-number">150 Hours</div>
            <div class="load-shedding-caption">Total Load Shedding Avoided</div>
          </div>
          <div class="chase-rank-list">
            <div class="chase-rank-item">
              <span class="chase-rank-name">BrightWave Energy (BRPL South)</span>
              <span class="chase-rank-value">50 hrs</span>
            </div>
            <div class="chase-rank-item">
              <span class="chase-rank-name">EcoPower Innovations (BYPL East)</span>
              <span class="chase-rank-value">20 hrs</span>
            </div>
            <div class="chase-rank-item">
              <span class="chase-rank-name">GreenGrid Utilities (TPDDL North)</span>
              <span class="chase-rank-value">15 hrs</span>
            </div>
            <div class="chase-rank-item">
              <span class="chase-rank-name">SolarSync Corporation (NDMC)</span>
              <span class="chase-rank-value">25 hrs</span>
            </div>
            <div class="chase-rank-item">
              <span class="chase-rank-name">FutureFlow Power Group (MES)</span>
              <span class="chase-rank-value">40 hrs</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Row 3: New Consumer Growth / Sector Demand ──────────────────────────────
    st.markdown("""
    <div class="chase-card">
      <div class="chase-card-header">
        <div>
          <div class="chase-card-title">New Consumer Growth</div>
          <div class="chase-card-subtitle">Delhi NCR Electrification & Sector Demand Expansion</div>
        </div>
        <div style="font-size:0.75rem;font-weight:600;color:#64748B;">This Month</div>
      </div>
    """, unsafe_allow_html=True)

    sector_df = pd.DataFrame({
        "Sector": ["Commercial High-Rise", "Industrial Parks", "EV Fast Charging", "Residential Domestic", "Institutional & Metro"],
        "Count": [450, 680, 890, 1120, 320]
    })
    fig_growth = px.bar(
        sector_df,
        x="Count",
        y="Sector",
        orientation="h",
        color_discrete_sequence=["#84CC16"]
    )
    fig_growth.update_traces(marker_line_width=0, opacity=0.85)
    build_chase_chart(fig_growth, height=200, y_title="")
    fig_growth.update_layout(xaxis_title="Active Consumer Nodes", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_growth, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: PERFORMANCE (Middle-Right Screen in Reference UI)
# ==============================================================================
elif nav_selection.startswith("📈"):
    render_topbar("Performance", "Jan 5, 2024 – Jan 5, 2025")

    # ── Top 6 Pastel KPI Cards matching reference ─────────────────────────────
    st.markdown("""
    <div class="kpi-pastel-grid">
      <div class="kpi-pastel-card kpi-cyan">
        <div class="kpi-pastel-top">
          <div class="kpi-pastel-icon">💡</div>
          <span style="font-size:0.70rem;font-weight:700;">+4.2%</span>
        </div>
        <div>
          <div class="kpi-pastel-val">1050 GWh</div>
          <div class="kpi-pastel-label">Total Generation</div>
        </div>
      </div>
      <div class="kpi-pastel-card kpi-mint">
        <div class="kpi-pastel-top">
          <div class="kpi-pastel-icon">⚡</div>
          <span style="font-size:0.70rem;font-weight:700;">99.8%</span>
        </div>
        <div>
          <div class="kpi-pastel-val">215:10:15</div>
          <div class="kpi-pastel-label">Up Time (Hrs)</div>
        </div>
      </div>
      <div class="kpi-pastel-card kpi-coral">
        <div class="kpi-pastel-top">
          <div class="kpi-pastel-icon">⏱️</div>
          <span style="font-size:0.70rem;font-weight:700;">-12.4%</span>
        </div>
        <div>
          <div class="kpi-pastel-val">08:10:15</div>
          <div class="kpi-pastel-label">Fall Time / Outage</div>
        </div>
      </div>
      <div class="kpi-pastel-card kpi-purple">
        <div class="kpi-pastel-top">
          <div class="kpi-pastel-icon">🏢</div>
          <span style="font-size:0.70rem;font-weight:700;">Active</span>
        </div>
        <div>
          <div class="kpi-pastel-val">110</div>
          <div class="kpi-pastel-label">New Substations</div>
        </div>
      </div>
      <div class="kpi-pastel-card kpi-peach">
        <div class="kpi-pastel-top">
          <div class="kpi-pastel-icon">🔧</div>
          <span style="font-size:0.70rem;font-weight:700;">Resolved</span>
        </div>
        <div>
          <div class="kpi-pastel-val">185</div>
          <div class="kpi-pastel-label">Grid Alerts Handled</div>
        </div>
      </div>
      <div class="kpi-pastel-card kpi-yellow">
        <div class="kpi-pastel-top">
          <div class="kpi-pastel-icon">⚙️</div>
          <span style="font-size:0.70rem;font-weight:700;">Online</span>
        </div>
        <div>
          <div class="kpi-pastel-val">515</div>
          <div class="kpi-pastel-label">Telemetry Nodes</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Middle Row: Generation Spline Curve | Up Time Bars ────────────────────
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Generation Profile</div>
              <div class="chase-card-subtitle">Monthly Grid Dispatch Volume (GWh)</div>
            </div>
          </div>
        """, unsafe_allow_html=True)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        gen_monthly = [520, 580, 690, 850, 1140, 1380, 1260, 1110, 940, 780, 610, 540]
        fig_gen = px.line(x=months, y=gen_monthly, markers=True)
        fig_gen.update_traces(line=dict(color="#06B6D4", width=3, shape="spline", smoothing=1.3), marker=dict(size=7, color="#06B6D4"))
        build_chase_chart(fig_gen, height=230, y_title="GWh")
        st.plotly_chart(fig_gen, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_p2:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Up Time</div>
              <div class="chase-card-subtitle">Substation Availability by Month (Hours)</div>
            </div>
          </div>
        """, unsafe_allow_html=True)
        uptime_vals = [720, 680, 740, 715, 742, 718, 740, 743, 718, 742, 719, 743]
        fig_uptime = px.bar(x=months, y=uptime_vals, color_discrete_sequence=["#10B981"])
        fig_uptime.update_traces(marker_line_width=0, opacity=0.88)
        build_chase_chart(fig_uptime, height=230, y_title="Hours")
        st.plotly_chart(fig_uptime, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Bottom Row: Fall Time | New Consumer | Complaint & Service Donut ───────
    col_b1, col_b2, col_b3 = st.columns([1.2, 1.2, 1.1])

    with col_b1:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Fall Time</div>
              <div class="chase-card-subtitle">Unplanned Outages Over Time</div>
            </div>
          </div>
        """, unsafe_allow_html=True)
        fall_vals = [12, 18, 14, 25, 38, 45, 32, 28, 20, 15, 10, 8]
        fig_fall = px.line(x=months, y=fall_vals)
        fig_fall.update_traces(
            line=dict(color="#F43F5E", width=2.5, shape="spline", smoothing=1.3),
            fill="tozeroy",
            fillcolor="rgba(244, 63, 94, 0.08)"
        )
        build_chase_chart(fig_fall, height=210, y_title="Hours")
        st.plotly_chart(fig_fall, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b2:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">New Consumer</div>
              <div class="chase-card-subtitle">Annual Consumer Growth Trajectory</div>
            </div>
          </div>
        """, unsafe_allow_html=True)
        consumer_growth = [110, 140, 180, 240, 310, 420, 510, 590, 660, 710, 780, 850]
        fig_cgrowth = px.line(x=months, y=consumer_growth)
        fig_cgrowth.update_traces(line=dict(color="#A855F7", width=2.5, shape="spline", smoothing=1.3))
        build_chase_chart(fig_cgrowth, height=210, y_title="New Connections")
        st.plotly_chart(fig_cgrowth, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b3:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Grid Demand Regimes</div>
              <div class="chase-card-subtitle">Operating Quantile Distribution</div>
            </div>
          </div>
        """, unsafe_allow_html=True)
        regimes_data = pd.DataFrame({
            "Regime": ["Normal (<5.5k MW)", "High (5.5-6.9k MW)", "Extreme Peak (>6.9k MW)"],
            "Share": [75.0, 20.0, 5.0]
        })
        fig_donut = px.pie(
            regimes_data,
            values="Share",
            names="Regime",
            hole=0.68,
            color_discrete_sequence=["#3B82F6", "#F87171", "#10B981"]
        )
        fig_donut.update_traces(textinfo="percent", textfont_size=11)
        fig_donut.update_layout(
            height=210,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=10))
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 3: DISTRIBUTION (Bottom-Right Screen in Reference UI)
# ==============================================================================
elif nav_selection.startswith("🗺️"):
    render_topbar("Distribution", "Delhi NCT 5 DISCOM Zones")

    # ── Top 4 Pastel Summary Pills + Large Stat Box ───────────────────────────
    col_d_left, col_d_right = st.columns([2.5, 1.0])

    with col_d_left:
        st.markdown("""
        <div class="kpi-pastel-grid" style="grid-template-columns: repeat(4, 1fr);">
          <div class="kpi-pastel-card kpi-mint">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">Base Tariff</span>
            </div>
            <div>
              <div class="kpi-pastel-val" style="font-size:1.35rem;">₹ 50,00,00,000</div>
              <div class="kpi-pastel-label">Bill Collection</div>
            </div>
          </div>
          <div class="kpi-pastel-card kpi-coral">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">Pending</span>
            </div>
            <div>
              <div class="kpi-pastel-val" style="font-size:1.35rem;">₹ 20,000</div>
              <div class="kpi-pastel-label">Due Bill Margin</div>
            </div>
          </div>
          <div class="kpi-pastel-card kpi-peach">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">Energy</span>
            </div>
            <div>
              <div class="kpi-pastel-val" style="font-size:1.35rem;">20,000 kWh</div>
              <div class="kpi-pastel-label">Consumed Electricity</div>
            </div>
          </div>
          <div class="kpi-pastel-card kpi-purple">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">Substations</span>
            </div>
            <div>
              <div class="kpi-pastel-val" style="font-size:1.35rem;">145 Nodes</div>
              <div class="kpi-pastel-label">Active Feeders</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_d_right:
        st.markdown("""
        <div class="chase-card" style="text-align:center;padding:16px 20px;">
          <div style="font-size:0.78rem;font-weight:700;color:#64748B;text-transform:uppercase;">Last Month's Revenue</div>
          <div style="font-size:2.2rem;font-weight:800;color:#4F46E5;margin:8px 0;">$ 825k</div>
          <div style="font-size:0.75rem;font-weight:600;color:#10B981;">▲ +8.4% vs Projected</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Middle Charts: Consumption Details Spline + Yearly Revenue ─────────────
    c_dist1, c_dist2 = st.columns([2.2, 1.0])

    with c_dist1:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Consumption Details</div>
              <div class="chase-card-subtitle">Monthly Demand Waveform with Peak Callout</div>
            </div>
            <div style="font-size:0.75rem;font-weight:700;color:#4F46E5;background:#EEF2FF;padding:4px 10px;border-radius:6px;">Monthly Waveform</div>
          </div>
        """, unsafe_allow_html=True)

        # Smooth spline line matching reference exactly with pin
        m_pts = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        c_curve = [4200, 3900, 4800, 5600, 7100, 8631, 7900, 7200, 6400, 5300, 4700, 4400]
        fig_cwave = go.Figure()
        fig_cwave.add_trace(go.Scatter(
            x=m_pts,
            y=c_curve,
            mode="lines",
            line=dict(color="#38BDF8", width=3, shape="spline", smoothing=1.3),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.08)"
        ))
        # Add the signature circular callout pin on June peak
        fig_cwave.add_annotation(
            x="Jun",
            y=8631,
            text="<b>8,632 MW</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#0284C7",
            ax=0,
            ay=-40,
            bgcolor="#0284C7",
            font=dict(color="#ffffff", size=11),
            borderpad=4
        )
        build_chase_chart(fig_cwave, height=220, y_title="Demand (MW)")
        st.plotly_chart(fig_cwave, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_dist2:
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Yearly Demand</div>
              <div class="chase-card-subtitle">Annual Growth 2021–2024</div>
            </div>
          </div>
        """, unsafe_allow_html=True)
        yr_pts = ["2021", "2022", "2023", "2024"]
        yr_vals = [7323, 7695, 7438, 8632]
        fig_yr = px.line(x=yr_pts, y=yr_vals, markers=True)
        fig_yr.update_traces(line=dict(color="#F59E0B", width=2.5), marker=dict(size=8, color="#F59E0B"))
        build_chase_chart(fig_yr, height=220, y_title="Annual Peak (MW)")
        st.plotly_chart(fig_yr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Bottom Section: Zone Wise Summary & Table ─────────────────────────────
    st.markdown("""
    <div class="chase-card">
      <div class="chase-card-header">
        <div>
          <div class="chase-card-title">Zone Wise Summary</div>
          <div class="chase-card-subtitle">Delhi State Electricity Distribution Companies (DISCOMs)</div>
        </div>
      </div>
      <table class="chase-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Zone / DISCOM</th>
            <th>Substation Area</th>
            <th>Grid Demand (MW)</th>
            <th>Collection Area</th>
            <th>Reliability Index</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>01</td>
            <td><b>BRPL</b> (BSES Rajdhani)</td>
            <td>South & West Delhi</td>
            <td><b>2,840 MW</b></td>
            <td>16,40,250</td>
            <td><span class="badge-status-active">99.85% Optimal</span></td>
            <td><a class="table-action-btn">Details</a></td>
          </tr>
          <tr>
            <td>02</td>
            <td><b>TPDDL</b> (Tata Power)</td>
            <td>North & North-West Delhi</td>
            <td><b>2,150 MW</b></td>
            <td>14,10,000</td>
            <td><span class="badge-status-active">99.92% High</span></td>
            <td><a class="table-action-btn">Details</a></td>
          </tr>
          <tr>
            <td>03</td>
            <td><b>BYPL</b> (BSES Yamuna)</td>
            <td>Central & East Delhi</td>
            <td><b>1,720 MW</b></td>
            <td>12,82,100</td>
            <td><span class="badge-status-active">99.70% Normal</span></td>
            <td><a class="table-action-btn">Details</a></td>
          </tr>
          <tr>
            <td>04</td>
            <td><b>NDMC</b> (New Delhi Municipal)</td>
            <td>Lutyens & Parliament Zone</td>
            <td><b>480 MW</b></td>
            <td>5,10,000</td>
            <td><span class="badge-status-active">100.00% Protected</span></td>
            <td><a class="table-action-btn">Details</a></td>
          </tr>
          <tr>
            <td>05</td>
            <td><b>MES</b> (Military Engineer Services)</td>
            <td>Delhi Cantonment</td>
            <td><b>230 MW</b></td>
            <td>2,40,000</td>
            <td><span class="badge-status-active">99.95% High</span></td>
            <td><a class="table-action-btn">Details</a></td>
          </tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 4: CONSUMER & TELEMETRY (Bottom-Left Screen in Reference UI)
# ==============================================================================
elif nav_selection.startswith("👥"):
    render_topbar("Consumer & Telemetry", "70,825 Hold-Out Records")

    st.markdown("""
    <div class="chase-card">
      <div class="chase-card-header">
        <div>
          <div class="chase-card-title">Delhi Grid Telemetry Explorer</div>
          <div class="chase-card-subtitle">5-Minute Intervals & Autoregressive Features</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    # Filter Bar matching reference UI
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.5, 2.0])
    with f1:
        cat_filter = st.selectbox("Category", ["All Sectors", "Residential", "Commercial", "Industrial"], index=0)
    with f2:
        model_view = st.selectbox("Benchmark", ["All Models", "Linear Regression", "Random Forest", "PyTorch LSTM"])
    with f3:
        search_kw = st.text_input("Search records", placeholder="Search timestamp or MW...")
    with f4:
        st.write("")
        st.write("")
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            st.button("🔍 Filter", type="secondary", use_container_width=True)
        with btn_c2:
            st.button("📥 Export CSV", type="primary", use_container_width=True)

    # Render Clean Data Table matching reference layout
    st.markdown("""
    <table class="chase-table" style="margin-top:14px;">
      <thead>
        <tr>
          <th>Sl</th>
          <th>Telemetry Timestamp</th>
          <th>Grid Zone</th>
          <th>Ambient Temp (°C)</th>
          <th>Humidity (%)</th>
          <th>Wind Speed (km/h)</th>
          <th>Actual Load (MW)</th>
          <th>Status</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
    """, unsafe_allow_html=True)

    # Show 15 formatted rows
    table_sample = history.tail(15).copy().reset_index(drop=True)
    table_rows_html = ""
    for idx, r in table_sample.iterrows():
        dt_str = pd.to_datetime(r["datetime"]).strftime("%Y-%m-%d %H:%M")
        load_val = f"{r['Power demand']:,.1f} MW"
        temp_val = f"{r['temp']:.1f} °C"
        rhum_val = f"{r['rhum']:.0f}%"
        wspd_val = f"{r['wspd']:.1f}"
        zone = "BRPL-South" if idx % 3 == 0 else ("TPDDL-North" if idx % 3 == 1 else "BYPL-East")
        status_pill = '<span class="badge-status-active">Normal</span>' if r['Power demand'] < 6000 else '<span class="badge-status-warning">Peak Load</span>'
        table_rows_html += f"""
        <tr>
          <td>{idx+1:02d}</td>
          <td><b>{dt_str}</b></td>
          <td>{zone}</td>
          <td>{temp_val}</td>
          <td>{rhum_val}</td>
          <td>{wspd_val}</td>
          <td><b>{load_val}</b></td>
          <td>{status_pill}</td>
          <td><a class="table-action-btn">Inspect</a></td>
        </tr>
        """

    st.markdown(table_rows_html + "</tbody></table>", unsafe_allow_html=True)

    # Pagination footer
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:16px 8px 6px;border-top:1px solid #F1F5F9;font-size:0.80rem;color:#64748B;">
      <div>Showing <b>1–15</b> of <b>393,440</b> telemetry records</div>
      <div style="display:flex;gap:6px;align-items:center;">
        <span style="padding:4px 10px;border-radius:6px;border:1px solid #E2E8F0;cursor:pointer;">‹ Previous</span>
        <span style="padding:4px 10px;border-radius:6px;background:#4F46E5;color:#ffffff;font-weight:700;">1</span>
        <span style="padding:4px 10px;border-radius:6px;border:1px solid #E2E8F0;cursor:pointer;">2</span>
        <span style="padding:4px 10px;border-radius:6px;border:1px solid #E2E8F0;cursor:pointer;">3</span>
        <span>...</span>
        <span style="padding:4px 10px;border-radius:6px;border:1px solid #E2E8F0;cursor:pointer;">10</span>
        <span style="padding:4px 10px;border-radius:6px;border:1px solid #E2E8F0;cursor:pointer;">Next ›</span>
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 5: FORECAST STUDIO (Recursive Multi-Step AI Predictor)
# ==============================================================================
elif nav_selection.startswith("🔮"):
    render_topbar("Forecast Studio", "Autonomous Forward AI Engine")

    st.markdown("""
    <div class="chase-card">
      <div class="chase-card-header">
        <div>
          <div class="chase-card-title">Multi-Step Recursive Forward Forecasting Engine</div>
          <div class="chase-card-subtitle">Autoregressive lag propagation on validated AI models</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        engine_opts = ["Linear Regression (Ultra-Fast)", "Random Forest (100 Trees Ensemble)"]
        if is_model_available("lstm"):
            engine_opts.append("PyTorch Deep LSTM (Recurrent Neural Network)")
        selected_engine = st.selectbox("Forecasting Model Architecture:", engine_opts)
        m_key = "linear_regression" if "Linear" in selected_engine else ("random_forest" if "Random" in selected_engine else "lstm")

    with f_col2:
        horizons_dict = {
            "Next 1 Hour (12 steps of 5-min intervals)": 12,
            "Next 24 Hours (288 steps of 5-min intervals)": 288,
            "Next 7 Days (2,016 steps of 5-min intervals)": 2016
        }
        chosen_h = st.selectbox("Forecast Horizon:", list(horizons_dict.keys()), index=1)
        step_n = horizons_dict[chosen_h]

    if st.button("🚀 Execute Recursive AI Forecast", type="primary", use_container_width=True):
        t_start = perf_counter()
        with st.spinner(f"Computing recursive trajectory for {step_n} intervals..."):
            iv = pd.Timedelta(history["datetime"].diff().dropna().mode().iloc[0])
            st_dt = history["datetime"].max() + iv
            fut_dts = pd.date_range(st_dt, periods=step_n, freq=iv)

            rec_demands = list(history["Power demand"].tail(2016).to_numpy(dtype=float))
            last_w = history.iloc[-1][WEATHER_COLUMNS].to_numpy(dtype=float)

            model = load_model(m_key)
            f_preds = []

            if m_key == "linear_regression":
                coef = np.asarray(model.coef_, dtype=np.float64)
                ic = float(model.intercept_)
                buf = np.empty(17, dtype=np.float64)
                buf[11:17] = last_w
                for ts in fut_dts:
                    buf[0] = ts.year; buf[1] = ts.month; buf[2] = ts.day
                    buf[3] = ts.hour; buf[4] = ts.minute; buf[5] = ts.dayofweek
                    buf[6] = 1.0 if ts.dayofweek in (5, 6) else 0.0
                    buf[7] = rec_demands[-1]
                    buf[8] = rec_demands[-12]
                    buf[9] = rec_demands[-288]
                    buf[10] = rec_demands[-2016]
                    pv = float(np.dot(buf, coef) + ic)
                    f_preds.append(pv)
                    rec_demands.append(pv)
            elif m_key == "random_forest":
                trees = [e.tree_ for e in model.estimators_]
                nt = len(trees)
                f_buf = np.empty((1, 17), dtype=np.float32)
                f_buf[0, 11:17] = last_w
                for ts in fut_dts:
                    f_buf[0, 0] = ts.year; f_buf[0, 1] = ts.month; f_buf[0, 2] = ts.day
                    f_buf[0, 3] = ts.hour; f_buf[0, 4] = ts.minute; f_buf[0, 5] = ts.dayofweek
                    f_buf[0, 6] = 1.0 if ts.dayofweek in (5, 6) else 0.0
                    f_buf[0, 7] = rec_demands[-1]
                    f_buf[0, 8] = rec_demands[-12]
                    f_buf[0, 9] = rec_demands[-288]
                    f_buf[0, 10] = rec_demands[-2016]
                    pv = float(sum(t.predict(f_buf)[0, 0] for t in trees) / nt)
                    f_preds.append(pv)
                    rec_demands.append(pv)
            else:
                seq_buf = []
                for idx in range(-6, 0):
                    rh = history.iloc[idx]
                    th = rh["datetime"]
                    fv = np.empty(17, dtype=np.float64)
                    fv[0] = th.year; fv[1] = th.month; fv[2] = th.day
                    fv[3] = th.hour; fv[4] = th.minute; fv[5] = th.dayofweek
                    fv[6] = 1.0 if th.dayofweek in (5, 6) else 0.0
                    pos = len(rec_demands) + idx
                    fv[7] = rec_demands[pos - 1]
                    fv[8] = rec_demands[pos - 12]
                    fv[9] = rec_demands[pos - 288]
                    fv[10] = rec_demands[pos - 2016]
                    fv[11:17] = rh[WEATHER_COLUMNS].to_numpy(dtype=float)
                    seq_buf.append(fv)

                for ts in fut_dts:
                    inp3 = np.array(seq_buf, dtype=np.float64).reshape(1, 6, 17)
                    pv = float(model.predict(inp3)[0])
                    f_preds.append(pv)
                    rec_demands.append(pv)

                    f_new = np.empty(17, dtype=np.float64)
                    f_new[0] = ts.year; f_new[1] = ts.month; f_new[2] = ts.day
                    f_new[3] = ts.hour; f_new[4] = ts.minute; f_new[5] = ts.dayofweek
                    f_new[6] = 1.0 if ts.dayofweek in (5, 6) else 0.0
                    f_new[7] = rec_demands[-1]
                    f_new[8] = rec_demands[-12]
                    f_new[9] = rec_demands[-288]
                    f_new[10] = rec_demands[-2016]
                    f_new[11:17] = last_w
                    seq_buf.pop(0)
                    seq_buf.append(f_new)

            elapsed = perf_counter() - t_start
            st.session_state["chase_fc"] = {
                "df": pd.DataFrame({"datetime": fut_dts, "predicted_demand_mw": f_preds}),
                "model": selected_engine,
                "horizon": chosen_h,
                "elapsed": elapsed
            }

    st.markdown("</div>", unsafe_allow_html=True)

    if "chase_fc" in st.session_state:
        fc = st.session_state["chase_fc"]
        df_res = fc["df"]

        # KPI Summary Pills
        st.markdown(f"""
        <div class="kpi-pastel-grid">
          <div class="kpi-pastel-card kpi-cyan">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">Mean Projected</span>
            </div>
            <div>
              <div class="kpi-pastel-val">{df_res['predicted_demand_mw'].mean():,.1f} MW</div>
              <div class="kpi-pastel-label">Average Load</div>
            </div>
          </div>
          <div class="kpi-pastel-card kpi-coral">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">Peak Forecast</span>
            </div>
            <div>
              <div class="kpi-pastel-val">{df_res['predicted_demand_mw'].max():,.1f} MW</div>
              <div class="kpi-pastel-label">Maximum Projected</div>
            </div>
          </div>
          <div class="kpi-pastel-card kpi-mint">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">Minimum</span>
            </div>
            <div>
              <div class="kpi-pastel-val">{df_res['predicted_demand_mw'].min():,.1f} MW</div>
              <div class="kpi-pastel-label">Trough Baseload</div>
            </div>
          </div>
          <div class="kpi-pastel-card kpi-purple">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">Performance</span>
            </div>
            <div>
              <div class="kpi-pastel-val">{fc['elapsed']:.2f} s</div>
              <div class="kpi-pastel-label">{len(df_res):,} Forward Steps</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Plotly Forward Projection
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Forecast Trajectory</div>
              <div class="chase-card-subtitle">Trailing Telemetry Context & Forward AI Projection</div>
            </div>
          </div>
        """, unsafe_allow_html=True)

        ctx_tail = history.tail(288)
        fig_fc_plot = go.Figure()
        fig_fc_plot.add_trace(go.Scatter(
            x=ctx_tail["datetime"],
            y=ctx_tail["Power demand"],
            mode="lines",
            name="Historical Context",
            line=dict(color="#4F46E5", width=2.2, shape="spline")
        ))
        fig_fc_plot.add_trace(go.Scatter(
            x=df_res["datetime"],
            y=df_res["predicted_demand_mw"],
            mode="lines",
            name="Forward Projection",
            line=dict(color="#06B6D4", width=2.8, shape="spline", dash="dot")
        ))
        build_chase_chart(fig_fc_plot, height=300, y_title="Megawatts (MW)")
        st.plotly_chart(fig_fc_plot, use_container_width=True)

        st.download_button(
            "📥 Download Forecast CSV",
            df_res.to_csv(index=False).encode(),
            "chase_delhi_forecast.csv",
            "text/csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 6: WEATHER SCENARIOS
# ==============================================================================
elif nav_selection.startswith("🌦️"):
    render_topbar("Weather Scenarios", "Atmospheric & Heat-Index Simulator")

    st.markdown("""
    <div class="chase-card">
      <div class="chase-card-header">
        <div>
          <div class="chase-card-title">Scenario Parameters</div>
          <div class="chase-card-subtitle">Atmospheric stress simulation presets & live Open-Meteo telemetry</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    sp_c1, sp_c2 = st.columns([1.5, 1.0])
    with sp_c1:
        preset_name = st.selectbox("Select Simulation Preset:", list(SCENARIO_PRESETS.keys()))
        preset_vals = SCENARIO_PRESETS[preset_name]
        st.caption(preset_vals["desc"])
    with sp_c2:
        if st.button("🌐 Fetch Live New Delhi Weather (Open-Meteo)", use_container_width=True):
            live_w = fetch_optional_delhi_weather(timeout_secs=3.0)
            if live_w:
                st.session_state["chase_live_w"] = live_w
                st.success(f"Live Weather: {live_w['temp']}°C, {live_w['rhum']}% RH, {live_w['pres']} hPa")
            else:
                st.warning("Could not reach weather endpoint; preset remains active.")

    w_src = st.session_state.get("chase_live_w", preset_vals)

    w1, w2, w3 = st.columns(3)
    with w1:
        t_in = st.slider("Ambient Temperature (°C)", -5.0, 55.0, float(w_src.get("temp", 35.0)), 0.5)
        d_in = st.slider("Dew Point (°C)", -20.0, 35.0, float(w_src.get("dwpt", 20.0)), 0.5)
    with w2:
        rh_in = st.slider("Relative Humidity (%)", 0.0, 100.0, float(w_src.get("rhum", 45.0)), 1.0)
        p_in = st.slider("Surface Pressure (hPa)", 950.0, 1050.0, float(w_src.get("pres", 998.0)), 0.5)
    with w3:
        ws_in = st.slider("Wind Speed (km/h)", 0.0, 100.0, float(w_src.get("wspd", 12.0)), 0.5)
        wd_in = st.slider("Wind Direction (°)", 0.0, 360.0, float(w_src.get("wdir", 270.0)), 5.0)

    hi = calculate_heat_index(t_in, rh_in)
    st.info(f"🌡️ **Calculated Apparent Heat Index:** **{hi:.1f} °C** (Wind: {get_wind_direction_label(wd_in)})")

    model_sel = st.selectbox("Select Model Engine for Simulation:", ["Linear Regression (Baseline)", "Random Forest (100 Trees)"] + (["PyTorch LSTM"] if is_model_available("lstm") else []))
    m_choice_k = "linear_regression" if "Linear" in model_sel else ("random_forest" if "Random" in model_sel else "lstm")

    if st.button("⚡ Simulate Peak Demand Under This Scenario", type="primary", use_container_width=True):
        f_dict = {
            "year": 2024.0, "month": 6.0, "day": 19.0, "hour": 15.0, "minute": 5.0,
            "day_of_week": 2.0, "is_weekend": 0.0,
            "demand_lag_5min": float(preset_vals.get("demand_lag_5min", 8500.0)),
            "demand_lag_1hour": float(preset_vals.get("demand_lag_1hour", 8200.0)),
            "demand_lag_24hour": float(preset_vals.get("demand_lag_24hour", 8100.0)),
            "demand_lag_7day": float(preset_vals.get("demand_lag_7day", 7800.0)),
            "temp": t_in, "dwpt": d_in, "rhum": rh_in, "pres": p_in, "wspd": ws_in, "wdir": wd_in
        }
        f_row = make_custom_feature_row(f_dict)
        mdl = load_model(m_choice_k)
        sim_mw = float(mdl.predict(f_row)[0])
        regime = classify_demand_regime(sim_mw)

        st.markdown(f"""
        <div class="forecast-result-card">
          <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#A5B4FC;">Simulated Grid Load Result</div>
          <div style="font-size:2.8rem;font-weight:800;margin:6px 0;">{sim_mw:,.2f} <span style="font-size:1.2rem;color:#818CF8;">MW</span></div>
          <div style="font-size:0.95rem;margin-top:6px;">
            Operating Tier: <b>{regime['badge']}</b> &bull; Stress: <b>{regime['stress']}</b>
          </div>
          <div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.2);font-size:0.85rem;color:#E0E7FF;">
            🛡️ <b>Operational Advisory:</b> {regime['advice']}
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 7: PEAK ANALYSIS (Historical Peaks & Benchmarks)
# ==============================================================================
elif nav_selection.startswith("🏔️"):
    render_topbar("Peak Analysis", "272 Monitored Daily Peaks")

    dp_file = ROOT / "outputs" / "peak_analysis" / "daily_peak_analysis.csv"
    if dp_file.exists():
        dp_df = pd.read_csv(dp_file)

        st.markdown("""
        <div class="kpi-pastel-grid">
          <div class="kpi-pastel-card kpi-coral">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">Record High</span>
            </div>
            <div>
              <div class="kpi-pastel-val">8,631.5 MW</div>
              <div class="kpi-pastel-label">19 June 2024 at 15:05 IST</div>
            </div>
          </div>
          <div class="kpi-pastel-card kpi-mint">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">Linear Reg Peak</span>
            </div>
            <div>
              <div class="kpi-pastel-val">8,560.4 MW</div>
              <div class="kpi-pastel-label">Peak Error: -71.1 MW (-0.8%)</div>
            </div>
          </div>
          <div class="kpi-pastel-card kpi-peach">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">Random Forest</span>
            </div>
            <div>
              <div class="kpi-pastel-val">7,677.3 MW</div>
              <div class="kpi-pastel-label">Peak Error: -954.2 MW (-11.1%)</div>
            </div>
          </div>
          <div class="kpi-pastel-card kpi-purple">
            <div class="kpi-pastel-top">
              <span style="font-size:0.70rem;font-weight:700;">PyTorch LSTM</span>
            </div>
            <div>
              <div class="kpi-pastel-val">8,179.7 MW</div>
              <div class="kpi-pastel-label">Peak Error: -451.9 MW (-5.2%)</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Daily Peak Demand Over Time</div>
              <div class="chase-card-subtitle">Actual Grid Peaks vs. Multi-Model Predictions</div>
            </div>
          </div>
        """, unsafe_allow_html=True)

        fig_peaks = go.Figure()
        d_times = pd.to_datetime(dp_df["actual_peak_time"])
        fig_peaks.add_trace(go.Scatter(x=d_times, y=dp_df["actual_daily_peak_mw"], mode="lines+markers", name="Actual Daily Peak", line=dict(color="#4F46E5", width=2.5)))
        fig_peaks.add_trace(go.Scatter(x=d_times, y=dp_df["lr_pred_at_actual_peak_mw"], mode="lines", name="Linear Regression", line=dict(color="#EF4444", width=1.4, dash="dash")))
        fig_peaks.add_trace(go.Scatter(x=d_times, y=dp_df["rf_pred_at_actual_peak_mw"], mode="lines", name="Random Forest", line=dict(color="#10B981", width=1.4, dash="dot")))
        if "lstm_pred_at_actual_peak_mw" in dp_df.columns:
            fig_peaks.add_trace(go.Scatter(x=d_times, y=dp_df["lstm_pred_at_actual_peak_mw"], mode="lines", name="PyTorch LSTM", line=dict(color="#A855F7", width=1.6, dash="dashdot")))

        build_chase_chart(fig_peaks, height=320, y_title="Peak Demand (MW)")
        st.plotly_chart(fig_peaks, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Top 10 Peaks Table
        st.markdown("""
        <div class="chase-card">
          <div class="chase-card-header">
            <div>
              <div class="chase-card-title">Top 10 All-Time Highest Daily Peaks (Test Period)</div>
              <div class="chase-card-subtitle">Peak demand records during summer heatwaves</div>
            </div>
          </div>
        """, unsafe_allow_html=True)
        top10 = dp_df.nlargest(10, "actual_daily_peak_mw")[["actual_peak_time", "actual_daily_peak_mw", "lr_pred_at_actual_peak_mw", "rf_pred_at_actual_peak_mw", "lstm_pred_at_actual_peak_mw"]]
        top10.columns = ["Peak Timestamp", "Actual Peak (MW)", "Linear Reg (MW)", "Random Forest (MW)", "LSTM (MW)"]
        st.dataframe(top10, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 8: REPORTS & AUDIT
# ==============================================================================
elif nav_selection.startswith("📑"):
    render_topbar("Reports & Audit", "Verified Project Artifacts")

    st.markdown("""
    <div class="chase-card">
      <div class="chase-card-header">
        <div>
          <div class="chase-card-title">Downloadable Reports & Verified Data Artifacts</div>
          <div class="chase-card-subtitle">Scientific evaluation benchmarks, data quality audits, and offline dashboards</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    r_c1, r_c2 = st.columns(2)
    with r_c1:
        st.markdown("**1. Multi-Model Benchmark Comparison**")
        comp_f = ROOT / "model_comparison_delhi_powerdemand.csv"
        if comp_f.exists():
            with open(comp_f, "rb") as f:
                st.download_button("📥 Download Model Comparison CSV", f.read(), "model_comparison_delhi_powerdemand.csv", "text/csv")

        st.markdown("<br>**2. Daily Peak Analysis CSV (272 Days)**", unsafe_allow_html=True)
        dp_f = ROOT / "outputs" / "peak_analysis" / "daily_peak_analysis.csv"
        if dp_f.exists():
            with open(dp_f, "rb") as f:
                st.download_button("📥 Download Daily Peak Analysis CSV", f.read(), "daily_peak_analysis.csv", "text/csv")

    with r_c2:
        st.markdown("**3. Peak Model Performance CSV**")
        pmp_f = ROOT / "outputs" / "peak_analysis" / "peak_model_performance.csv"
        if pmp_f.exists():
            with open(pmp_f, "rb") as f:
                st.download_button("📥 Download Peak Performance CSV", f.read(), "peak_model_performance.csv", "text/csv")

        st.markdown("<br>**4. Standalone Offline HTML Dashboard**", unsafe_allow_html=True)
        html_f = ROOT / "outputs" / "project_dashboard" / "project_dashboard.html"
        if html_f.exists():
            with open(html_f, "rb") as f:
                st.download_button("🌐 Download Offline HTML Dashboard", f.read(), "project_dashboard.html", "text/html")
        else:
            st.caption("Generate with `python project_results_dashboard.py`")

    st.markdown("</div>", unsafe_allow_html=True)
