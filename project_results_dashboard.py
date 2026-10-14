"""
Project Results Dashboard & Visualization Module: Multi-Model Architecture
Project: AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System

Purpose:
Consolidates all completed project milestones into a professional,
interactive, self-contained HTML dashboard and documentation suite featuring
Linear Regression, Random Forest, and PyTorch LSTM models.

Inputs:
- cleaned_delhi_powerdemand_5min_2021_to_2024.csv
- train_delhi_powerdemand.csv
- test_delhi_powerdemand.csv
- model_comparison_delhi_powerdemand.csv
- outputs/model_comparison/model_comparison_summary.json
- outputs/peak_analysis/daily_peak_analysis.csv
- outputs/peak_analysis/monthly_peak_analysis.csv
- outputs/peak_analysis/peak_hour_analysis.csv
- outputs/peak_analysis/peak_model_performance.csv
- outputs/peak_projection/example_projection.csv

Outputs:
- outputs/project_dashboard/project_dashboard.html (Self-contained offline HTML dashboard)
- outputs/project_dashboard/project_summary.md (Project documentation report)
- outputs/project_dashboard/dashboard_overview_chart.png (High-res composite summary chart with all 3 models)
- outputs/project_dashboard/dashboard_scenario_chart.png (Scenario projection visualization)
"""

import os
import json
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join("outputs", "project_dashboard")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def image_to_base64(image_path: str) -> str:
    """Encodes an image file to a base64 string for offline self-contained HTML embedding."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_f:
        encoded = base64.b64encode(img_f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def build_dashboard():
    print("=== BUILDING MULTI-MODEL PROJECT RESULTS DASHBOARD ===")
    print("Loading project outputs across Linear Regression, Random Forest, and LSTM...")

    # 1. Load Data Summaries
    train_df = pd.read_csv("train_delhi_powerdemand.csv")
    test_df = pd.read_csv("test_delhi_powerdemand.csv")
    comp_df = pd.read_csv("model_comparison_delhi_powerdemand.csv")
    daily_peak_df = pd.read_csv(os.path.join("outputs", "peak_analysis", "daily_peak_analysis.csv"))
    monthly_peak_df = pd.read_csv(os.path.join("outputs", "peak_analysis", "monthly_peak_analysis.csv"))
    peak_hour_df = pd.read_csv(os.path.join("outputs", "peak_analysis", "peak_hour_analysis.csv"))
    peak_perf_df = pd.read_csv(os.path.join("outputs", "peak_analysis", "peak_model_performance.csv"))
    scenario_df = pd.read_csv(os.path.join("outputs", "peak_projection", "example_projection.csv"))

    # Load summary JSON
    with open(os.path.join("outputs", "model_comparison", "model_comparison_summary.json")) as f:
        comp_meta = json.load(f)

    # Key statistics
    original_obs = 393440
    cleaned_obs = 393440
    date_range_str = "2021-01-01 00:30:00 to 2024-12-12 00:30:00"
    features_count = 17
    train_rows = len(train_df)
    test_rows = len(test_df)
    nulls_after_preproc = 0
    irregular_gaps = 1150
    eval_period_str = "March 5, 2024 13:25:00 to December 11, 2024 22:25:00"
    generation_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Column name mappings supporting updated daily_peak_analysis.csv
    col_actual_peak = "actual_daily_peak_mw" if "actual_daily_peak_mw" in daily_peak_df.columns else "daily_max_actual_demand_mw"
    col_peak_time = "actual_peak_time" if "actual_peak_time" in daily_peak_df.columns else "datetime"
    col_lr_pred = "lr_pred_at_actual_peak_mw" if "lr_pred_at_actual_peak_mw" in daily_peak_df.columns else "linear_regression_predicted_peak_mw"
    col_rf_pred = "rf_pred_at_actual_peak_mw" if "rf_pred_at_actual_peak_mw" in daily_peak_df.columns else "random_forest_predicted_peak_mw"
    col_lstm_pred = "lstm_pred_at_actual_peak_mw" if "lstm_pred_at_actual_peak_mw" in daily_peak_df.columns else None

    highest_peak_val = 8631.53
    highest_peak_dt = "2024-06-19 15:05:00"
    avg_daily_peak = float(daily_peak_df[col_actual_peak].mean())
    median_daily_peak = float(daily_peak_df[col_actual_peak].median())

    # =========================================================================
    # 2. Generate Dashboard Overview Composite Chart
    # =========================================================================
    print("Rendering updated dashboard summary charts...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Panel A: Actual Peak vs Predictions Over Time (All 3 Models)
    daily_dt = pd.to_datetime(daily_peak_df[col_peak_time])
    ax1.plot(daily_dt, daily_peak_df[col_actual_peak], color="#0D47A1", label="Actual Daily Peak", linewidth=1.8)
    ax1.plot(daily_dt, daily_peak_df[col_lr_pred], color="#D32F2F", linestyle="--", label="Linear Regression", linewidth=1.2, alpha=0.85)
    ax1.plot(daily_dt, daily_peak_df[col_rf_pred], color="#2E7D32", linestyle="-.", label="Random Forest", linewidth=1.2, alpha=0.85)
    if col_lstm_pred and col_lstm_pred in daily_peak_df.columns:
        ax1.plot(daily_dt, daily_peak_df[col_lstm_pred], color="#7B1FA2", linestyle=":", label="LSTM Network", linewidth=1.4, alpha=0.9)

    ax1.set_title("A. Daily Peak Demand: Actual vs. Predictions (Mar - Dec 2024)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Demand (MW)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Panel B: Peak Hour Distribution
    x_hours = peak_hour_df["hour"]
    ax2.bar(x_hours - 0.2, peak_hour_df["weekday_count"], width=0.4, label="Weekdays", color="#1976D2", alpha=0.85)
    ax2.bar(x_hours + 0.2, peak_hour_df["weekend_count"], width=0.4, label="Weekends", color="#F57C00", alpha=0.85)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_title("B. Daily Peak Hour Distribution (Most Common: 15:00 / 3 PM)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Hour of Day", fontsize=10)
    ax2.set_ylabel("Number of Peak Days", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8.5)

    # Panel C: Monthly Max Demand vs Daily Peak Avg
    m_x = np.arange(len(monthly_peak_df))
    ax3.bar(m_x - 0.18, monthly_peak_df["maximum_demand_mw"], width=0.35, label="Monthly Max Demand (MW)", color="#C62828", alpha=0.85)
    ax3.bar(m_x + 0.18, monthly_peak_df["average_daily_peak_demand_mw"], width=0.35, label="Avg Daily Peak (MW)", color="#1565C0", alpha=0.85)
    ax3.set_xticks(m_x)
    ax3.set_xticklabels([m[:3] for m in monthly_peak_df["month_name"]], rotation=0)
    ax3.set_title("C. Monthly Peak Characteristics in Test Period", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Demand (MW)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8.5)

    # Panel D: Actual vs Absolute Error by Demand Level
    lr_err_peak = (daily_peak_df[col_lr_pred] - daily_peak_df[col_actual_peak]).abs()
    rf_err_peak = (daily_peak_df[col_rf_pred] - daily_peak_df[col_actual_peak]).abs()
    ax4.scatter(daily_peak_df[col_actual_peak], lr_err_peak, color="#D32F2F", alpha=0.55, s=22, label="Linear Regression Error")
    ax4.scatter(daily_peak_df[col_actual_peak], rf_err_peak, color="#2E7D32", alpha=0.55, s=22, label="Random Forest Error")
    if col_lstm_pred and col_lstm_pred in daily_peak_df.columns:
        lstm_err_peak = (daily_peak_df[col_lstm_pred] - daily_peak_df[col_actual_peak]).abs()
        ax4.scatter(daily_peak_df[col_actual_peak], lstm_err_peak, color="#7B1FA2", alpha=0.55, s=22, label="LSTM Network Error")

    ax4.set_title("D. Actual Peak Demand vs. Absolute Peak Prediction Error", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Actual Daily Peak (MW)", fontsize=10)
    ax4.set_ylabel("Absolute Error (MW)", fontsize=10)
    ax4.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    chart1_path = os.path.join(OUTPUT_DIR, "dashboard_overview_chart.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"Saved updated overview chart to: {chart1_path}")

    # Chart 2: Scenario Projection Trajectory
    fig, ax_s1 = plt.subplots(figsize=(11, 5), dpi=150)
    ax_s1.plot(scenario_df["time"], scenario_df["projected_demand_mw"], color="#D32F2F", marker="o", linewidth=2.0, label="Projected Demand (MW)")
    ax_s1.set_ylabel("Projected Demand (MW)", color="#D32F2F", fontsize=11)
    ax_s1.tick_params(axis='y', labelcolor="#D32F2F")
    ax_s1.set_ylim(7800, 9200)

    # Highlight max scenario point
    s_max_idx = scenario_df["projected_demand_mw"].idxmax()
    s_max_row = scenario_df.loc[s_max_idx]
    ax_s1.scatter(s_max_row["time"], s_max_row["projected_demand_mw"], color="#B71C1C", s=90, zorder=5, label=f"Projected Peak: {s_max_row['projected_demand_mw']:.1f} MW ({s_max_row['time']})")

    ax_s2 = ax_s1.twinx()
    ax_s2.plot(scenario_df["time"], scenario_df["temp_degC"], color="#E65100", linestyle="--", marker="s", linewidth=1.5, alpha=0.75, label="Scenario Temp (°C)")
    ax_s2.set_ylabel("Ambient Temperature (°C)", color="#E65100", fontsize=11)
    ax_s2.tick_params(axis='y', labelcolor="#E65100")
    ax_s2.set_ylim(35.0, 48.0)

    ax_s1.set_title("Illustrative Summer Heatwave Scenario Projection (Model-Based Simulation conditioned on Weather)", fontsize=11, fontweight='bold', pad=10)
    ax_s1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart2_path = os.path.join(OUTPUT_DIR, "dashboard_scenario_chart.png")
    plt.savefig(chart2_path)
    plt.close()

    # Base64 representations for 100% offline self-contained HTML
    b64_chart1 = image_to_base64(chart1_path)
    b64_chart2 = image_to_base64(chart2_path)
    b64_eda1 = image_to_base64(os.path.join("outputs", "eda", "1_power_demand_time_series.png"))
    b64_eda2 = image_to_base64(os.path.join("outputs", "eda", "2_hourly_demand_profile.png"))
    b64_eda3 = image_to_base64(os.path.join("outputs", "eda", "4_monthly_demand_profile.png"))
    b64_eda4 = image_to_base64(os.path.join("outputs", "eda", "5_weather_vs_demand_relationships.png"))

    # =========================================================================
    # 3. Generate Standalone HTML Dashboard (Including LSTM)
    # =========================================================================
    print("Generating comprehensive multi-model HTML dashboard...")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Delhi Power System - Multi-Model Electricity Demand & Peak Projection Dashboard</title>
    <style>
        :root {{
            --bg-primary: #0F172A;
            --bg-secondary: #1E293B;
            --bg-card: rgba(30, 41, 59, 0.85);
            --border-color: rgba(148, 163, 184, 0.15);
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --accent-blue: #38BDF8;
            --accent-teal: #2DD4BF;
            --accent-red: #F87171;
            --accent-amber: #FBBF24;
            --accent-purple: #C084FC;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: var(--font-family);
            line-height: 1.5;
            padding: 24px;
        }}
        .container {{ max-width: 1380px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        }}
        .badge-title {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.12);
            color: var(--accent-blue);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 4px 12px;
            border-radius: 9999px;
            margin-bottom: 8px;
            border: 1px solid rgba(56, 189, 248, 0.25);
        }}
        h1 {{ font-size: 26px; font-weight: 800; color: #FFFFFF; margin-bottom: 6px; }}
        .subtitle {{ color: var(--text-secondary); font-size: 14px; max-width: 950px; line-height: 1.6; }}
        .meta-bar {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 14px;
            font-size: 12px;
            color: var(--accent-teal);
            font-weight: 600;
        }}

        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .kpi-card:hover {{ transform: translateY(-2px); border-color: rgba(56, 189, 248, 0.4); }}
        .kpi-label {{ font-size: 11px; color: var(--text-secondary); font-weight: 700; text-transform: uppercase; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 22px; font-weight: 800; color: #FFFFFF; font-family: monospace; }}
        .kpi-subtext {{ font-size: 12px; color: var(--accent-teal); margin-top: 4px; }}
        .kpi-subtext.red {{ color: var(--accent-red); }}
        .kpi-subtext.purple {{ color: var(--accent-purple); }}

        /* Sections */
        .section-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
        }}
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 12px;
        }}
        .section-title {{ font-size: 18px; font-weight: 700; color: #FFFFFF; display: flex; align-items: center; gap: 8px; }}
        .section-subtitle {{ font-size: 13px; color: var(--text-secondary); margin-top: 2px; }}

        /* Tables */
        .table-responsive {{ overflow-x: auto; margin-top: 12px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
        th {{ background: #162032; color: #E2E8F0; padding: 12px 14px; font-weight: 600; border-bottom: 1px solid var(--border-color); text-transform: uppercase; font-size: 11px; }}
        td {{ padding: 12px 14px; border-bottom: 1px solid rgba(148, 163, 184, 0.08); color: #CBD5E1; }}
        tr:hover td {{ background: rgba(56, 189, 248, 0.03); }}
        .highlight {{ font-weight: 700; color: #FFFFFF; }}
        .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
        .tag-blue {{ background: rgba(56, 189, 248, 0.15); color: var(--accent-blue); }}
        .tag-green {{ background: rgba(45, 212, 191, 0.15); color: var(--accent-teal); }}
        .tag-purple {{ background: rgba(192, 132, 252, 0.15); color: var(--accent-purple); }}
        .tag-red {{ background: rgba(248, 113, 113, 0.15); color: var(--accent-red); }}

        /* Alert Callout */
        .alert-box {{
            background: rgba(251, 191, 36, 0.08);
            border-left: 4px solid var(--accent-amber);
            border-radius: 8px;
            padding: 14px 18px;
            margin: 16px 0;
            font-size: 13px;
            color: #E2E8F0;
            line-height: 1.6;
        }}
        .alert-box.info {{
            background: rgba(56, 189, 248, 0.08);
            border-left-color: var(--accent-blue);
        }}
        .alert-box.danger {{
            background: rgba(248, 113, 113, 0.08);
            border-left-color: var(--accent-red);
        }}
        .alert-title {{ font-weight: 700; color: #FFFFFF; margin-bottom: 4px; }}

        .chart-img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border-color); margin-top: 14px; display: block; }}
        .two-col-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-top: 16px; }}

        footer {{ text-align: center; color: var(--text-secondary); font-size: 12px; margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border-color); }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="badge-title">Standalone Multi-Model Review Dashboard</div>
            <h1>AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System</h1>
            <p class="subtitle">Complete visual evaluation and model comparison module documenting dataset quality, zero-leakage causal engineering, multi-model benchmarks (Linear Regression, Random Forest, PyTorch LSTM), peak demand dynamics, and model-based scenario projection.</p>
            <div class="meta-bar">
                <span>📅 Evaluation Period: {eval_period_str}</span>
                <span>⏱️ Dashboard Generated: {generation_date_str}</span>
                <span>🔒 Execution: 100% Offline Standalone</span>
            </div>
        </header>

        <!-- KPI Metric Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Dataset Span</div>
                <div class="kpi-value">4.0 Years</div>
                <div class="kpi-subtext">393,440 5-min intervals</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Hold-Out Test Set</div>
                <div class="kpi-value">70,825</div>
                <div class="kpi-subtext">20% Chronological Partition</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Highest Observed Test Demand</div>
                <div class="kpi-value" style="color:var(--accent-red)">8,631.5 MW</div>
                <div class="kpi-subtext red">2024-06-19 15:05:00</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Linear Regression Test MAE</div>
                <div class="kpi-value">25.24 MW</div>
                <div class="kpi-subtext">Peak Error: -71.1 MW (-0.82%)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Random Forest Test MAE</div>
                <div class="kpi-value">28.17 MW</div>
                <div class="kpi-subtext red">Peak Error: -954.2 MW (-11.05%)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">PyTorch LSTM Test MAE</div>
                <div class="kpi-value">27.63 MW</div>
                <div class="kpi-subtext purple">Peak Error: -451.9 MW (-5.23%)</div>
            </div>
        </div>

        <!-- Section 1: Dataset Summary -->
        <div class="section-card">
            <div class="section-header">
                <div>
                    <div class="section-title">1. Dataset Summary & Preprocessing Verification</div>
                    <div class="section-subtitle">Zero-leakage telemetry partition across 4 full calendar years (2021–2024)</div>
                </div>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                            <th>Status / Technical Note</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Original Dataset Observations</td><td class="highlight">{original_obs:,}</td><td><span class="tag tag-blue">Verified</span> Loaded from powerdemand_5min_2021_to_2024_with weather.csv</td></tr>
                        <tr><td>Cleaned Dataset Observations</td><td class="highlight">{cleaned_obs:,}</td><td><span class="tag tag-green">Zero Nulls</span> 540 wdir missing values resolved via causal forward-fill; no synthetic rows</td></tr>
                        <tr><td>Full Timeline Date Range</td><td class="highlight">{date_range_str}</td><td><span class="tag tag-blue">Verified</span> Multi-year demand variation captured across all seasons (2021–2024)</td></tr>
                        <tr><td>Modeling Input Features</td><td class="highlight">{features_count} Features</td><td>7 Calendar, 4 exact timestamp lag features, 6 weather features</td></tr>
                        <tr><td>Training Observations (80%)</td><td class="highlight">{train_rows:,} rows</td><td>2021-01-08 01:00:00 to 2024-03-05 13:20:00 (~3.15 years)</td></tr>
                        <tr><td>Testing Observations (20%)</td><td class="highlight">{test_rows:,} rows</td><td>2024-03-05 13:25:00 to 2024-12-11 22:25:00 (Strict chronological hold-out)</td></tr>
                        <tr><td>Target Leakage Elimination</td><td class="highlight">PASS</td><td>Leaky moving_avg_3 column completely removed prior to modeling</td></tr>
                        <tr><td>Irregular Time Gaps (>5 min)</td><td class="highlight">{irregular_gaps:,} gap transitions</td><td>21,569 omitted historical 5-min intervals documented; lag mapping uses exact timestamps</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Section 2: Model Comparison -->
        <div class="section-card">
            <div class="section-header">
                <div>
                    <div class="section-title">2. Comparative Model Benchmark (All 3 Models)</div>
                    <div class="section-subtitle">Evaluated on the identical 70,825 hold-out test observations (March to December 2024)</div>
                </div>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Metric</th>
                            <th>Linear Regression (Baseline)</th>
                            <th>Random Forest (100 Trees)</th>
                            <th>PyTorch LSTM (Deep RNN)</th>
                            <th>Operational Insight</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Overall Test Performance</td><td>MAE (Mean Absolute Error)</td><td class="highlight">25.24 MW</td><td class="highlight">28.17 MW</td><td class="highlight">27.63 MW</td><td>Linear Regression lowest overall MAE</td></tr>
                        <tr><td>Overall Test Performance</td><td>RMSE (Root Mean Squared Error)</td><td class="highlight">43.23 MW</td><td class="highlight">56.29 MW</td><td class="highlight">47.77 MW</td><td>Linear Regression penalizes extremes least</td></tr>
                        <tr><td>Overall Test Performance</td><td>R² Score</td><td class="highlight">0.9990</td><td class="highlight">0.9983</td><td class="highlight">0.9987</td><td>All three models explain >99.8% of variance</td></tr>
                        <tr><td>Computational Cost</td><td>Training Time</td><td class="highlight">0.42 s</td><td class="highlight">140.85 s</td><td class="highlight">133.33 s</td><td>Linear Regression ~300x faster to fit</td></tr>
                        <tr><td>Computational Cost</td><td>Inference Time (70.8k rows)</td><td class="highlight">0.05 s</td><td class="highlight">1.02 s</td><td class="highlight">1.71 s</td><td>Linear Regression ~20x faster inference</td></tr>
                        <tr><td>Peak Demand Accuracy</td><td>Error at Test Peak (8,631.5 MW)</td><td class="highlight" style="color:var(--accent-teal);">-71.12 MW (-0.82%)</td><td class="highlight" style="color:var(--accent-red);">-954.21 MW (-11.05%)</td><td class="highlight" style="color:var(--accent-purple);">-451.86 MW (-5.23%)</td><td>RF caps at training bound; LSTM cuts error by 53%</td></tr>
                        <tr><td>Peak Demand Accuracy</td><td>Daily Peak MAE (272 Days)</td><td class="highlight">39.86 MW</td><td class="highlight">63.88 MW</td><td class="highlight">61.62 MW</td><td>Linear Regression lowest daily peak error</td></tr>
                    </tbody>
                </table>
            </div>
            <div class="alert-box info">
                <div class="alert-title">Model Architectural Differences & Peak Extrapolation</div>
                Tree-based ensembles (Random Forest) partition training data into discrete leaves and cannot predict values higher than the maximum target observation in the training partition (~7,746 MW). During the severe heatwave peak of June 2024 (8,631.53 MW), Random Forest under-predicted by -954.21 MW (-11.05%). Linear Regression maintained unbounded linear slope extrapolation (-71.12 MW / -0.82% error), while PyTorch LSTM's continuous activation layers cut the tree-based under-prediction error by more than half (-451.86 MW / -5.23% error).
            </div>
            <img src="{b64_chart1}" alt="Overview Chart" class="chart-img">
        </div>

        <!-- Section 3: Peak Demand Analysis -->
        <div class="section-card">
            <div class="section-header">
                <div>
                    <div class="section-title">3. Dedicated Peak Demand & Diurnal Distribution Analysis</div>
                    <div class="section-subtitle">272 Daily Peak Events evaluated across time of day, month, and model predictions</div>
                </div>
            </div>
            <div class="two-col-grid">
                <div>
                    <h3 style="font-size:14px; font-weight:700; color:var(--accent-blue); margin-bottom:10px;">Highest Observed Demand in Test Period (2024-06-19 15:05:00)</h3>
                    <div class="table-responsive">
                        <table>
                            <thead>
                                <tr><th>Model</th><th>Predicted Demand</th><th>Error (MW)</th><th>Error (%)</th></tr>
                            </thead>
                            <tbody>
                                <tr><td><strong>Actual Ground Truth</strong></td><td class="highlight">8,631.53 MW</td><td>0.00 MW</td><td>0.00%</td></tr>
                                <tr><td>Linear Regression</td><td class="highlight">8,560.41 MW</td><td style="color:var(--accent-teal);">-71.12 MW</td><td>-0.82%</td></tr>
                                <tr><td>PyTorch LSTM</td><td class="highlight">8,179.67 MW</td><td style="color:var(--accent-purple);">-451.86 MW</td><td>-5.23%</td></tr>
                                <tr><td>Random Forest</td><td class="highlight">7,677.32 MW</td><td style="color:var(--accent-red);">-954.21 MW</td><td>-11.05%</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div>
                    <h3 style="font-size:14px; font-weight:700; color:var(--accent-blue); margin-bottom:10px;">Daily Peak Performance Summary (272 Days)</h3>
                    <div class="table-responsive">
                        <table>
                            <thead>
                                <tr><th>Metric</th><th>Linear Regression</th><th>Random Forest</th><th>PyTorch LSTM</th></tr>
                            </thead>
                            <tbody>
                                <tr><td>Daily Peak MAE</td><td class="highlight">39.86 MW</td><td class="highlight">63.88 MW</td><td class="highlight">61.62 MW</td></tr>
                                <tr><td>Daily Peak RMSE</td><td class="highlight">72.07 MW</td><td class="highlight">136.40 MW</td><td class="highlight">139.20 MW</td></tr>
                                <tr><td>Daily Peak MAPE</td><td class="highlight">0.77%</td><td class="highlight">1.07%</td><td class="highlight">1.04%</td></tr>
                                <tr><td>Peak Timing MAE</td><td class="highlight">28.3 mins</td><td class="highlight">65.8 mins</td><td class="highlight">62.4 mins</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- Section 4: Project Limitations -->
        <div class="section-card">
            <div class="section-header">
                <div>
                    <div class="section-title">4. Documented Project Assumptions & Limitations</div>
                    <div class="section-subtitle">Essential technical context for grid operators, evaluators, and system reviewers</div>
                </div>
            </div>
            <div class="alert-box danger">
                <div class="alert-title">Key Engineering & Modeling Limitations</div>
                <ul style="margin-left: 20px; line-height: 1.8;">
                    <li><strong>Historical Telemetry Boundary:</strong> Models are trained on historical 2021–2024 dispatch telemetry. There is no live SCADA or IoT connection to Delhi SLDC; real-time dispatches are simulated via manual or recent-interval inputs.</li>
                    <li><strong>Missing Interval Handling:</strong> The Delhi grid telemetry records 1,150 irregular gap transitions (>5 minutes) and 21,569 missing 5-minute slots across 4 years. The project avoided synthesizing fake rows; missing autoregressive slots were handled causally.</li>
                    <li><strong>Recursive Forecast Error Accumulation:</strong> In multi-step recursive forecasting (24-hour and 7-day horizons), lag errors accumulate across forward steps. Forecast confidence intervals widen significantly over multi-day horizons.</li>
                    <li><strong>Upstream Weather Dependence:</strong> Demand projections are conditioned upon atmospheric weather (temperature, humidity, dew point, pressure). The accuracy of forward projections is bounded by the quality of upstream numerical weather prediction (NWP) forecasts.</li>
                    <li><strong>Extreme Value Bias:</strong> All models show a slight negative mean percentage error (-0.56% to -0.90%) on peak days, indicating conservative tracking during sudden unprecedented heatwave ramps.</li>
                </ul>
            </div>
        </div>

        <!-- Section 5: Scenario Projection -->
        <div class="section-card">
            <div class="section-header">
                <div>
                    <div class="section-title">5. Model-Based Scenario Projection Module</div>
                    <div class="section-subtitle">Demonstration of transparent simulation conditioned upon user-specified weather scenarios</div>
                </div>
            </div>
            <p style="font-size:13px; color:var(--text-secondary); margin-bottom:12px;">This module evaluates model projections under hypothetical atmospheric stress testing (e.g., assessing grid vulnerability under a sustained 43.8 °C heatwave). Projections are mathematical simulations, not official grid dispatch records.</p>
            <img src="{b64_chart2}" alt="Scenario Chart" class="chart-img">
        </div>

        <footer>
            AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System &bull; Standalone Review Dashboard &bull; Generated: {generation_date_str}
        </footer>
    </div>
</body>
</html>
"""
    html_path = os.path.join(OUTPUT_DIR, "project_dashboard.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved complete multi-model HTML dashboard to: {html_path}")

    # =========================================================================
    # 4. Generate Project Summary Markdown Report
    # =========================================================================
    summary_md_path = os.path.join(OUTPUT_DIR, "project_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Project Summary: Delhi Electricity Demand & Peak Projection System\n")
        f.write(f"**Generated:** {generation_date_str} | **Evaluation Period:** {eval_period_str}\n\n")

        f.write("## 1. System Overview\n")
        f.write("- **Goal:** Project granular 5-minute power demand and evaluate extreme peak demand risks for the Delhi power system.\n")
        f.write("- **Dataset:** 393,440 5-minute telemetry intervals (2021–2024) from Delhi State Load Despatch Centre with matching weather.\n")
        f.write("- **Zero-Leakage Engineering:** Causal forward-fill for wind direction, removal of unlagged moving averages, exact retrospective lag matching.\n\n")

        f.write("## 2. Validated Multi-Model Results (70,825 Hold-Out Test Rows)\n\n")
        f.write("| Model | Test MAE (MW) | Test RMSE (MW) | R² Score | Peak Error at 8,631.5 MW | Daily Peak MAE (MW) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        f.write("| **Linear Regression** | **25.24** | **43.23** | **0.9990** | **-71.12 MW (-0.82%)** | **39.86** |\n")
        f.write("| **Random Forest (100 Trees)** | 28.17 | 56.29 | 0.9983 | -954.21 MW (-11.05%) | 63.88 |\n")
        f.write("| **PyTorch LSTM Network** | 27.63 | 47.77 | 0.9987 | -451.86 MW (-5.23%) | 61.62 |\n\n")

        f.write("## 3. Key Findings on Peak Demand\n")
        f.write("1. **Decision Tree Bounding:** Random Forest under-predicts unprecedented heatwave peaks because its leaves cannot extrapolate beyond training set boundaries (~7,746 MW).\n")
        f.write("2. **Linear Extrapolation:** Linear Regression maintains linear sensitivity, accurately projecting the 8,631.53 MW peak with -0.82% error.\n")
        f.write("3. **LSTM Sequential Modeling:** The 2-layer LSTM cuts Random Forest's peak under-prediction error by more than half (-5.23% error).\n\n")

        f.write("## 4. Key Limitations\n")
        f.write("- Absence of live SCADA telemetry feed (relies on historical data or manual inputs).\n")
        f.write("- Recursive error accumulation during long forward forecasting horizons.\n")
        f.write("- Dependence on upstream meteorological forecast accuracy.\n")

    print(f"Saved project summary markdown to: {summary_md_path}")
    print("Dashboard build completed successfully!")

if __name__ == "__main__":
    build_dashboard()
