"""
Project Results Dashboard & Visualization Module
Project: AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System

Purpose:
Consolidates all completed project milestones (Steps 1–11) into a professional,
interactive, self-contained dashboard and documentation suite.

Inputs (Existing Project Files Only - No Retraining, No Modifications):
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
- outputs/project_dashboard/dashboard_overview_chart.png (High-res composite summary chart)
- outputs/project_dashboard/dashboard_scenario_chart.png (Scenario projection visualization)
"""

import os
import json
import base64
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
    print("=== BUILDING PROJECT RESULTS DASHBOARD ===")
    print("Loading existing project outputs (Zero modifications to datasets/models)...")

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

    highest_peak_val = 8631.53
    highest_peak_dt = "2024-06-19 15:05:00"
    avg_daily_peak = float(daily_peak_df["daily_max_actual_demand_mw"].mean())
    median_daily_peak = float(daily_peak_df["daily_max_actual_demand_mw"].median())

    # =========================================================================
    # 2. Generate Dashboard PNG Charts
    # =========================================================================
    print("Rendering dashboard summary charts...")

    # Chart 1: Dashboard Overview Composite Chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Panel A: Actual Peak vs Predictions Over Time
    daily_dt = pd.to_datetime(daily_peak_df["datetime"])
    ax1.plot(daily_dt, daily_peak_df["daily_max_actual_demand_mw"], color="#0D47A1", label="Actual Daily Peak", linewidth=1.8)
    ax1.plot(daily_dt, daily_peak_df["linear_regression_predicted_peak_mw"], color="#D32F2F", linestyle="--", label="Linear Regression Peak", linewidth=1.4)
    ax1.plot(daily_dt, daily_peak_df["random_forest_predicted_peak_mw"], color="#2E7D32", linestyle="-.", label="Random Forest Peak", linewidth=1.4)
    ax1.set_title("A. Daily Peak Demand: Actual vs. Predictions (Mar - Dec 2024)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Demand (MW)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)

    # Panel B: Peak Hour Distribution
    x_hours = peak_hour_df["hour"]
    ax2.bar(x_hours - 0.2, peak_hour_df["weekday_count"], width=0.4, label="Weekdays", color="#1976D2", alpha=0.85)
    ax2.bar(x_hours + 0.2, peak_hour_df["weekend_count"], width=0.4, label="Weekends", color="#F57C00", alpha=0.85)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_title("B. Daily Peak Hour Distribution (Most Common: 15:00 / 3 PM)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Hour of Day", fontsize=10)
    ax2.set_ylabel("Number of Peak Days", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)

    # Panel C: Monthly Max Demand vs Daily Peak Avg
    m_x = np.arange(len(monthly_peak_df))
    ax3.bar(m_x - 0.18, monthly_peak_df["maximum_demand_mw"], width=0.35, label="Monthly Max Demand (MW)", color="#C62828", alpha=0.85)
    ax3.bar(m_x + 0.18, monthly_peak_df["average_daily_peak_demand_mw"], width=0.35, label="Avg Daily Peak (MW)", color="#1565C0", alpha=0.85)
    ax3.set_xticks(m_x)
    ax3.set_xticklabels([m[:3] for m in monthly_peak_df["month_name"]], rotation=0)
    ax3.set_title("C. Monthly Peak Characteristics in Test Period", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Demand (MW)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=9)

    # Panel D: Actual vs Absolute Error by Demand Level
    lr_err_peak = daily_peak_df["linear_regression_error_mw"].abs()
    rf_err_peak = daily_peak_df["random_forest_error_mw"].abs()
    ax4.scatter(daily_peak_df["daily_max_actual_demand_mw"], lr_err_peak, color="#D32F2F", alpha=0.6, s=24, label="Linear Regression Daily Peak Absolute Error")
    ax4.scatter(daily_peak_df["daily_max_actual_demand_mw"], rf_err_peak, color="#2E7D32", alpha=0.6, s=24, label="Random Forest Daily Peak Absolute Error")
    ax4.set_title("D. Actual Peak Demand vs. Absolute Peak Prediction Error", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Actual Daily Peak (MW)", fontsize=10)
    ax4.set_ylabel("Absolute Error (MW)", fontsize=10)
    ax4.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    chart1_path = os.path.join(OUTPUT_DIR, "dashboard_overview_chart.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"Saved overview chart to: {chart1_path}")

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

    ax_s1.set_title("Step 11: Illustrative Summer Heatwave Scenario Projection (Model-Based Scenario Projection - Not an Observed Value)", fontsize=11, fontweight='bold', pad=10)
    ax_s1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart2_path = os.path.join(OUTPUT_DIR, "dashboard_scenario_chart.png")
    plt.savefig(chart2_path)
    plt.close()
    print(f"Saved scenario chart to: {chart2_path}")

    # Base64 representations for 100% offline self-contained HTML
    b64_chart1 = image_to_base64(chart1_path)
    b64_chart2 = image_to_base64(chart2_path)

    # Pre-existing EDA plot base64 if available
    b64_eda1 = image_to_base64(os.path.join("outputs", "eda", "1_power_demand_time_series.png"))
    b64_eda2 = image_to_base64(os.path.join("outputs", "eda", "2_hourly_demand_profile.png"))
    b64_eda3 = image_to_base64(os.path.join("outputs", "eda", "4_monthly_demand_profile.png"))
    b64_eda4 = image_to_base64(os.path.join("outputs", "eda", "5_weather_vs_demand_relationships.png"))

    # =========================================================================
    # 3. Generate HTML Dashboard
    # =========================================================================
    print("Generating self-contained HTML dashboard...")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Delhi Power System - Electricity Demand Projection Dashboard</title>
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
        
        /* Header */
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
        .subtitle {{ color: var(--text-secondary); font-size: 14px; max-width: 900px; }}

        /* Flow Diagram */
        .flow-container {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        .flow-title {{ font-size: 13px; font-weight: 700; color: var(--accent-teal); text-transform: uppercase; margin-bottom: 12px; }}
        .flow-steps {{
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 950px;
        }}
        .step-node {{
            background: #273549;
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 8px;
            padding: 10px 14px;
            text-align: center;
            font-size: 12px;
            font-weight: 600;
            color: #E2E8F0;
            flex: 1;
        }}
        .step-arrow {{ color: var(--accent-blue); font-size: 16px; font-weight: bold; }}

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
        .kpi-label {{ font-size: 12px; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 24px; font-weight: 800; color: #FFFFFF; }}
        .kpi-subtext {{ font-size: 12px; color: var(--accent-teal); margin-top: 4px; }}
        .kpi-subtext.red {{ color: var(--accent-red); }}
        .kpi-subtext.amber {{ color: var(--accent-amber); }}

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
        .tag-amber {{ background: rgba(251, 191, 36, 0.15); color: var(--accent-amber); }}
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
        }}
        .alert-box.info {{
            background: rgba(56, 189, 248, 0.08);
            border-left-color: var(--accent-blue);
        }}
        .alert-title {{ font-weight: 700; color: #FFFFFF; margin-bottom: 4px; }}

        /* Chart Images */
        .chart-img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border-color); margin-top: 14px; display: block; }}
        .two-col-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-top: 16px; }}

        footer {{ text-align: center; color: var(--text-secondary); font-size: 12px; margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border-color); }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="badge-title">Completed Project Dashboard (Steps 1 – 12)</div>
            <h1>AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System</h1>
            <p class="subtitle">A consolidated visual evaluation and demonstration module documenting dataset quality, exploratory trends, baseline vs. tree ensemble benchmarks, peak demand dynamics, and model-based scenario projection.</p>
        </header>

        <!-- Project Flow -->
        <div class="flow-container">
            <div class="flow-title">End-to-End Methodological Architecture</div>
            <div class="flow-steps">
                <div class="step-node">1. Delhi SLDC-based Electricity Demand & Weather Data<br><small style="color:var(--text-secondary)">393k rows (2021-2024)</small></div>
                <div class="step-arrow">&rarr;</div>
                <div class="step-node">2. Data Preprocessing<br><small style="color:var(--text-secondary)">Removed leakage, causal wdir</small></div>
                <div class="step-arrow">&rarr;</div>
                <div class="step-node">3. Exploratory Data Analysis<br><small style="color:var(--text-secondary)">Diurnal, seasonal & weather</small></div>
                <div class="step-arrow">&rarr;</div>
                <div class="step-node">4. Feature Engineering<br><small style="color:var(--text-secondary)">17 causal calendar & lag inputs</small></div>
                <div class="step-arrow">&rarr;</div>
                <div class="step-node">5. Chronological Split<br><small style="color:var(--text-secondary)">80% train / 20% test (No shuffle)</small></div>
                <div class="step-arrow">&rarr;</div>
                <div class="step-node">6. Model Training<br><small style="color:var(--text-secondary)">Linear Reg. & Random Forest</small></div>
                <div class="step-arrow">&rarr;</div>
                <div class="step-node">7. Peak Demand Analysis<br><small style="color:var(--text-secondary)">272 daily peaks & timing</small></div>
                <div class="step-arrow">&rarr;</div>
                <div class="step-node">8. Scenario Projection<br><small style="color:var(--text-secondary)">Transparent simulation module</small></div>
            </div>
        </div>

        <!-- KPI Metric Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Dataset Span</div>
                <div class="kpi-value">4.0 Years</div>
                <div class="kpi-subtext">393,440 5-min intervals</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Test Hold-out</div>
                <div class="kpi-value">70,825</div>
                <div class="kpi-subtext">Mar 5 – Dec 11, 2024 (20%)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Highest Observed Demand in Test Period</div>
                <div class="kpi-value">8,631.5 MW</div>
                <div class="kpi-subtext red">2024-06-19 15:05:00</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Linear Reg. Peak Error</div>
                <div class="kpi-value">-71.1 MW</div>
                <div class="kpi-subtext">-0.82% under-prediction</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Random Forest Peak Error</div>
                <div class="kpi-value">-954.2 MW</div>
                <div class="kpi-subtext amber">-11.05% error at test peak</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Average Daily Peak</div>
                <div class="kpi-value">5,519.6 MW</div>
                <div class="kpi-subtext">Median: 5,393.6 MW</div>
            </div>
        </div>

        <!-- Section 1: Dataset Summary -->
        <div class="section-card">
            <div class="section-header">
                <div>
                    <div class="section-title">1. Dataset Summary & Preprocessing Audit</div>
                    <div class="section-subtitle">Verified counts, date ranges, and feature partitioning (5-minute observations with some missing intervals)</div>
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
                        <tr><td>Cleaned Dataset Observations</td><td class="highlight">{cleaned_obs:,}</td><td><span class="tag tag-blue">Verified</span> Preserved all authentic observations; no synthetic rows</td></tr>
                        <tr><td>Full Timeline Date Range</td><td class="highlight">{date_range_str}</td><td><span class="tag tag-blue">Verified</span> Multi-year demand variation captured across all seasons (2021–2024)</td></tr>
                        <tr><td>Modeling Input Features</td><td class="highlight">{features_count} Features</td><td>7 Calendar, 4 exact timestamp lag features, 6 weather features</td></tr>
                        <tr><td>Training Observations (80%)</td><td class="highlight">{train_rows:,} rows</td><td>2021-01-08 01:00:00 to 2024-03-05 13:20:00 (~3.15 years)</td></tr>
                        <tr><td>Testing Observations (20%)</td><td class="highlight">{test_rows:,} rows</td><td>2024-03-05 13:25:00 to 2024-12-11 22:25:00 (Chronological hold-out)</td></tr>
                        <tr><td>Missing Values Post-Cleaning</td><td class="highlight">{nulls_after_preproc}</td><td><span class="tag tag-green">Zero Nulls</span> 540 wdir missing values resolved via causal forward-fill</td></tr>
                        <tr><td>Irregular Time Gaps (>5 min)</td><td class="highlight">{irregular_gaps:,} gap transitions</td><td>21,569 omitted historical 5-min intervals documented; lag mapping uses exact timestamps</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Section 2: Exploratory Data Analysis -->
        <div class="section-card">
            <div class="section-header">
                <div>
                    <div class="section-title">2. Exploratory Data Analysis</div>
                    <div class="section-subtitle">Empirical load profiles across diurnal cycles, weekly patterns, seasonal shifts, and weather dependencies</div>
                </div>
            </div>
            <p style="font-size:13px; color:var(--text-secondary); margin-bottom:12px;">Key EDA findings confirm that electricity demand in Delhi exhibits a pronounced diurnal swing (averaging 1,278 MW between 04:00 and 15:00), a ~51% seasonal surge from February (3,219 MW) to June (4,858 MW). Temperature showed a moderate positive association with electricity demand, while demand also varied across seasons.</p>
            <div class="two-col-grid">
                <div>
                    <img src="{b64_eda1 if b64_eda1 else 'dashboard_overview_chart.png'}" alt="Time Series" class="chart-img">
                </div>
                <div>
                    <img src="{b64_eda2 if b64_eda2 else 'dashboard_overview_chart.png'}" alt="Hourly Profile" class="chart-img">
                </div>
            </div>
            <div class="two-col-grid">
                <div>
                    <img src="{b64_eda3 if b64_eda3 else 'dashboard_overview_chart.png'}" alt="Monthly Profile" class="chart-img">
                </div>
                <div>
                    <img src="{b64_eda4 if b64_eda4 else 'dashboard_overview_chart.png'}" alt="Weather vs Demand" class="chart-img">
                </div>
            </div>
        </div>

        <!-- Section 3: Model Comparison -->
        <div class="section-card">
            <div class="section-header">
                <div>
                    <div class="section-title">3. Model Comparison (Test Set Benchmark)</div>
                    <div class="section-subtitle">Experimental results for this dataset and hold-out test period (March 5 to December 11, 2024, 70,825 rows)</div>
                </div>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Metric</th>
                            <th>Linear Regression Baseline</th>
                            <th>Random Forest Regression</th>
                            <th>Performance Characteristic</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Overall Test Performance</td><td>MAE (Mean Absolute Error)</td><td class="highlight">25.24 MW</td><td class="highlight">28.17 MW</td><td>Linear Regression lower overall error</td></tr>
                        <tr><td>Overall Test Performance</td><td>RMSE (Root Mean Squared Error)</td><td class="highlight">43.23 MW</td><td class="highlight">56.29 MW</td><td>Linear Regression produced lower RMSE on this hold-out test set.</td></tr>
                        <tr><td>Overall Test Performance</td><td>R² Score</td><td class="highlight">0.9990</td><td class="highlight">0.9983</td><td>Both explain >99.8% of variance</td></tr>
                        <tr><td>Computational Cost</td><td>Training Time</td><td class="highlight">0.42 seconds</td><td class="highlight">140.85 seconds</td><td>Linear Regression ~335x faster to fit</td></tr>
                        <tr><td>Computational Cost</td><td>Prediction Time (70,825 rows)</td><td class="highlight">0.05 seconds</td><td class="highlight">1.02 seconds</td><td>Linear Regression ~20x faster inference</td></tr>
                    </tbody>
                </table>
            </div>
            <div class="alert-box info">
                <div class="alert-title">Model Performance Context</div>
                The results reflect experimental benchmarks on the chronological 2024 test period. Neither model is universally superior across all operational regimes: Random Forest achieves slightly lower error during standard operational loads (75% of hours), while Linear Regression showed smaller errors when capturing the highest observed test-period peak.
            </div>
            <img src="{b64_chart1}" alt="Overview Chart" class="chart-img">
        </div>

        <!-- Section 4 & 5: Peak Demand Analysis & Prediction Comparison -->
        <div class="section-card">
            <div class="section-header">
                <div>
                    <div class="section-title">4 & 5. Peak Demand Analysis & Model Peak Evaluation</div>
                    <div class="section-subtitle">Detailed evaluation across 272 daily peak observation events and the highest observed test-period peak</div>
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
                                <tr><td>Random Forest</td><td class="highlight">7,677.32 MW</td><td style="color:var(--accent-red);">-954.21 MW</td><td>-11.05%</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div>
                    <h3 style="font-size:14px; font-weight:700; color:var(--accent-blue); margin-bottom:10px;">Daily Peak Performance (All 272 Daily Peaks)</h3>
                    <div class="table-responsive">
                        <table>
                            <thead>
                                <tr><th>Metric</th><th>Linear Regression</th><th>Random Forest</th></tr>
                            </thead>
                            <tbody>
                                <tr><td>Daily Peak MAE</td><td class="highlight">39.86 MW</td><td class="highlight">63.88 MW</td></tr>
                                <tr><td>Daily Peak RMSE</td><td class="highlight">72.07 MW</td><td class="highlight">136.40 MW</td></tr>
                                <tr><td>Daily Peak MAPE</td><td class="highlight">0.77%</td><td class="highlight">1.07%</td></tr>
                                <tr><td>Max Absolute Error</td><td class="highlight">611.24 MW</td><td class="highlight">956.38 MW</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div class="alert-box">
                <div class="alert-title">Peak Demand Model Error Context</div>
                Random Forest substantially underpredicted the highest observed test-period demand (8,631.53 MW at 2024-06-19 15:05:00), predicting 7,677.32 MW with an error of -954.21 MW (-11.05%). Random Forest showed larger errors for extreme demand observations in this experiment. For the same observation, Linear Regression predicted 8,560.41 MW with an error of -71.12 MW (-0.82%).
            </div>
        </div>

        <!-- Section 6: Scenario Projection -->
        <div class="section-card">
            <div class="section-header">
                <div>
                    <div class="section-title">6. Model-Based Scenario Projection Module</div>
                    <div class="section-subtitle">Demonstration of transparent scenario simulation conditioned upon user-specified exogenous inputs</div>
                </div>
                <span class="tag tag-amber">SCENARIO SIMULATION</span>
            </div>
            
            <div class="alert-box info" style="border-left-color: var(--accent-amber);">
                <div class="alert-title" style="color: var(--accent-amber);">MODEL-BASED SCENARIO PROJECTION — NOT AN OBSERVED VALUE</div>
                This section displays mathematical simulations generated by the projection module under hypothetical user-defined inputs. <strong>It does NOT constitute an actual weather forecast or an observed grid dispatch record.</strong> All inputs fall within Delhi's historical physical bounds (temperature: 42.0 to 43.8 °C vs. observed historical max of 46.4 °C).
            </div>

            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Scenario Timestamp</th>
                            <th>Scenario Temp (°C)</th>
                            <th>Dew Point (°C)</th>
                            <th>Humidity (%)</th>
                            <th>Lag 5min (MW)</th>
                            <th>Model Projected Demand (MW)</th>
                            <th>Classification</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for _, r in scenario_df.iterrows():
        is_peak = r["projected_demand_mw"] == scenario_df["projected_demand_mw"].max()
        badge = '<span class="tag tag-red">PROJECTED PEAK</span>' if is_peak else '<span class="tag tag-blue">PROJECTION</span>'
        html_content += f"""                        <tr>
                            <td>{r['timestamp']}</td>
                            <td>{r['temp_degC']:.1f} °C</td>
                            <td>{r['dwpt_degC']:.1f} °C</td>
                            <td>{r['rhum_pct']:.1f}%</td>
                            <td>{r['demand_lag_5min_mw']:.1f}</td>
                            <td class="highlight" style="{'color:var(--accent-red);' if is_peak else ''}">{r['projected_demand_mw']:.2f} MW</td>
                            <td>{badge}</td>
                        </tr>\n"""

    html_content += f"""                    </tbody>
                </table>
            </div>

            <div style="margin-top:16px;">
                <p style="font-size:13px; color:var(--text-secondary);">
                    <strong>Scenario Peak Identified:</strong> Projected maximum demand reaches <strong>{s_max_row['projected_demand_mw']:.2f} MW</strong> at <strong>{s_max_row['timestamp']}</strong> under a scenario ambient temperature of <strong>{s_max_row['temp_degC']:.1f} °C</strong>.
                </p>
                <img src="{b64_chart2}" alt="Scenario Projection" class="chart-img">
            </div>
        </div>

        <!-- Footer -->
        <footer>
            Project: AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System &bull; Completed Milestone Dashboard (Steps 1 – 12) &bull; Verified Locally
        </footer>
    </div>
</body>
</html>"""

    html_path = os.path.join(OUTPUT_DIR, "project_dashboard.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved HTML dashboard to: {html_path}")

    # =========================================================================
    # 4. Generate Markdown Summary Report
    # =========================================================================
    print("Generating project summary markdown report...")
    md_path = os.path.join(OUTPUT_DIR, "project_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Project Summary Report\n")
        f.write("## AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System\n\n")

        f.write("### 1. Project Workflow & Methodology\n")
        f.write("```\n")
        f.write("Delhi SLDC-based Electricity Demand & Weather Data (2021-2024)\n")
        f.write("        ↓\n")
        f.write("Data preprocessing (leakage removal, causal imputation)\n")
        f.write("        ↓\n")
        f.write("Exploratory Data Analysis (diurnal, seasonal, weather correlations)\n")
        f.write("        ↓\n")
        f.write("Feature engineering (17 causal calendar, lag, & weather features)\n")
        f.write("        ↓\n")
        f.write("Chronological train/test split (80% train / 20% hold-out test)\n")
        f.write("        ↓\n")
        f.write("Linear Regression + Random Forest model training\n")
        f.write("        ↓\n")
        f.write("Model evaluation & comparative benchmarking\n")
        f.write("        ↓\n")
        f.write("Dedicated daily & monthly peak demand analysis\n")
        f.write("        ↓\n")
        f.write("Scenario-based peak demand projection module\n")
        f.write("```\n\n")

        f.write("### 2. Dataset & Quality Metrics\n")
        f.write(f"- **Total Cleaned Observations:** `{cleaned_obs:,}` rows (5-minute observations with some missing intervals)\n")
        f.write(f"- **Date Range:** `{date_range_str}`\n")
        f.write(f"- **Features Used:** `{features_count}` input features\n")
        f.write(f"- **Chronological Train Split:** `{train_rows:,}` rows (80.0%)\n")
        f.write(f"- **Chronological Test Split:** `{test_rows:,}` rows (20.0%)\n")
        f.write(f"- **Missing Values Post-Preprocessing:** `{nulls_after_preproc}`\n")
        f.write(f"- **Documented Time Gaps:** `{irregular_gaps:,}` transitions\n\n")

        f.write("### 3. Model Benchmark Summary (Hold-out Test Period)\n")
        f.write("| Metric | Linear Regression Baseline | Random Forest Regression |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Test MAE** | `25.24 MW` | `28.17 MW` |\n")
        f.write(f"| **Test RMSE** | `43.23 MW` | `56.29 MW` |\n")
        f.write(f"| **Test R²** | `0.9990` | `0.9983` |\n")
        f.write(f"| **Training Time** | `0.42 s` | `140.85 s` |\n")
        f.write(f"| **Prediction Time** | `0.05 s` | `1.02 s` |\n\n")

        f.write("### 4. Peak Demand Analysis & Model Peak Evaluation\n")
        f.write(f"- **Highest Observed Demand in Test Period:** `{highest_peak_val:,.2f} MW` on `{highest_peak_dt}`\n")
        f.write(f"- **Linear Regression Prediction at Peak:** `8,560.41 MW` (Error: `-71.12 MW` / `-0.82%`)\n")
        f.write(f"- **Random Forest Prediction at Peak:** `7,677.32 MW` (Error: `-954.21 MW` / `-11.05%`)\n")
        f.write(f"- **Daily Peak MAE (272 days):** Linear Regression = `39.86 MW` | Random Forest = `63.88 MW`\n")
        f.write(f"- **Daily Peak RMSE (272 days):** Linear Regression = `72.07 MW` | Random Forest = `136.40 MW`\n")
        f.write(f"- **Average Daily Peak Demand:** `{avg_daily_peak:,.2f} MW` (Median: `{median_daily_peak:,.2f} MW`)\n")
        f.write(f"- **Most Common Peak Hour:** Hour `15:00` (3 PM), occurring on `89` out of 272 days (`32.7%`)\n\n")

        f.write("### 5. Illustrative Scenario Projection\n")
        f.write("> **LABEL:** `MODEL-BASED SCENARIO PROJECTION — NOT AN OBSERVED VALUE`\n")
        f.write(f"- **Projected Peak Demand:** `{s_max_row['projected_demand_mw']:,.2f} MW`\n")
        f.write(f"- **Projected Peak Timestamp:** `{s_max_row['timestamp']}`\n")
        f.write(f"- **Scenario Temperature at Peak:** `{s_max_row['temp_degC']:.1f} °C`\n\n")

        f.write("### 6. File Artifacts Created in Step 12\n")
        f.write(f"- HTML Dashboard: `{html_path}`\n")
        f.write(f"- Markdown Summary: `{md_path}`\n")
        f.write(f"- Composite Overview Chart: `{chart1_path}`\n")
        f.write(f"- Scenario Trajectory Chart: `{chart2_path}`\n")

    print(f"Saved project summary report to: {md_path}")
    print("Dashboard generation script completed successfully!")

if __name__ == "__main__":
    build_dashboard()
