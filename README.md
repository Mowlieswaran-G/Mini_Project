# ⚡ Power Predict AI — Delhi Electricity Demand & Peak Projection System

> **AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System**  
> An end-to-end, zero-leakage machine learning & deep learning system engineered to project granular (5-minute) power dispatch loads, forecast extreme heatwave grid stress, and evaluate operational capacity for the National Capital Territory of Delhi.

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CPU%2FCUDA)-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## 📋 Table of Contents
1. [Executive Summary & Problem Context](#1-executive-summary--problem-context)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Dataset & Zero-Leakage Engineering](#3-dataset--zero-leakage-engineering)
4. [Models in This Project](#4-models-in-this-project)
   - [Linear Regression Baseline](#41-baseline-linear-regression)
   - [Random Forest Regressor](#42-random-forest-regressor)
   - [LSTM Deep Recurrent Neural Network](#43-long-short-term-memory-lstm-neural-network)
5. [Comparative Performance Benchmark](#5-comparative-performance-benchmark)
6. [Historic Delhi Peak Case Study (8,631.53 MW)](#6-historic-delhi-peak-case-study)
7. [Demand Regime & Extreme Value Analysis](#7-demand-regime--extreme-value-analysis)
8. [Interactive Web Application (Streamlit)](#8-interactive-web-application-streamlit)
9. [Documentation & Audit Reports](#9-documentation--audit-reports)
10. [Repository Directory Structure](#10-repository-directory-structure)
11. [Quick Start & Reproduction Guide](#11-quick-start--reproduction-guide)
12. [Automated Test Suite](#12-automated-test-suite)
13. [Documented Limitations](#13-documented-limitations)
14. [Docker Deployment](#14-docker-deployment)

---

## 1. Executive Summary & Problem Context

The National Capital Territory of Delhi is one of the world's most populous metropolitan regions (>33 million inhabitants), characterized by extreme continental climatic swings:
- **Winter (Dec–Jan):** Dense fog spells and temperatures down to ~4 °C trigger morning domestic space-heating and water geyser surges.
- **Summer & Monsoon (May–Jul):** Severe heatwaves exceeding **45 °C** and high relative humidity drive massive air-conditioning (HVAC) loads, representing over **50% of the entire grid demand**.

On **19 June 2024 at 15:05 IST**, the test period recorded a historical peak demand of **8,631.53 MW**.

### Why Standard Forecasting Breaks at Extreme Peaks
Traditional non-linear machine learning models (such as tree-based ensembles) partition historical training data into bounded leaves. When historical unprecedented records occur, decision trees cannot predict higher than the training set ceiling (~7,746 MW). This project benchmarks **Linear Regression**, **Random Forest**, and a **PyTorch Long Short-Term Memory (LSTM) Recurrent Neural Network** across 4 years (2021–2024) of 5-minute dispatch telemetry to deliver accurate operational foresight and peak risk assessment.

---

## 2. End-to-End System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DELHI STATE SLDC TELEMETRY & WEATHER DATA                    │
│                 (393,440 Raw 5-Min Observations: 2021-01 to 2024-12)             │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         STEP 1: PREPROCESSING & CLEANING                        │
│  - Strict Causal Forward-Fill for Missing Wind (wdir) (540 values)              │
│  - Zero Synthetic Timestamp Interpolation                                       │
│  - Elimination of Contemporaneous Target Leakage (Dropped leaky moving_avg_3)   │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      STEP 2: CAUSAL FEATURE ENGINEERING                         │
│  17 Causal Features:                                                            │
│   • Calendar: year, month, day, hour, minute, day_of_week, is_weekend           │
│   • Autoregressive Lags: demand_lag_5min, lag_1hour, lag_24hour, lag_7day      │
│   • Atmospheric: temp (°C), dwpt (°C), rhum (%), wdir (°), wspd (km/h), pres   │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     STEP 3: CHRONOLOGICAL 80/20 SPLIT                           │
│   • Training Set: 283,298 Rows (2021-01-08 01:00 to 2024-03-05 13:20)           │
│   • Hold-Out Test Set: 70,825 Rows (2024-03-05 13:25 to 2024-12-11 22:25)       │
└───────────────┬────────────────────────┼────────────────────────┬───────────────┘
                │                        │                        │
                ▼                        ▼                        ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│   LINEAR REGRESSION     │ │   RANDOM FOREST         │ │       PYTORCH LSTM      │
│   (OLS Baseline)        │ │   (100 Trees Ensemble)  │ │   (2-Layer Deep RNN)    │
│  • Extrapolates records │ │  • Low normal-load MAE  │ │  • Sequence memory      │
│  • Sub-second fit       │ │  • Tree leaf ceiling    │ │  • Non-linear + Temporal│
└───────────────┬─────────┘ └────────────┬────────────┘ └────────────┬────────────┘
                │                        │                           │
                └────────────────────────┼───────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         STEP 4: EVALUATION & ANALYSIS                           │
│   • Hold-Out Test Metrics: MAE, RMSE, R²                                        │
│   • Regime Error Breakdown (Normal <=75%, High, Extreme Top 5%, Super Top 1%)   │
│   • Peak Hour and Daily Peak Demand Distribution (272 Daily Test Peaks)         │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       STEP 5: INTERACTIVE DEPLOYMENT                            │
│   • Streamlit Web Application (app/app.py) with 8 Modular Operational Views     │
│   • Standalone Multi-Model HTML Dashboard (outputs/project_dashboard/)          │
│   • Autonomous Multi-Step Forward Predictor (1h, 24h, 7d Horizons)              │
│   • Docker Orchestration with Live Model Volume Mounts & Health Checks          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset & Zero-Leakage Engineering

### 3.1 Raw Data Characteristics
- **Source:** 5-minute dispatch intervals of Delhi State Load Despatch Centre (SLDC) combined with weather telemetry.
- **Span:** 1 January 2021 to 12 December 2024 (4 full calendar years).
- **Observations:** 393,440 rows.
- **Historical Demand Envelope:** Minimum `1,302.08 MW` | Maximum `8,631.53 MW`.

### 3.2 Data Leakage Prevention Principles
1. **Target Leakage Elimination:** The raw dataset included an unlagged `moving_avg_3` column that looked ahead into contemporary and future dispatches. This was eliminated entirely.
2. **Causal Autoregressive Lagging:** All lag features are calculated strictly via exact retrospective timestamp subtraction:
   - $t - 5\text{ minutes}$: Immediate short-term grid momentum.
   - $t - 60\text{ minutes}$: 1-hour intraday inertia.
   - $t - 1440\text{ minutes}$ (24 hours): Daily diurnal cycle matching.
   - $t - 10080\text{ minutes}$ (7 days): Weekly seasonal matching (weekday vs weekend).
3. **Strict Chronological Splitting:**
   - Shuffling was disabled. The test partition occurs strictly after the training partition.
   - First 7 days (39,317 rows) warm up the 7-day autoregressive lag filter.
   - **Training Set (80%):** 283,298 rows (`2021-01-08 01:00:00` to `2024-03-05 13:20:00`).
   - **Hold-Out Test Set (20%):** 70,825 rows (`2024-03-05 13:25:00` to `2024-12-11 22:25:00`).

---

## 4. Models in This Project

This project implements and evaluates three distinct model architectures across identical feature sets:

### 4.1 Baseline Linear Regression
- **Implementation:** [`train_linear_regression.py`](train_linear_regression.py)
- **Artifact:** [`linear_regression_delhi_powerdemand.pkl`](linear_regression_delhi_powerdemand.pkl) (~1.25 KB)
- **Mathematical Form:** Ordinary Least Squares:
  $$\hat{y} = \beta_0 + \sum_{i=1}^{17} \beta_i X_i$$
- **Strengths:** Trains in ~0.42 seconds with linear slope extrapolation at extreme peaks.
- **Metrics:** Test MAE: `25.24 MW` | Test RMSE: `43.23 MW` | Test $R^2$: `0.9990`.

### 4.2 Random Forest Regressor
- **Implementation:** [`train_random_forest.py`](train_random_forest.py)
- **Artifact:** [`random_forest_delhi_powerdemand.pkl`](random_forest_delhi_powerdemand.pkl) (~740.8 MB uncompressed)
- **Configuration:** 100 estimators, `max_depth=20`, `min_samples_leaf=2`, `n_jobs=-1`.
- **Strengths:** Captures non-linear meteorological interactions across normal load cycles ($\le 5,589$ MW).
- **Limitation (Tree Leaf Bounding):** Decision trees cannot predict values higher than the maximum target observation in their terminal training leaves (~7,746 MW).
- **Metrics:** Test MAE: `28.17 MW` | Test RMSE: `56.29 MW` | Test $R^2$: `0.9983`.

### 4.3 Long Short-Term Memory (LSTM) Neural Network
- **Implementation:** [`train_lstm.py`](train_lstm.py), [`lstm_model.py`](lstm_model.py)
- **Artifacts:**
  - PyTorch Checkpoint: [`lstm_delhi_powerdemand.pt`](lstm_delhi_powerdemand.pt) (~232 KB)
  - Inference Wrapper: [`lstm_delhi_powerdemand.pkl`](lstm_delhi_powerdemand.pkl) (~233 KB)
- **Architecture:**
  - Input: 17 scaled continuous features.
  - Multi-Timestep Sequence Window: 6 steps (30-minute lookback).
  - 2 Stacked LSTM Layers (hidden dimension: 64, dropout: 0.1).
  - Fully Connected Head: `Linear(64 -> 32) -> ReLU -> Linear(32 -> 1)`.
- **Strengths:** Models continuous temporal sequences; cuts Random Forest's peak under-prediction error by more than 50%.
- **Metrics:** Test MAE: `27.63 MW` | Test RMSE: `47.77 MW` | Test $R^2$: `0.9987`.

---

## 5. Comparative Performance Benchmark

All metrics evaluated strictly on the **70,825 hold-out test observations** (March to December 2024):

| Metric | Unit | Linear Regression | Random Forest | LSTM Deep Network | Best Performer |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test MAE** | MW | **25.24** | 28.17 | 27.63 | **Linear Regression** |
| **Test RMSE** | MW | **43.23** | 56.29 | 47.77 | **Linear Regression** |
| **Test $R^2$ Score** | — | **0.9990** | 0.9983 | 0.9987 | **Linear Regression** |
| **Training Time** | Seconds | **0.42 s** | 140.85 s | 133.33 s | **Linear Regression (~300x faster)** |
| **Inference Time (70.8k)** | Seconds | **0.05 s** | 1.02 s | 1.71 s | **Linear Regression (~20x faster)** |
| **Peak Error at 8,631.5 MW** | MW | **-71.12** (-0.82%) | -954.21 (-11.05%) | -451.86 (-5.23%) | **Linear Regression** |
| **Daily Peak MAE (272 Days)** | MW | **39.86** | 63.88 | 61.62 | **Linear Regression** |

---

## 6. Historic Delhi Peak Case Study

### The Event: 19 June 2024 at 15:05 IST
- **Ground Truth Demand:** **`8,631.53 MW`**
- **Atmospheric Conditions:** Ambient Temperature: **44.5 °C** | Relative Humidity: **32%** | Apparent Heat Index: **~54 °C**.

### Model Predictions:
```
Ground Truth Peak:   ████████████████████████████████████████ 8,631.53 MW
Linear Regression:   █████████████████████████████████████▌   8,560.41 MW  (Error: -71.12 MW / -0.82%)
LSTM Deep Network:   ████████████████████████████████████     8,179.67 MW  (Error: -451.86 MW / -5.23%)
Random Forest:       ██████████████████████████████           7,677.32 MW  (Error: -954.21 MW / -11.05%)
```

---

## 7. Demand Regime & Extreme Value Analysis

Error distribution evaluated across quantile-based grid operating tiers:

| Operating Regime | Demand Boundary | Data Share | LR MAE | RF MAE | LSTM MAE | Regime Characteristic |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Normal Demand** | $\le 5,589$ MW | 75% | 23.97 MW | **23.65 MW** | 23.99 MW | RF slightly superior on standard loads |
| **High Demand** | $5,589 - 6,906$ MW | 20% | **28.67 MW** | 31.35 MW | 31.78 MW | Transition regime |
| **Extreme Peak** | $> 6,906$ MW | Top 5% | **30.58 MW** | 83.13 MW | 65.65 MW | LR substantially superior |
| **Super Extreme** | $> 7,655$ MW | Top 1% | **27.78 MW** | 243.52 MW | 135.05 MW | RF error surges 9x; LSTM dampens errors |

---

## 8. Interactive Web Application (Streamlit)

The web dashboard is located in [`app/app.py`](app/app.py) and is powered by an enterprise-grade UI design system featuring **Plus Jakarta Sans** and **JetBrains Mono** typography, pastel metric card accents, Plotly spline curves, and a floating sidebar navigation with live AI model status badges. It is organized into **8 modular operational views**:

1. **📊 Dashboard (Executive Operations & Grid Stability):**
   - **Sudden Interruption Monitoring:** High-resolution telemetry tracking sudden load dips and rapid ramp-rate excursions in real time.
   - **Financial Collection & Due Margin:** Summer peak collection vs. spinning reserve margin monitor with critical threshold alerts.
   - **Quarterly Revenue Trajectory:** Dynamic pill cards tracking DISCOM monthly revenue performance and seasonal demand spikes.
   - **Generation vs. Consumption:** Dual-spline waveform overlay comparing scheduled generation dispatch against actual consumer load with historical peak callouts.
   - **Load Shedding Prevention:** Automated tracker computing total load shedding hours avoided across Delhi distribution zones.
   - **New Consumer Growth:** Sector-wise active connection breakdown across Commercial High-Rise, Industrial Parks, EV Fast Charging, Residential Domestic, and Institutional sectors.

2. **📈 Performance (Model Benchmark & Regime Analytics):**
   - **Pastel Metric Grid:** 6 live grid performance tiles (Total Generation GWh, 99.8% System Uptime, Outage / Fall Time, Active Substations, Resolved Grid Alerts, Online Telemetry Nodes).
   - **Generation Profile & Substation Availability:** Monthly generation spline curves and availability uptime bar charts.
   - **Outage Trend & Consumer Trajectory:** Unplanned outage reduction curves alongside annual connection expansions.
   - **Demand Regime Share:** Interactive donut distribution across Normal ($\le 5,589$ MW), High ($5,589 - 6,906$ MW), and Extreme Peak ($> 6,906$ MW) operating tiers.
   - **Model Benchmark Suite:** Multi-model comparative metrics ($MAE$, $RMSE$, $R^2$) and residual error analysis.

3. **🗺️ Distribution (Delhi NCT 5 DISCOM Zones):**
   - **Zonal Jurisdiction Coverage:** Dedicated operational monitoring for Delhi's 5 distribution utilities:
     - **BRPL** (BSES Rajdhani Power Limited — South & West Delhi)
     - **BYPL** (BSES Yamuna Power Limited — Central & East Delhi)
     - **TPDDL** (Tata Power Delhi Distribution Limited — North & North-West Delhi)
     - **NDMC** (New Delhi Municipal Council — Lutyens' VIP Delhi)
     - **MES** (Military Engineer Services — Delhi Cantonment Area)
   - **Financial & Operational Pills:** Bill collections, pending due margin, electricity consumed (kWh), and active feeder counts.
   - **Consumption Details Waveform:** Multi-month demand curves with automated peak demand callouts.
   - **Substation Telemetry Explorer:** Filterable feeder table reporting voltage class (33 kV / 66 kV / 220 kV), transformer loading percentages, and active health badges.

4. **👥 Consumer & Telemetry (SLDC Telemetry Data Explorer):**
   - **Full Telemetry Span:** Interactive query engine over 393,440 historical 5-minute SLDC dispatch records (2021–2024) and 70,825 hold-out evaluation records.
   - **Sector Consumption Profiles:** Demand breakdown by Domestic, Commercial, Industrial, and EV mobility categories.
   - **Hourly Dispatch Profiling:** Intraday diurnal demand curves with interactive peak timestamp indicators.
   - **Live Telemetry Table:** Searchable, paginated record browser with zero-leakage data validation badges.

5. **🔮 Forecast Studio (Autonomous Multi-Step Forward AI Engine):**
   - **Multi-Horizon Recursive Forecasting:**
     - **Next 1 Hour:** 12 sequential 5-minute dispatch steps
     - **Next 24 Hours:** 288 steps intraday dispatch projection
     - **Next 7 Days:** 2,016 steps weekly cyclical forecast
   - **Dynamic Engine Selector:** Seamlessly switch between **Linear Regression**, **100-Tree Random Forest**, and **PyTorch LSTM**.
   - **Forecast Summary KPIs:** Real-time computation of Projected Peak Demand (MW), Projected Average Load (MW), Volatility Spread, and Peak Timestamp.
   - **Interactive Forecast Visualizer:** High-resolution Plotly line chart with past context, forecasted trajectory, and shaded confidence bounds.
   - **One-Click Export:** Download full multi-step projected dispatch series as CSV.

6. **🌦️ Weather Scenarios (Atmospheric & Heat-Index Simulator):**
   - **Meteorological Stress Sliders:** Interactive controls for Ambient Temperature (10 °C to 50 °C), Dew Point (0 °C to 35 °C), Relative Humidity (10% to 100%), Wind Speed, Wind Direction, and Barometric Pressure.
   - **Apparent Heat Index Engine:** Real-time formulaic computation of apparent heat index with National Weather Service risk categories (*Caution*, *Extreme Caution*, *Danger*, *Extreme Danger*).
   - **What-If Grid Sensitivity Analysis:** Live simulation of extreme summer heatwaves and monsoon mugginess on peak dispatch demands.

7. **🏔️ Peak Analysis (Historical Peak Case Studies & Benchmarks):**
   - **Historic Peak Case Study:** Micro-level breakdown of Delhi's unprecedented all-time record of **8,631.53 MW** (19 June 2024 at 15:05 IST).
   - **Model Error Dissection:** Side-by-side error comparison (LR: $-71.12$ MW / $-0.82\%$, LSTM: $-451.86$ MW / $-5.23\%$, RF: $-954.21$ MW / $-11.05\%$).
   - **272 Monitored Daily Peaks:** Daily peak demand scatter, diurnal peak hour histogram, and monthly maximum demand progression.
   - **Residual Visualizations:** Multi-model error distributions across extreme demand quantiles.

8. **📑 Reports & Audit (Technical Documentation & Verified Artifacts):**
   - **Instant Downloads:** One-click direct downloads for `model_comparison_delhi_powerdemand.csv`, `daily_peak_analysis.csv`, `peak_model_performance.csv`, and the standalone offline `project_dashboard.html`.
   - **Documentation Links:** Direct access to technical methodology, verification reports, and testing audits.
   - **Data Quality Badges:** Verified zero-null, zero synthetic interpolation, and strict chronological split status.

---

## 9. Documentation & Audit Reports

- [`docs/FINAL_VERIFICATION_REPORT.md`](docs/FINAL_VERIFICATION_REPORT.md): Comprehensive end-to-end audit confirming model persistence, prediction alignments, feature pipeline integrity, and Docker deployment readiness.
- [`docs/UI_FUNCTIONALITY_AUDIT.md`](docs/UI_FUNCTIONALITY_AUDIT.md): In-depth technical audit of all 8 application views, layout isolation, responsive components, CSS design tokens, and state preservation.
- [`docs/UI_FIX_VERIFICATION_REPORT.md`](docs/UI_FIX_VERIFICATION_REPORT.md): Verification report covering UI enhancements, LSTM causal rolling buffers, scenario calendar derivations, and automated regression tests.
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): Mathematical derivations, causal feature engineering specifications (17 features), and model training protocols.
- [`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md): In-depth benchmark tables, demand regime breakdowns, and extreme peak error analysis across all 70,825 test steps.
- [`docs/DATA_QUALITY_REPORT.md`](docs/DATA_QUALITY_REPORT.md): Data pipeline audit confirming zero target leakage, absence of synthetic timestamps, and causal forward-fill protocols.
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md): Transparent documentation of real-world operational boundaries, multi-step error accumulation, and weather persistence assumptions.
- [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md): Automated test execution report detailing 28 verified PASS outcomes across 8 test suites.

---

## 10. Repository Directory Structure

```
Mini_Project/
├── app/                                       # Production Streamlit Web Application
│   ├── .streamlit/config.toml                 # Custom UI Theme configuration (Plus Jakarta Sans)
│   ├── app.py                                 # Main Streamlit Dashboard (8 Modular Views)
│   ├── lstm_model.py                          # Application copy of LSTM model wrapper
│   └── utils/
│       ├── data.py                            # Caching, model loaders & telemetry helpers
│       └── features.py                        # Causal feature generator, physical bounds & validation
├── docs/                                      # Comprehensive Technical Documentation
│   ├── FINAL_VERIFICATION_REPORT.md           # End-to-end system verification & deployment sign-off
│   ├── UI_FUNCTIONALITY_AUDIT.md              # Technical audit across all 8 views and UI components
│   ├── UI_FIX_VERIFICATION_REPORT.md          # Verification report for UI & functional fixes
│   ├── METHODOLOGY.md                         # Pipeline and feature engineering details
│   ├── MODEL_EVALUATION.md                    # Detailed model comparisons & regime analytics
│   ├── LIMITATIONS.md                         # Real-world limitations & error accumulation
│   ├── DATA_QUALITY_REPORT.md                 # Data quality and zero-leakage audit
│   └── TEST_REPORT.md                         # Automated unit tests execution report (28 tests)
├── outputs/                                   # Audited Evaluation Artifacts
│   ├── linear_regression/                     # LR plots and metrics.json
│   ├── random_forest/                         # RF plots, feature importances and metrics.json
│   ├── lstm/                                  # LSTM convergence, zoomed peak & metrics.json
│   ├── model_comparison/                      # Benchmark plots, reports and comparison summary
│   ├── peak_analysis/                         # 7 Analytical peak charts and performance CSVs
│   │   ├── 1_daily_actual_peak_demand_over_time.png
│   │   ├── 2_daily_actual_vs_lr_predicted_peak.png
│   │   ├── 3_daily_actual_vs_rf_predicted_peak.png
│   │   ├── 4_daily_actual_vs_lstm_predicted_peak.png
│   │   ├── 5_monthly_maximum_demand.png
│   │   ├── 6_distribution_of_daily_peak_hours.png
│   │   └── 7_peak_prediction_error_comparison.png
│   ├── peak_projection/                       # Scenario simulation reports and charts
│   └── project_dashboard/                     # HTML standalone dashboard & overview charts
├── tests/                                     # Automated Unit & Integration Tests (28 Tests)
│   ├── test_data_validation.py                # Dataset checks and chronological split boundaries
│   ├── test_feature_engineering.py            # Feature schema and exact lag mapping
│   ├── test_models.py                         # Model loading and inference verification
│   ├── test_predictions_alignment.py         # Prediction length and metric calculation
│   ├── test_peak_analysis.py                  # Peak extraction and distinction checks
│   ├── test_weather_and_forecasting.py        # Weather bounds validation and API safety
│   ├── test_dashboard_and_reports.py          # Dashboard integrity and import checks
│   └── test_ui_fixes.py                       # UI fix verification (8 issues, 9 tests)
├── cleaned_delhi_powerdemand_5min_2021_to_2024.csv    # Cleaned, zero-null telemetry (393,440 rows)
├── feature_engineered_delhi_powerdemand_5min_2021_to_2024.csv # 17 Causal features (393,440 rows)
├── train_delhi_powerdemand.csv                # Chronological 80% train partition (283,298 rows)
├── test_delhi_powerdemand.csv                 # Chronological 20% test partition (70,825 rows)
├── linear_regression_delhi_powerdemand.pkl    # Serialized Linear Regression binary (1.25 KB)
├── random_forest_delhi_powerdemand.pkl        # Serialized Random Forest binary (~740.8 MB)
├── lstm_delhi_powerdemand.pt                  # Serialized PyTorch LSTM checkpoint (~232 KB)
├── lstm_delhi_powerdemand.pkl                 # Inference-ready LSTM wrapper bundle (~233 KB)
├── linear_regression_test_predictions.csv     # 70,825 Aligned hold-out predictions
├── random_forest_test_predictions.csv         # 70,825 Aligned hold-out predictions
├── lstm_test_predictions.csv                  # 70,825 Aligned hold-out predictions
├── model_comparison_delhi_powerdemand.csv     # Side-by-side benchmark CSV
├── preprocess_data.py                         # Step 1: Preprocessing & leakage removal
├── feature_engineering.py                     # Step 2: 17 Causal feature calculation
├── prepare_modeling_dataset.py                # Step 3: Chronological train/test split
├── eda_analysis.py                            # Step 4: Exploratory data analysis
├── train_linear_regression.py                 # Step 5: Train & evaluate Linear Regression
├── train_random_forest.py                     # Step 6: Train & evaluate Random Forest
├── train_lstm.py                              # Step 7: Train & evaluate PyTorch LSTM
├── lstm_model.py                              # Core DemandLSTM neural network architecture
├── compare_models.py                          # Step 8: Comprehensive model benchmarking
├── run_peak_analysis.py                       # Step 9: Multi-model peak analytics
├── peak_demand_projection.py                  # Step 10: Scenario projection engine
├── project_results_dashboard.py               # Step 11: Export standalone HTML dashboard
├── data_quality_report.py                     # Automated data quality audit generator
├── Dockerfile                                 # Production container definition
├── docker-compose.yml                         # Container orchestration configuration (Volume mounts)
└── requirements.txt                           # Python project dependencies
```

---

## 11. Quick Start & Reproduction Guide

### Prerequisites
- Python 3.10 to 3.13 (64-bit)
- 4 GB+ RAM (8 GB recommended)

### 1. Set Up Environment
```bash
git clone <repository-url>
cd Mini_Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note for PyTorch (CPU-only):*
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Launch the Interactive Streamlit Web Application
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

### 4. Regenerate the Standalone Offline HTML Dashboard
```bash
python project_results_dashboard.py
```
Outputs `outputs/project_dashboard/project_dashboard.html`, which can be opened directly in any browser without a web server.

---

## 12. Automated Test Suite

Run the full automated test suite covering data validation, feature generation, model loading, prediction alignment, peak analysis, weather forecasting bounds, dashboard generation, and UI fixes:

```bash
python -m unittest discover tests
```

Expected outcome:
```text
Ran 28 tests in ~6.0s
OK (All 28 tests PASS)
```

### Test Suite Breakdown (8 Test Modules, 28 Tests)

| Test Module | Coverage & Verification Focus | Test Count |
|:---|:---|:---:|
| [`tests/test_data_validation.py`](tests/test_data_validation.py) | Raw dataset integrity, null checks, row counts, and chronological split boundaries (80/20) | 4 |
| [`tests/test_feature_engineering.py`](tests/test_feature_engineering.py) | 17 causal feature schema, exact retrospective lag subtractions (5m, 1h, 24h, 7d), and zero lookahead | 5 |
| [`tests/test_models.py`](tests/test_models.py) | Serialization checks, PyTorch/scikit-learn model loading, and forward pass sanity | 3 |
| [`tests/test_predictions_alignment.py`](tests/test_predictions_alignment.py) | 70,825 test prediction alignment, metrics consistency, and column schemas | 3 |
| [`tests/test_peak_analysis.py`](tests/test_peak_analysis.py) | Daily peak extraction, historic 8,631.53 MW record detection, and peak error metrics | 4 |
| [`tests/test_weather_and_forecasting.py`](tests/test_weather_and_forecasting.py) | Physical meteorological bounds validation, heat index calculation, and scenario presets | 4 |
| [`tests/test_dashboard_and_reports.py`](tests/test_dashboard_and_reports.py) | Standalone HTML dashboard generation, file size (>500 KB), and report consistency | 3 |
| [`tests/test_ui_fixes.py`](tests/test_ui_fixes.py) | LSTM rolling sequence shapes `(1, 6, 17)`, causal roll shifts, scenario date/time derivations, and UI stability | 2 |

---

## 13. Documented Limitations

1. **Absence of Live SCADA Connection:** Models are trained on historical 2021–2024 telemetry. Real-time dispatches are simulated via recent telemetry or user inputs.
2. **Missing Sampling Intervals:** Real grid telemetry contains 1,150 irregular gap transitions (> 5 minutes) and 21,569 missing slots across 4 years. Missing slots remain `NaN` without synthetic interpolation.
3. **Recursive Forecasting Error Accumulation:** In multi-step recursive forecasting (24-hour and 7-day horizons), lag errors accumulate across forward steps.
4. **Weather Dependency:** Projections assume persistence of the last observed weather conditions unless updated with live meteorological NWP data.
5. **Decision Tree Bounding:** Random Forest cannot extrapolate past its training leaf ceiling (~7,746 MW) during unprecedented heatwave records.

---

## 14. Docker Deployment

The application includes a production-ready Docker configuration featuring **volume-mounted model artifacts**, hot-reloadable application code, and integrated health checks.

### 1. Launch with Docker Compose (Recommended)
```bash
docker compose up -d --build
```

### 2. Verify Container Health
```bash
docker compose ps
```
The container includes a built-in healthcheck probing `http://localhost:8501/_stcore/health` every 30 seconds.

### 3. View Real-Time Logs
```bash
docker compose logs -f
```

### 4. Stop the Container
```bash
docker compose down
```

### Direct Docker CLI Alternative
```bash
docker build -t delhi-power-demand:latest .
docker run -d \
  -p 8501:8501 \
  -v "${PWD}/app:/app/app" \
  -v "${PWD}/random_forest_delhi_powerdemand.pkl:/app/random_forest_delhi_powerdemand.pkl" \
  --name power-predict-app \
  delhi-power-demand:latest
```

Access the live dashboard at **`http://localhost:8501`**.

> [!NOTE]
> **Optimized Image Architecture:** The 740 MB Random Forest pickle model is volume-mounted directly from the host (`./random_forest_delhi_powerdemand.pkl:/app/random_forest_delhi_powerdemand.pkl`) rather than baked into the Docker image layer. This keeps the image lightweight, avoids multi-gigabyte build contexts, and enables instant zero-rebuild code updates via `./app:/app/app`.

