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
│   • Streamlit Web Application (app/app.py) with 10 Structured Sections          │
│   • Standalone Multi-Model HTML Dashboard (outputs/project_dashboard/)          │
│   • Automated Autoregressive Lag Estimator & Heat Index Calculator              │
│   • Docker Containerization & Multi-Model Inference API                         │
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

The web dashboard is located in [`app/app.py`](app/app.py) and is organized into 10 structured operational sections:
1. **🏠 Project Overview:** System architecture, telemetry summary, and model characteristics.
2. **🧹 Dataset & Data Quality:** Preprocessing audit, zero-leakage verification, and data table explorer.
3. **📊 Exploratory Data Analysis:** Diurnal curves, weekly profiles, seasonal variation, and temperature scatter.
4. **🤖 Model Comparison:** Side-by-side benchmark metrics and residual error histograms.
5. **🔮 Demand Forecasting:** Recursive forward projections (Next 1 Hour, Next 24 Hours, Next 7 Days) with CSV export.
6. **🏔️ Peak-Demand Analysis:** Test period peak events, actual peak timestamps vs. model maximum values, and diurnal peak frequency.
7. **🌦️ Weather Scenario Analysis:** Interactive What-If simulation with physical boundary validation and optional Open-Meteo live weather query.
8. **🚨 Forecast Alerts & Regimes:** Operational tier classification with user-configurable simulation alert thresholds.
9. **🎯 Model Evaluation & Residuals:** Hold-out test residual plots and actual vs. predicted overlays.
10. **📥 Downloadable Reports:** Download CSV reports for model comparisons, daily peaks, and project summaries.

---

## 9. Documentation & Audit Reports

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): Comprehensive documentation of preprocessing, feature engineering, and model designs.
- [`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md): In-depth benchmark tables, regime breakdowns, and peak error analysis.
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md): Transparent documentation of real-world boundaries, recursive lag accumulation, and weather dependencies.
- [`docs/DATA_QUALITY_REPORT.md`](docs/DATA_QUALITY_REPORT.md): Data pipeline audit confirming zero leakage and chronological discipline.
- [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md): Automated test execution report detailing 19 verified PASS outcomes.

---

## 10. Repository Directory Structure

```
Mini_Project/
├── app/                                       # Streamlit Web Application
│   ├── .streamlit/config.toml                 # Custom UI Theme configuration
│   ├── app.py                                 # Main Streamlit Dashboard (10 Sections)
│   ├── lstm_model.py                          # Application copy of LSTM model wrapper
│   └── utils/
│       ├── data.py                            # Caching, model loaders & telemetry helpers
│       └── features.py                        # Causal feature generator, physical bounds & validation
├── docs/                                      # Comprehensive Technical Documentation
│   ├── METHODOLOGY.md                         # Pipeline and feature engineering details
│   ├── MODEL_EVALUATION.md                    # Detailed model comparisons & regime analytics
│   ├── LIMITATIONS.md                         # Real-world limitations & error accumulation
│   ├── DATA_QUALITY_REPORT.md                 # Data quality and zero-leakage audit
│   ├── UI_FUNCTIONALITY_AUDIT.md              # Comprehensive UI audit across all 10 app sections
│   ├── UI_FIX_VERIFICATION_REPORT.md          # Verification report for the 8 UI and functional fixes
│   └── TEST_REPORT.md                         # Automated unit tests execution report
├── outputs/                                   # Audited Evaluation Artifacts
│   ├── linear_regression/                     # LR plots and metrics.json
│   ├── random_forest/                         # RF plots, feature importances and metrics.json
│   ├── lstm/                                  # LSTM convergence, zoomed peak & metrics.json
│   ├── model_comparison/                      # Benchmark plots, reports and comparison summary
│   ├── peak_analysis/                         # Daily peaks, hourly distribution and performance CSVs
│   ├── peak_projection/                       # Scenario simulation reports and charts
│   └── project_dashboard/                     # HTML standalone dashboard & overview charts
├── tests/                                     # Automated Unit & Integration Tests
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
├── docker-compose.yml                         # Container orchestration configuration
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
cd Miniproject
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

Run the full automated test suite covering data validation, feature generation, model loading, prediction alignment, peak analysis, and dashboard generation:

```bash
python -m unittest discover tests
```

Expected outcome:
```text
Ran 28 tests in ~7.0s
OK (All 28 tests PASS)
```

---

## 13. Documented Limitations

1. **Absence of Live SCADA Connection:** Models are trained on historical 2021–2024 telemetry. Real-time dispatches are simulated via recent telemetry or user inputs.
2. **Missing Sampling Intervals:** Real grid telemetry contains 1,150 irregular gap transitions (> 5 minutes) and 21,569 missing slots across 4 years. Missing slots remain `NaN` without synthetic interpolation.
3. **Recursive Forecasting Error Accumulation:** In multi-step recursive forecasting (24-hour and 7-day horizons), lag errors accumulate across forward steps.
4. **Weather Dependency:** Projections assume persistence of the last observed weather conditions unless updated with live meteorological NWP data.
5. **Decision Tree Bounding:** Random Forest cannot extrapolate past its training leaf ceiling (~7,746 MW) during unprecedented heatwave records.

---

## 14. Docker Deployment

Deploy using Docker:
```bash
docker build -t delhi-power-demand:latest .
docker run -d -p 8501:8501 --name power-predict-app delhi-power-demand:latest
```

Or with Docker Compose:
```bash
docker compose up -d
```
Access the application at `http://localhost:8501`.
