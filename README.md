# ⚡ Power Predict AI — Delhi Electricity Demand & Peak Projection System

> **AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System**  
> An end-to-end, zero-leakage machine learning & deep learning system engineered to project granular (5-minute) power dispatch loads, forecast extreme heatwave grid stress, and evaluate operational capacity for the National Capital Territory of Delhi.

[![Python](https://img.shields.io/badge/Python-3.13%20%7C%203.14-blue.svg)](https://www.python.org/)
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
9. [Repository Directory Structure](#9-repository-directory-structure)
10. [Quick Start & Reproduction Guide](#10-quick-start--reproduction-guide)
11. [Docker Deployment](#11-docker-deployment)

---

## 1. Executive Summary & Problem Context

The National Capital Territory of Delhi is one of the world's most populous metropolitan regions (>33 million inhabitants), characterized by extreme continental climatic swings:
- **Winter (Dec–Jan):** Dense fog spells and temperatures down to ~4 °C trigger morning domestic space-heating and water geyser surges.
- **Summer & Monsoon (May–Jul):** Severe heatwaves exceeding **45 °C** and high relative humidity drive massive air-conditioning (HVAC) loads, representing over **50% of the entire grid demand**.

On **19 June 2024 at 15:05 IST**, Delhi recorded its **highest power demand in history: 8,631.53 MW**.

### Why Standard Forecasting Breaks at Extreme Peaks
Traditional non-linear machine learning models (such as tree-based ensembles) partition historical training data into bounded leaves. When historical unprecedented records occur, decision trees cannot predict higher than the training set ceiling. This project benchmarks **Linear Regression**, **Random Forest**, and a **PyTorch Long Short-Term Memory (LSTM) Recurrent Neural Network** across 4 years (2021–2024) of 5-minute dispatch telemetry to deliver accurate operational foresight and peak risk assessment.

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
│   • Streamlit Web Application (app/app.py) with Real-Time What-If Simulator     │
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
- **Historical Demand Envelope:** Minimum `1,302.0 MW` | Maximum `8,631.53 MW`.

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
- **Implementation:** [`train_linear_regression.py`](file:///c:/Users/MOWLIESWARAN/OneDrive/Documents/Mini_Project/train_linear_regression.py)
- **Artifact:** [`linear_regression_delhi_powerdemand.pkl`](file:///c:/Users/MOWLIESWARAN/OneDrive/Documents/Mini_Project/linear_regression_delhi_powerdemand.pkl) (~1.25 KB)
- **Mathematical Form:** Ordinary Least Squares:
  $$\hat{y} = \beta_0 + \sum_{i=1}^{17} \beta_i X_i$$
- **Strengths:**
  - Trains virtually instantaneously (~0.42 seconds).
  - Maintains unbounded linear slopes, allowing it to extrapolate to unprecedented peak records never observed during the training period.
- **Metrics:** Test MAE: `25.24 MW` | Test RMSE: `43.23 MW` | Test $R^2$: `0.9990`.

### 4.2 Random Forest Regressor
- **Implementation:** [`train_random_forest.py`](file:///c:/Users/MOWLIESWARAN/OneDrive/Documents/Mini_Project/train_random_forest.py)
- **Artifact:** [`random_forest_delhi_powerdemand.pkl`](file:///c:/Users/MOWLIESWARAN/OneDrive/Documents/Mini_Project/random_forest_delhi_powerdemand.pkl) (~225 MB)
- **Configuration:** 100 estimators, `max_depth=20`, `min_samples_leaf=2`, `n_jobs=-1`.
- **Strengths:**
  - Captures complex non-linear meteorological interactions across normal load cycles ($\le 5,589$ MW).
- **Limitation (Tree Leaf Bounding):**
  - Decision trees cannot predict values higher than the maximum target observation in their terminal training leaves (~7,695 MW). Under-predicts historic peaks severely.
- **Metrics:** Test MAE: `28.17 MW` | Test RMSE: `56.29 MW` | Test $R^2$: `0.9983`.

### 4.3 Long Short-Term Memory (LSTM) Neural Network
- **Implementation:** [`train_lstm.py`](file:///c:/Users/MOWLIESWARAN/OneDrive/Documents/Mini_Project/train_lstm.py)
- **Artifacts:**
  - PyTorch Checkpoint: [`lstm_delhi_powerdemand.pt`](file:///c:/Users/MOWLIESWARAN/OneDrive/Documents/Mini_Project/lstm_delhi_powerdemand.pt)
  - Inference Wrapper: [`lstm_delhi_powerdemand.pkl`](file:///c:/Users/MOWLIESWARAN/OneDrive/Documents/Mini_Project/lstm_delhi_powerdemand.pkl)
- **Architecture:**
  - Input: 17 scaled continuous features.
  - Multi-Timestep Sequence Window: 6 steps (30-minute lookback).
  - 2 Stacked LSTM Layers (hidden dimension: 64, dropout: 0.1).
  - Fully Connected Head: `Linear(64 -> 32) -> ReLU -> Linear(32 -> 1)`.
  - Optimizer: `AdamW` (learning rate: $2 \times 10^{-3}$, weight decay: $10^{-4}$) with `ReduceLROnPlateau`.
- **Strengths:**
  - Directly models continuous temporal trajectories and multi-step inertia.
  - Integrates smoothly with real-time inference via an automated scikit-learn compatible wrapper.

---

## 5. Comparative Performance Benchmark

All metrics evaluated strictly on the **70,825 hold-out test observations** (March to December 2024):

| Metric | Unit | Linear Regression | Random Forest | LSTM Deep Network | Best Performer |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test MAE** | MW | **25.24** | 28.17 | 27.63 | **Linear Regression** |
| **Test RMSE** | MW | **43.23** | 56.29 | 47.77 | **Linear Regression** |
| **Test $R^2$ Score** | — | **0.9990** | 0.9983 | 0.9987 | **Linear Regression** |
| **Training Time** | Seconds | **0.42 s** | 140.85 s | 133.33 s | **Linear Regression** |
| **Inference Time (70.8k)** | Seconds | **0.05 s** | 1.02 s | 1.71 s | **Linear Regression** |
| **Model Disk Footprint** | Bytes | **1.25 KB** | 225 MB | ~1.1 MB (.pt) | **Linear Regression** |
| **Historic Peak Error** | MW | **-71.12** (-0.82%) | -954.21 (-11.05%) | -451.86 (-5.23%) | **Linear Regression** |

---

## 6. Historic Delhi Peak Case Study

### The Event: 19 June 2024 at 15:05 IST
- **Ground Truth Demand:** **`8,631.53 MW`**
- **Atmospheric Conditions:** Ambient Temperature: **44.5 °C** | Dew Point: **19.5 °C** | Relative Humidity: **32%** | Heat Index: **~54 °C**.

### Model Predictions:
```
Ground Truth Peak:   ████████████████████████████████████████ 8,631.53 MW
Linear Regression:   █████████████████████████████████████▌   8,560.41 MW  (Error: -71.12 MW / -0.82%)
LSTM Deep Network:   ████████████████████████████████████     8,179.67 MW  (Error: -451.86 MW / -5.23%)
Random Forest:       ██████████████████████████████           7,677.32 MW  (Error: -954.21 MW / -11.05%)
```

**Key Finding:** Tree-based ensemble predictions flatten out when attempting to extrapolate past previous training ceilings, resulting in a dangerous ~1,000 MW shortfall during grid emergencies. Linear Regression maintains direct slope extrapolation, while LSTM's continuous activation layers cut the tree-based under-prediction error by more than **50%**.

---

## 7. Demand Regime & Extreme Value Analysis

Error distribution evaluated across quantile-based grid operating tiers:

| Operating Regime | Demand Boundary | Data Share | LR MAE | RF MAE | LSTM MAE | Regime Characteristic |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Normal Demand** | $\le 5,589$ MW | 75% | 23.97 MW | **23.65 MW** | 23.99 MW | RF slightly superior on standard loads |
| **High Demand** | $5,589 - 6,906$ MW | 20% | **28.67 MW** | 31.35 MW | 31.78 MW | Transition regime |
| **Extreme Peak** | $> 6,906$ MW | Top 5% | **30.58 MW** | 83.13 MW | 65.64 MW | Heavy air-conditioning stress |
| **Super Extreme** | $> 7,655$ MW | Top 1% | **27.78 MW** | 243.52 MW | 135.05 MW | RF error surges 9x; LSTM dampens errors |

---

## 8. Interactive Web Application (Streamlit)

The web dashboard is located in [`app/app.py`](file:///c:/Users/MOWLIESWARAN/OneDrive/Documents/Mini_Project/app/app.py) and provides utility grid operators with:
1. **Interactive What-If Scenario Builder:** Adjust ambient temperature, humidity, dew point, pressure, and momentum lags to simulate real-time dispatch reactions.
2. **Preset Atmospheric Stress Tests:** Preconfigured presets (e.g. *🔥 June Heatwave Record*, *🌙 Midnight AC Cooling Surge*, *❄️ Winter Cold Wave*, *🍃 Pleasant Spring Baseload*).
3. **Automated Autoregressive Lag Estimator:** Automatically estimates 5-min, 1-hour, 24-hour, and 7-day momentum values if historical data is unavailable.
4. **Historical Highlights Explorer:** Inspect all-time grid events and examine model predictions against actual historical telemetry.
5. **Model Comparison & Diagnostics:** Side-by-side visualization of Linear Regression, Random Forest, and LSTM trajectories.

---

## 9. Repository Directory Structure

```
Mini_Project/
├── app/                                       # Streamlit Web Application
│   ├── .streamlit/config.toml                 # Custom UI Theme configuration
│   ├── app.py                                 # Main Streamlit Dashboard
│   └── utils/
│       ├── data.py                            # Caching, model loaders & telemetry helpers
│       └── features.py                        # Causal feature generator & presets
├── outputs/                                   # Audited Evaluation Artifacts
│   ├── linear_regression/                     # LR plots and metrics.json
│   ├── random_forest/                         # RF plots, feature importances and metrics.json
│   ├── lstm/                                  # LSTM convergence, zoomed peak & metrics.json
│   ├── model_comparison/                      # Benchmark plots and comparison report
│   ├── peak_analysis/                         # Daily and monthly peak analysis CSVs
│   ├── peak_projection/                       # Scenario simulation reports
│   └── project_dashboard/                     # HTML standalone dashboard
├── cleaned_delhi_powerdemand_5min_2021_to_2024.csv    # Cleaned, zero-null telemetry
├── feature_engineered_delhi_powerdemand_5min_2021_to_2024.csv # 17 Causal features
├── train_delhi_powerdemand.csv                # Chronological 80% train partition (283,298 rows)
├── test_delhi_powerdemand.csv                 # Chronological 20% test partition (70,825 rows)
├── linear_regression_delhi_powerdemand.pkl    # Serialized Linear Regression binary
├── random_forest_delhi_powerdemand.pkl        # Serialized Random Forest binary
├── lstm_delhi_powerdemand.pt                  # Serialized PyTorch LSTM checkpoint
├── lstm_delhi_powerdemand.pkl                 # Inference-ready LSTM wrapper bundle
├── linear_regression_test_predictions.csv     # 70,825 Aligned test predictions
├── random_forest_test_predictions.csv         # 70,825 Aligned test predictions
├── lstm_test_predictions.csv                  # 70,825 Aligned test predictions
├── model_comparison_delhi_powerdemand.csv     # Side-by-side benchmark CSV
├── preprocess_data.py                         # Step 1: Preprocessing & leakage removal
├── eda_analysis.py                            # Step 2: Exploratory data analysis
├── feature_engineering.py                     # Step 3: 17 Causal feature calculation
├── prepare_modeling_dataset.py                # Step 4: Chronological train/test split
├── train_linear_regression.py                 # Step 5: Train & evaluate Linear Regression
├── train_random_forest.py                     # Step 6: Train & evaluate Random Forest
├── train_lstm.py                              # Step 7: Train & evaluate PyTorch LSTM
├── compare_models.py                          # Step 8: Comprehensive model benchmarking
├── run_peak_analysis.py                       # Step 9: Peak demand regime analytics
├── peak_demand_projection.py                  # Step 10: Scenario projection engine
├── project_results_dashboard.py               # Step 11: Export standalone HTML dashboard
├── Dockerfile                                 # Production container definition
├── docker-compose.yml                         # Container orchestration configuration
└── requirements.txt                           # Python project dependencies
```

---

## 10. Quick Start & Reproduction Guide

### Prerequisites
- Python 3.10 to 3.14 (64-bit)
- 4 GB+ RAM

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Mowlieswaran-G/Mini_Project.git
cd Mini_Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Scripts
```bash
# 1. Feature Engineering & Chronological Train/Test Partition
python feature_engineering.py
python prepare_modeling_dataset.py

# 2. Train Models
python train_linear_regression.py
python train_random_forest.py
python train_lstm.py

# 3. Model Comparison & Peak Analytics
python compare_models.py
python run_peak_analysis.py
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 11. Docker Deployment

Deploy the full application stack using Docker:

```bash
# Build the image
docker build -t delhi-power-demand:latest .

# Run container on port 8501
docker run -d -p 8501:8501 --name power-predict-app delhi-power-demand:latest
```

Or with Docker Compose:
```bash
docker compose up -d
```
Access the application at `http://localhost:8501`.

---

## ⚖️ License & Attribution
- **Dataset:** Delhi State Load Despatch Centre (SLDC) public grid telemetry & meteorological station logs (2021–2024).
- **Project Scope:** Academic & industrial research in machine learning applications for power systems engineering.
