# Project Summary: Delhi Electricity Demand & Peak Projection System
**Generated:** 2026-10-14 10:20:42 | **Evaluation Period:** March 5, 2024 13:25:00 to December 11, 2024 22:25:00

## 1. System Overview
- **Goal:** Project granular 5-minute power demand and evaluate extreme peak demand risks for the Delhi power system.
- **Dataset:** 393,440 5-minute telemetry intervals (2021–2024) from Delhi State Load Despatch Centre with matching weather.
- **Zero-Leakage Engineering:** Causal forward-fill for wind direction, removal of unlagged moving averages, exact retrospective lag matching.

## 2. Validated Multi-Model Results (70,825 Hold-Out Test Rows)

| Model | Test MAE (MW) | Test RMSE (MW) | R² Score | Peak Error at 8,631.5 MW | Daily Peak MAE (MW) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **25.24** | **43.23** | **0.9990** | **-71.12 MW (-0.82%)** | **39.86** |
| **Random Forest (100 Trees)** | 28.17 | 56.29 | 0.9983 | -954.21 MW (-11.05%) | 63.88 |
| **PyTorch LSTM Network** | 27.63 | 47.77 | 0.9987 | -451.86 MW (-5.23%) | 61.62 |

## 3. Key Findings on Peak Demand
1. **Decision Tree Bounding:** Random Forest under-predicts unprecedented heatwave peaks because its leaves cannot extrapolate beyond training set boundaries (~7,746 MW).
2. **Linear Extrapolation:** Linear Regression maintains linear sensitivity, accurately projecting the 8,631.53 MW peak with -0.82% error.
3. **LSTM Sequential Modeling:** The 2-layer LSTM cuts Random Forest's peak under-prediction error by more than half (-5.23% error).

## 4. Key Limitations
- Absence of live SCADA telemetry feed (relies on historical data or manual inputs).
- Recursive error accumulation during long forward forecasting horizons.
- Dependence on upstream meteorological forecast accuracy.
