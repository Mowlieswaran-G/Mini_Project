# Model Comparison: Linear Regression vs. Random Forest Regression
## Delhi Electricity Demand and Peak Demand Projection

### 1. Dataset and Alignment Verification
- **Test Observations:** `70,825` observations (5-minute observations with some missing intervals from March 5, 2024 to December 11, 2024).
- **Timestamp Alignment:** 100% identical timestamps across both test prediction files.
- **Target Ground Truth:** 100% identical `actual_demand` values across both files.

### 2. Overall Performance Metrics

| Category | Metric | Unit | Linear Regression | Random Forest | Advantage / Characteristic |
| --- | --- | --- | --- | --- | --- |
| Overall Test Performance | MAE (Mean Absolute Error) | MW | 25.24 | 28.17 | LR lower overall error |
| Overall Test Performance | RMSE (Root Mean Squared Error) | MW | 43.23 | 56.29 | LR penalizes extremes less |
| Overall Test Performance | R² Score | Dimensionless | 0.999 | 0.9983 | Both explain >99.8% variance |
| Computational Cost | Training Time | Seconds | 0.42 | 140.85 | LR ~300x faster to fit |
| Computational Cost | Prediction Time (70,825 rows) | Seconds | 0.05 | 1.02 | LR ~20x faster inference |

### 3. Peak Demand Projection at Historic Grid Record (2024-06-19 15:05:00)

| Category | Metric | Unit | Linear Regression | Random Forest | Advantage / Characteristic |
| --- | --- | --- | --- | --- | --- |
| Peak Demand (2024-06-19 15:05:00) | Actual Peak Demand | MW | 8631.53 | 8631.53 | Historic Delhi record |
| Peak Demand (2024-06-19 15:05:00) | Predicted Peak Demand | MW | 8560.41 | 7677.32 | LR extrapolates; RF caps |
| Peak Demand (2024-06-19 15:05:00) | Peak Prediction Error (MW) | MW | -71.12 | -954.21 | LR error is -71 MW vs RF -954 MW |
| Peak Demand (2024-06-19 15:05:00) | Peak Prediction Error (%) | % | -0.82 | -11.05 | LR -0.82% vs RF -11.05% |

### 4. Demand Regime Error Analysis (Quantile Thresholds)

| Category | Metric | Unit | Linear Regression | Random Forest | Advantage / Characteristic |
| --- | --- | --- | --- | --- | --- |
| Regime MAE: Normal Demand | Demand <= 5589 MW (75% data) | MW | 23.97 | 23.65 | RF slightly more accurate in normal regime |
| Regime MAE: High Demand | Demand 5589 to 6906 MW (20% data) | MW | 28.67 | 31.35 | LR slightly more accurate |
| Regime MAE: Extreme Peak Demand | Demand > 6906 MW (Top 5% data) | MW | 30.58 | 83.13 | LR substantially superior on extremes |
| Regime MAE: Super Extreme Peak | Demand > 7655 MW (Top 1% data) | MW | 27.78 | 243.52 | RF error surges ~9x above training ceiling |

### 5. Architectural Distinctions and Key Findings
1. **Normal Operational Loads (<= 75th percentile):** Random Forest delivers slightly superior precision (MAE 23.65 MW vs 23.97 MW), effectively capturing non-linear interactions across standard diurnal variations.
2. **Extreme Peak Loads (> 95th and > 99th percentiles):** Linear Regression maintains stable errors (30.58 MW and 27.78 MW), whereas Random Forest errors escalate drastically to 83.13 MW and 243.52 MW because decision trees cannot predict values higher than the maximum training leaf average (~7,695 MW).
3. **Computational Efficiency:** Linear Regression trains virtually instantaneously (~0.4s) compared to Random Forest (~141s), offering faster inference.
