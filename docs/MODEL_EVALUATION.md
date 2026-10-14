# Model Evaluation & Benchmark Analysis

**Evaluation Dataset:** Chronological 20% hold-out test set (`test_delhi_powerdemand.csv`), containing **70,825 observations** from **March 5, 2024 13:25:00** to **December 11, 2024 22:25:00**.

---

## 1. Verified Model Performance Metrics

All metrics were evaluated against the identical 70,825 test observations with matching timestamps:

| Category | Metric | Unit | Linear Regression (Baseline) | Random Forest (100 Trees) | PyTorch LSTM (Deep Network) | Analytical Characteristic |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Overall Test Performance** | **MAE** | MW | **25.24** | 28.17 | 27.63 | Linear Regression lowest overall error |
| | **RMSE** | MW | **43.23** | 56.29 | 47.77 | Linear Regression penalizes extreme errors least |
| | **R² Score** | — | **0.9990** | 0.9983 | 0.9987 | All models explain >99.8% of variance |
| **Computational Efficiency** | **Training Time** | Seconds | **0.42 s** | 140.85 s | 133.33 s | Linear Regression ~300x faster to fit |
| | **Inference Time (70.8k)** | Seconds | **0.05 s** | 1.02 s | 1.71 s | Linear Regression ~20x faster inference |
| | **Model Binary Size** | Bytes | **1.25 KB** | ~740.8 MB | ~1.1 MB (.pt+.pkl) | Linear Regression minimal disk footprint |
| **Highest Observed Peak**<br>*(2024-06-19 15:05:00)* | **Actual Demand** | MW | 8,631.53 | 8,631.53 | 8,631.53 | Ground truth test-period peak record |
| | **Predicted Demand** | MW | **8,560.41** | 7,677.32 | 8,179.67 | LR linear extrapolation; RF leaf bounding |
| | **Prediction Error** | MW | **-71.12** | -954.21 | -451.86 | LR error is -71 MW vs. RF -954 MW |
| | **Prediction Error** | % | **-0.82%** | -11.05% | -5.23% | LSTM cuts RF error by more than 50% |
| **Daily Peak Performance**<br>*(All 272 Test Days)* | **Daily Peak MAE** | MW | **39.86** | 63.88 | 61.62 | Linear Regression tracks daily crests closest |
| | **Daily Peak RMSE** | MW | **72.07** | 136.40 | 139.20 | Linear Regression lowest variance on daily peaks |
| | **Daily Peak MAPE** | % | **0.77%** | 1.07% | 1.04% | All models achieve ~1% mean peak percentage error |
| | **Peak Timing MAE** | Minutes | **28.3** | 65.8 | 62.4 | Linear Regression closest in peak timing |

---

## 2. Demand Regime Analysis

The hold-out test set was partitioned into quantile-based operational demand regimes to reveal model behavior across standard versus extreme loads:

| Operating Regime | Demand Boundary | Data Share | LR MAE | RF MAE | LSTM MAE | Operational Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Normal Demand** | $\le 5,589.38$ MW | 75.0% | 23.97 MW | **23.65 MW** | 23.99 MW | Random Forest slightly superior on standard loads |
| **High Demand** | $5,589 - 6,906$ MW | 20.0% | **28.67 MW** | 31.35 MW | 31.78 MW | Transition load tier |
| **Extreme Peak Demand** | $> 6,905.92$ MW | 5.0% | **30.58 MW** | 83.13 MW | 65.65 MW | Linear Regression substantially superior |
| **Super Extreme Peak** | $> 7,654.84$ MW | 1.0% | **27.78 MW** | 243.52 MW | 135.05 MW | Random Forest error surges 9x above training bound |

---

## 3. Analytical Findings on Peak Demand Under-Prediction

1. **Extrapolation Limits of Decision Trees:**
   Random Forest constructs piecewise constant approximations. It cannot predict values greater than the highest observed target value in its terminal training leaves (~7,746 MW). When an unprecedented heatwave peak of 8,631.53 MW occurs, the tree ensemble saturates, under-predicting by **-954.21 MW (-11.05%)**.
2. **Linear Regression Slope Preservation:**
   Because Linear Regression learns unbounded global hyperplane slopes, its strong positive coefficients for `demand_lag_5min` ($\beta \approx 1.046$) and `temp` ($\beta \approx +0.395$) allow it to project extreme heatwave loads with only **-71.12 MW (-0.82%)** error.
3. **LSTM Deep Recurrent Activation:**
   The PyTorch LSTM network's continuous non-linear activation functions (`tanh`, `sigmoid`, and linear head) allow it to extrapolate partially beyond the training ceiling, achieving **8,179.67 MW** and cutting Random Forest's peak under-prediction error by more than 50% (**-451.86 MW / -5.23%**).
