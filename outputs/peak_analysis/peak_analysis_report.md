# Dedicated Peak Demand Analysis Report: Delhi Power System
## Test Period: March 5, 2024 to December 11, 2024 (272 Days, 5-min resolution with some missing intervals)

### 1. Key Peak Identification & Summary Statistics
- **Highest Daily Peak:** **`8631.53 MW`**
- **Date of Highest Peak:** **`2024-06-19`**
- **Timestamp of Highest Peak:** **`2024-06-19 15:05:00`**
- **Average Daily Peak Demand:** **`5519.61 MW`**
- **Median Daily Peak Demand:** **`5393.60 MW`**

### 2. Top 10 Daily Peak Demand Observations

| date | time | Actual Peak (MW) | LR Predicted (MW) | RF Predicted (MW) |
| --- | --- | --- | --- | --- |
| 2024-06-19 | 15:05 | 8631.53 | 8560.405600198928 | 7677.320841720776 |
| 2024-06-18 | 15:15 | 8610.85 | 8595.576778333043 | 7654.470167911253 |
| 2024-05-29 | 15:45 | 8254.3 | 8236.505299451112 | 7658.747220000002 |
| 2024-06-17 | 23:30 | 8201.11 | 8196.372892875625 | 7655.335011785715 |
| 2024-07-31 | 15:20 | 8163.08 | 8138.119072591791 | 7669.128002380952 |
| 2024-05-30 | 15:20 | 8051.03 | 8058.815376801544 | 7659.535515714287 |
| 2024-05-31 | 23:25 | 8019.96 | 8002.685103122345 | 7675.62282166667 |
| 2024-07-30 | 15:30 | 7999.99 | 7961.865749800541 | 7643.524353333333 |
| 2024-05-06 | 15:50 | 7989.34 | 7950.194292709093 | 7655.486164101736 |
| 2024-05-22 | 15:05 | 7954.84 | 7930.966437199948 | 7649.492257380954 |

### 3. Peak Timing & Diurnal Distribution
- **Most Common Peak Hour:** **Hour `15:00` (3:00 PM – 3:55 PM)**, accounting for **`89` days (`32.7%` of all days)**.
- **Secondary Peak Hours:** Hour 10:00 (10 AM morning ramp, 40 days / 14.7%), Hour 18:00 (6 PM evening transition, 26 days / 9.6%), Hour 23:00 (11 PM nocturnal AC cooling peak, 24 days / 8.8%).
- **Weekday vs Weekend Timing Dynamics:**
  - On weekdays, peaks cluster heavily during midday/afternoon (Hours 10 to 15, driven by commercial and industrial operations).
  - On weekends, daytime commercial load drops, causing nocturnal hours (Hours 22 and 23) to surge as a larger proportion of daily peaks due to residential air conditioning loads.

### 4. Monthly Maximum Demand Analysis

| month_name | maximum_demand_mw | peak_datetime | average_daily_peak_demand_mw | days_observed |
| --- | --- | --- | --- | --- |
| March | 7826.89 | 2024-03-06 15:05:00 | 4291.58 | 26 |
| April | 7809.14 | 2024-04-06 14:55:00 | 4930.2 | 27 |
| May | 8254.3 | 2024-05-29 15:45:00 | 6705.14 | 28 |
| June | 8631.53 | 2024-06-19 15:05:00 | 6769.67 | 27 |
| July | 8163.08 | 2024-07-31 15:20:00 | 6416.78 | 31 |
| August | 7309.49 | 2024-08-06 00:10:00 | 5777.61 | 31 |
| September | 6840.38 | 2024-09-06 22:50:00 | 5456.74 | 30 |
| October | 7430.38 | 2024-10-06 15:50:00 | 4933.11 | 31 |
| November | 7652.97 | 2024-11-06 23:40:00 | 4419.61 | 30 |
| December | 7740.33 | 2024-12-06 23:40:00 | 5351.86 | 11 |

### 5. Model Peak Demand Performance Comparison

| model | daily_peak_mae_mw | daily_peak_rmse_mw | mean_percentage_error_pct | mean_absolute_percentage_error_pct | max_absolute_peak_error_mw | all_time_peak_pred_mw | all_time_peak_error_mw | all_time_peak_error_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | 39.86 | 72.07 | -0.56 | 0.77 | 611.24 | 8560.41 | -71.12 | -0.82 |
| Random Forest | 63.88 | 136.4 | -0.83 | 1.07 | 956.38 | 7677.32 | -954.21 | -11.05 |

### 6. Critical Analytical Insights
1. **Extrapolation Limits of Decision Trees on Peaks:** Random Forest's daily peak MAE is `63.88 MW` with an RMSE of `136.40 MW`, compared to Linear Regression's MAE of `39.86 MW` and RMSE of `72.07 MW`. The discrepancy is heavily concentrated during the extreme summer heatwave spikes where actual demand exceeded 7,700 MW.
2. **Under-Prediction Bias on Peak Loads:** Both models show a slight negative mean percentage error on peak days (Linear Regression: -0.56%, Random Forest: -0.83%), indicating that during sudden unexpected surges, autoregressive models conservatively track slightly behind the peak crest.
3. **Historical Context vs. Future Projection:** This analysis evaluates actual test-period peak observations under observed meteorological conditions. Future scenario projections (e.g., assessing grid vulnerability under a hypothetical 48 °C sustained heatwave) would require explicit exogenous weather scenario modeling.
