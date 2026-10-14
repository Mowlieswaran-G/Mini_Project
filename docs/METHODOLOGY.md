# Methodological Architecture & Feature Engineering

**Project Title:** AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System  
**Dataset:** 393,440 continuous 5-minute dispatch intervals (2021-01-01 to 2024-12-12) from Delhi State Load Despatch Centre (SLDC) combined with matching weather telemetry.

---

## 1. Zero-Leakage Data Preprocessing

To ensure rigorous real-world validity and eliminate lookahead bias, the preprocessing pipeline applies four strict data engineering principles:

1. **Target Leakage Elimination:** The original raw telemetry file (`powerdemand_5min_2021_to_2024_with weather.csv`) contained an unlagged `moving_avg_3` column that looked ahead into contemporaneous and future dispatches. This column was completely eliminated prior to feature generation and is not present in any modeling dataset.
2. **Strict Causal Wind Direction Imputation:** The only missing feature in the raw dataset was wind direction (`wdir`), which had 540 null values. These missing values were imputed strictly using causal forward-fill (`ffill()`), preserving preceding station observations without looking ahead into future measurements.
3. **Preservation of Authentic Sampling:** Across the 4-year span, the telemetry records 1,150 irregular gap transitions (> 5 minutes) and 21,569 omitted historical 5-minute intervals. No synthetic or interpolated rows were generated to artificially fill these gaps, preserving authentic power grid behavior.
4. **Chronological Splitting Without Shuffling:** The dataset was partitioned strictly in time:
   - **Training Set (80%):** 283,298 observations (`2021-01-08 01:00:00` to `2024-03-05 13:20:00`).
   - **Hold-Out Test Set (20%):** 70,825 observations (`2024-03-05 13:25:00` to `2024-12-11 22:25:00`).
   - The first 7 days (39,317 rows) are used strictly as a warm-up period for the 7-day autoregressive lag filter.

---

## 2. Causal Feature Engineering (17 Input Features)

The feature matrix consists of 17 candidate features categorized into three distinct operational domains:

### 2.1 Calendar & Temporal Features (7)
- `year` (2021–2024): Captures annual structural load growth and Delhi metropolitan expansion.
- `month` (1–12): Captures strong continental seasonal cycles (Winter heating, Summer cooling, Monsoon humidity).
- `day` (1–31): Captures intra-month billing and activity cycles.
- `hour` (0–23): Diurnal cycle capturing morning ramps (~10:00), afternoon peaks (~15:00), and nocturnal AC surges (~23:00).
- `minute` (0–55): 5-minute dispatch interval.
- `day_of_week` (0=Monday, 6=Sunday): Captures weekly commercial and industrial load variations.
- `is_weekend` (0 or 1): Binary indicator (1 for Saturday/Sunday, 0 for Weekday).

### 2.2 Autoregressive Demand Lags (4)
Lags are constructed using **exact timestamp lookup** via series index mapping (`df['datetime'] - delta`), rather than naive positional shifting (`shift(k)`). If a past timestamp is missing due to a grid gap, the value is set to `NaN` and dropped, preventing temporal distortion across missing intervals:
- `demand_lag_5min` ($t - 5\text{ minutes}$): Short-term grid momentum.
- `demand_lag_1hour` ($t - 60\text{ minutes}$): 1-hour intra-day inertia.
- `demand_lag_24hour` ($t - 1440\text{ minutes}$): 24-hour diurnal rhythm matching same time yesterday.
- `demand_lag_7day` ($t - 10080\text{ minutes}$): 7-day weekly cycle matching same time last week.

### 2.3 Atmospheric Weather Variables (6)
- `temp`: Ambient outdoor dry-bulb temperature (°C).
- `dwpt`: Dew point temperature (°C).
- `rhum`: Relative humidity (%).
- `wdir`: Wind direction in degrees (°).
- `wspd`: Wind speed in km/h.
- `pres`: Surface atmospheric barometric pressure in hPa.

---

## 3. Modeling Architectures

Three distinct model families are evaluated on the identical 70,825 hold-out test observations:

1. **Ordinary Least Squares Linear Regression (Baseline):**
   - High extrapolation capacity at extreme record peaks.
   - Sub-second fitting time (~0.42 s).
2. **Random Forest Regressor (100 Trees):**
   - Captures non-linear meteorological interactions across normal load tiers ($\le 5,589$ MW).
   - Constrained by tree-leaf bounding at unprecedented heatwave peaks.
3. **PyTorch Deep LSTM Network:**
   - 2-layer stacked LSTM with sequence sliding windows (6 steps = 30 minutes).
   - Captures continuous temporal memory, cutting Random Forest's peak under-prediction error by more than 50%.
