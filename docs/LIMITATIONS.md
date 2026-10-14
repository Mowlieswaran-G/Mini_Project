# Documented System Limitations & Operational Boundaries

**Project Title:** AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System

---

## 1. Absence of Live SCADA / IoT Telemetry Feed
- **Description:** The system is trained and evaluated using historical telemetry records (2021–2024) from the Delhi State Load Despatch Centre (SLDC).
- **Impact:** Live real-time dispatch monitoring is simulated based on recent historical telemetry or user-defined inputs. The application is not directly connected to a real-time SCADA/EMS transmission bus.

## 2. Irregular Historical Sampling Intervals
- **Description:** Real-world utility telemetry contains transmission dropouts and sensor communication lags. Across the 4-year span, the raw dataset contains 1,150 irregular gap transitions (> 5 minutes) and 21,569 omitted historical 5-minute slots.
- **Impact:** The project deliberately avoided synthesizing artificial demand rows to maintain data authenticity. Autoregressive lags at gap boundaries return `NaN` and were excluded during model fitting.

## 3. Recursive Forecasting Error Accumulation
- **Description:** Multi-step forward forecasting (e.g. 24-hour and 7-day horizons) updates autoregressive lag features (`demand_lag_5min`, `demand_lag_1hour`, `demand_lag_24hour`, `demand_lag_7day`) recursively using earlier model predictions.
- **Impact:** Inevitable prediction errors at step $t$ propagate into lags for steps $t+1, t+12$, etc. Forecast uncertainty and prediction variance widen progressively over extended horizons.

## 4. Dependence on Upstream Meteorological Forecasts
- **Description:** Electricity demand in Delhi is heavily driven by weather conditions—particularly ambient temperature, relative humidity, and dew point during summer heatwaves.
- **Impact:** In forward multi-step forecasting, the model assumes persistence of the last observed weather conditions unless live meteorological numerical weather prediction (NWP) data is supplied. Forward projection reliability is fundamentally bounded by the accuracy of upstream weather forecasts.

## 5. Architectural Leaf Bounding on Extreme Peaks
- **Description:** Decision tree ensembles (Random Forest) cannot extrapolate beyond the maximum target value present in their terminal training leaves (~7,746 MW).
- **Impact:** During the unprecedented heatwave peak of June 19, 2024 (8,631.53 MW), Random Forest under-predicted by -954.21 MW (-11.05%). While Linear Regression and LSTM mitigate this limitation, all models exhibit a slight negative bias during sudden unexpected load spikes.

## 6. Simulation Alert Thresholds vs. Official Grid Limits
- **Description:** The user-configurable alert threshold in the application is an operational simulation aid designed for capacity studies.
- **Impact:** User-configured thresholds do not represent statutory or official SLDC grid emergency limits.
