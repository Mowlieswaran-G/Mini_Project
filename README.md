# Delhi Power Demand Forecasting & Peak Load Analysis

A machine learning project for analyzing and forecasting 5-minute interval power demand in Delhi (2021–2024), evaluating weather dependencies, and projecting future peak load scenarios.

## Features & Pipeline
1. **Preprocessing (`preprocess_data.py`)**: Data cleaning, missing value handling, temporal alignment.
2. **Feature Engineering (`feature_engineering.py`)**: Lagged variables, rolling statistics, calendar indicators, and weather interaction terms.
3. **Exploratory Data Analysis (`eda_analysis.py`)**: Demand heatmaps, hourly/seasonal profiles, and weather correlations.
4. **Model Training & Comparison (`train_linear_regression.py`, `train_random_forest.py`, `compare_models.py`)**:
   - Baseline Linear Regression
   - Random Forest Regressor
   - Comprehensive performance metrics (MAE, RMSE, R², MAPE)
5. **Peak Demand Analysis (`run_peak_analysis.py`)**: High-demand interval evaluation and peak tracking.
6. **Scenario Projections (`peak_demand_projection.py`)**: Future peak demand simulation under extreme weather conditions.
7. **Interactive Dashboard (`project_results_dashboard.py`)**: Visualizes historical patterns, model evaluations, and scenario projections.

## Outputs
- Interactive HTML Dashboard: `outputs/project_dashboard/project_dashboard.html`
- Analysis Reports: Located in `outputs/` directory.

## Streamlit web application (Stage 11)

The application in `app/` loads the saved cleaned dataset, evaluation artefacts and model files; it does not retrain either model. It provides pages for the project overview, data exploration, model performance, single-timestamp prediction, recursive future forecasting, peak analysis, and methodology.

Install the required packages and start it from the project root:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app reads `cleaned_delhi_powerdemand_5min_2021_to_2024.csv`, the saved test-prediction CSVs, saved metrics in `outputs/`, and any available model pickle. The Linear Regression pickle is currently present at the project root. If the Random Forest pickle is absent, its saved evaluation results remain available but live Random Forest inference is shown as unavailable.
