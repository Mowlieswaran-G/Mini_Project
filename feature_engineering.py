"""
Feature Engineering Script for Delhi Power System Electricity Demand Forecasting
Project: AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System

Dataset Input: cleaned_delhi_powerdemand_5min_2021_to_2024.csv
Dataset Output: feature_engineered_delhi_powerdemand_5min_2021_to_2024.csv

Requirements:
1. Load cleaned dataset without modifying it.
2. Generate calendar/time features: year, month, day, hour, minute, day_of_week, is_weekend.
3. Compute exact datetime-based historical demand lag features:
   - demand_lag_5min (t - 5 minutes)
   - demand_lag_1hour (t - 1 hour)
   - demand_lag_24hour (t - 24 hours)
   - demand_lag_7day (t - 7 days)
   * Handled using exact timestamp matching to prevent gap corruption (DO NOT assume shift(n) is continuous).
4. Retain 6 core weather variables: temp, dwpt, rhum, wdir, wspd, pres.
5. Strictly avoid data leakage (no moving_avg_3, no future values, strictly causal).
6. Perform manual timestamp inspection checks.
7. Save output as feature_engineered_delhi_powerdemand_5min_2021_to_2024.csv.
"""

import os
import json
import pandas as pd
import numpy as np

def run_feature_engineering(
    input_path: str = "cleaned_delhi_powerdemand_5min_2021_to_2024.csv",
    output_path: str = "feature_engineered_delhi_powerdemand_5min_2021_to_2024.csv"
):
    print(f"Loading cleaned data from: {input_path}")
    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    total_rows = len(df)
    print(f"Loaded {total_rows} rows.")

    # 1. Calendar / Temporal Features
    print("Creating calendar/time features...")
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)  # 1 if Sat/Sun, 0 otherwise

    # 2. Causal Historical Demand Features (Exact Datetime-Mapped)
    print("Constructing exact datetime-based causal lag features...")
    # Lookup series indexed by datetime
    demand_lookup = df.set_index("datetime")["Power demand"]

    # Define exact historical time deltas
    lag_definitions = {
        "demand_lag_5min": pd.Timedelta(minutes=5),
        "demand_lag_1hour": pd.Timedelta(hours=1),
        "demand_lag_24hour": pd.Timedelta(hours=24),
        "demand_lag_7day": pd.Timedelta(days=7),
    }

    lag_stats = {}
    for col_name, delta in lag_definitions.items():
        # Match past timestamp; if historical slot is missing due to gap or start of series, returns NaN
        target_past_timestamps = df["datetime"] - delta
        df[col_name] = target_past_timestamps.map(demand_lookup)

        valid_count = int(df[col_name].notnull().sum())
        missing_count = int(df[col_name].isnull().sum())
        missing_pct = (missing_count / total_rows) * 100

        lag_stats[col_name] = {
            "valid_count": valid_count,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4),
            "historical_delta": str(delta),
            "lookahead_check": "Strictly historical past observation (Causal)"
        }
        print(f"  Feature {col_name} (t - {delta}): {valid_count} valid, {missing_count} missing ({missing_pct:.2f}%)")

    # 3. Manual Inspection on Selected Verification Points
    print("\n--- MANUAL TIMESTAMP VERIFICATION CHECKS ---")
    inspection_timestamps = [
        pd.Timestamp("2021-01-15 12:00:00"),
        pd.Timestamp("2022-06-10 15:30:00"),
        pd.Timestamp("2023-08-20 09:00:00"),
        pd.Timestamp("2024-06-19 15:05:00")  # Historic all-time peak
    ]

    inspection_results = []
    for dt in inspection_timestamps:
        row = df[df["datetime"] == dt].iloc[0]
        actual_current_demand = row["Power demand"]
        check_entry = {
            "datetime": str(dt),
            "actual_demand": actual_current_demand,
            "checks": []
        }
        print(f"\nVerifying Datetime: {dt} (Actual Current Demand = {actual_current_demand:.2f} MW)")
        for col_name, delta in lag_definitions.items():
            expected_past_dt = dt - delta
            expected_past_val = demand_lookup.get(expected_past_dt, np.nan)
            feature_val = row[col_name]
            is_match = (
                (pd.isna(feature_val) and pd.isna(expected_past_val)) or
                (np.isclose(feature_val, expected_past_val, atol=1e-4))
            )
            print(f"  {col_name}:")
            print(f"    Target past time : {expected_past_dt}")
            print(f"    Feature value    : {feature_val:.2f} MW")
            print(f"    Expected lookup  : {expected_past_val:.2f} MW")
            print(f"    Exact match      : {is_match}")
            check_entry["checks"].append({
                "feature": col_name,
                "past_timestamp": str(expected_past_dt),
                "feature_value": float(feature_val) if pd.notnull(feature_val) else None,
                "expected_value": float(expected_past_val) if pd.notnull(expected_past_val) else None,
                "verified_match": bool(is_match)
            })
            if not is_match:
                raise ValueError(f"Mismatch in lag calculation for {col_name} at {dt}!")
        inspection_results.append(check_entry)

    # 4. Verify gap boundary behavior (ensure no false continuity across gaps)
    gaps = df[df["datetime"].diff() > pd.Timedelta(minutes=5)]
    first_gap_idx = gaps.index[0]
    gap_row = df.loc[first_gap_idx]
    prev_row = df.loc[first_gap_idx - 1]
    print(f"\nGap Boundary Verification:")
    print(f"  Pre-gap timestamp : {prev_row['datetime']}")
    print(f"  Post-gap timestamp: {gap_row['datetime']} (Elapsed: {gap_row['datetime'] - prev_row['datetime']})")
    print(f"  Post-gap demand_lag_5min value: {gap_row['demand_lag_5min']} (Correctly NaN: {pd.isna(gap_row['demand_lag_5min'])})")
    if not pd.isna(gap_row['demand_lag_5min']):
        raise ValueError("Lag 5min incorrectly populated across a gap boundary!")

    # 5. Order and Select Columns
    ordered_columns = [
        "datetime",
        "Power demand",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "day_of_week",
        "is_weekend",
        "demand_lag_5min",
        "demand_lag_1hour",
        "demand_lag_24hour",
        "demand_lag_7day",
        "temp",
        "dwpt",
        "rhum",
        "wdir",
        "wspd",
        "pres"
    ]
    df_out = df[ordered_columns]

    # 6. Assess completeness when dropping rows with unavailable lag history
    all_lags_present = (
        df_out["demand_lag_5min"].notnull() &
        df_out["demand_lag_1hour"].notnull() &
        df_out["demand_lag_24hour"].notnull() &
        df_out["demand_lag_7day"].notnull()
    )
    rows_complete_lags = int(all_lags_present.sum())
    pct_complete_lags = (rows_complete_lags / total_rows) * 100
    rows_dropped_lags = total_rows - rows_complete_lags
    pct_dropped_lags = (rows_dropped_lags / total_rows) * 100

    print(f"\nLag History Completeness:")
    print(f"  Total rows: {total_rows}")
    print(f"  Rows with ALL 4 lags valid: {rows_complete_lags} ({pct_complete_lags:.2f}%)")
    print(f"  Rows with at least one missing lag: {rows_dropped_lags} ({pct_dropped_lags:.2f}%)")

    # 7. Save feature-engineered dataset
    df_out.to_csv(output_path, index=False)
    print(f"\nFeature-engineered dataset successfully saved to: {output_path}")

    # 8. Save metadata summary to json
    summary = {
        "final_row_count": len(df_out),
        "columns_count": len(df_out.columns),
        "columns": list(df_out.columns),
        "missing_values_per_feature": df_out.isnull().sum().to_dict(),
        "lag_statistics": lag_stats,
        "rows_with_all_lags_available": rows_complete_lags,
        "pct_rows_with_all_lags_available": round(pct_complete_lags, 2),
        "rows_dropped_if_na_lags_removed": rows_dropped_lags,
        "pct_rows_dropped_if_na_lags_removed": round(pct_dropped_lags, 2),
        "manual_inspections": inspection_results
    }
    with open("outputs/feature_engineering_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    summary = run_feature_engineering()
    print("\nFeature engineering execution finished successfully!")
