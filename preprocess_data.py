"""
Preprocessing Script for Delhi Power System Electricity Demand Dataset
Project: AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System

Dataset: powerdemand_5min_2021_to_2024_with weather.csv
Output: cleaned_delhi_powerdemand_5min_2021_to_2024.csv

Requirements:
1. Load original CSV without modifying it.
2. Remove 'Unnamed: 0' column.
3. Convert 'datetime' to pandas datetime and sort chronologically.
4. Verify duplicate timestamps.
5. Drop 'moving_avg_3' to prevent target data leakage.
6. Impute 540 missing 'wdir' values using forward-fill (causal, no lookahead bias).
7. Preserve all authentic electricity demand values without inventing or modifying them.
8. Preserve original observed time series without synthesizing missing time grid slots.
9. Verify validity of Power demand (no missing, negative, or zero values).
10. Retain all 6 weather features and datetime components.
11. Save as a separate new cleaned CSV file.
"""

import os
import pandas as pd
import numpy as np

def preprocess_delhi_power_demand(
    input_path: str = "powerdemand_5min_2021_to_2024_with weather.csv",
    output_path: str = "cleaned_delhi_powerdemand_5min_2021_to_2024.csv"
):
    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)
    original_row_count = len(df)
    original_col_count = df.shape[1]
    print(f"Original shape: {original_row_count} rows, {original_col_count} columns")

    # 1. Remove 'Unnamed: 0' index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("Dropped 'Unnamed: 0' index column.")

    # 2. Convert 'datetime' to proper pandas datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    print("Converted 'datetime' column to pd.to_datetime.")

    # 3. Sort chronologically and reset index
    df = df.sort_values("datetime").reset_index(drop=True)
    is_sorted = df["datetime"].is_monotonic_increasing
    print(f"Sorted chronologically: {is_sorted}")

    # 4. Check duplicate timestamps
    dup_count = df["datetime"].duplicated().sum()
    print(f"Duplicate timestamps count: {dup_count}")
    if dup_count > 0:
        raise ValueError(f"Found {dup_count} duplicate timestamps!")

    # 5. Drop 'moving_avg_3' to avoid target data leakage
    if "moving_avg_3" in df.columns:
        df = df.drop(columns=["moving_avg_3"])
        print("Dropped 'moving_avg_3' column to prevent target data leakage.")

    # 6. Handle 540 missing 'wdir' values
    initial_null_wdir = df["wdir"].isnull().sum()
    print(f"Missing 'wdir' values before handling: {initial_null_wdir}")
    
    # Forward fill: Causal, preserves preceding hourly station observation without future lookahead
    # Calm wind (wspd == 0) naturally retains the last stable direction or calm indicator
    df["wdir"] = df["wdir"].ffill()
    
    # In the unlikely event of any leading nulls, backfill
    if df["wdir"].isnull().sum() > 0:
        df["wdir"] = df["wdir"].bfill()
        
    post_null_wdir = df["wdir"].isnull().sum()
    print(f"Missing 'wdir' values after forward-fill: {post_null_wdir}")

    # 7. Quality check on 'Power demand'
    demand_col = "Power demand"
    null_demand = df[demand_col].isnull().sum()
    neg_demand = (df[demand_col] < 0).sum()
    zero_demand = (df[demand_col] == 0).sum()
    min_demand = df[demand_col].min()
    max_demand = df[demand_col].max()

    print(f"Power demand checks:")
    print(f"  - Missing: {null_demand}")
    print(f"  - Negative: {neg_demand}")
    print(f"  - Zero: {zero_demand}")
    print(f"  - Min: {min_demand:.2f} MW")
    print(f"  - Max: {max_demand:.2f} MW")

    if null_demand > 0 or neg_demand > 0 or zero_demand > 0:
        raise ValueError("Invalid power demand values detected!")

    # 8. Document time gaps without inventing synthetic rows
    diffs = df["datetime"].diff()
    expected_step = pd.Timedelta(minutes=5)
    gaps = df[diffs > expected_step]
    gap_count = len(gaps)
    total_missing_steps = int(((df["datetime"].max() - df["datetime"].min()) / expected_step) + 1 - len(df))
    print(f"Time series gap documentation:")
    print(f"  - Consecutive intervals matching 5-min: {(diffs == expected_step).sum()}")
    print(f"  - Irregular gap transitions (> 5 min): {gap_count}")
    print(f"  - Total omitted 5-min intervals across timeline: {total_missing_steps}")

    # 9. Verify retained columns
    expected_weather = ["temp", "dwpt", "rhum", "wdir", "wspd", "pres"]
    for w in expected_weather:
        if w not in df.columns:
            raise KeyError(f"Weather feature {w} missing!")

    # 10. Check total missing values remaining across all columns
    total_nulls = df.isnull().sum().sum()
    print(f"Total missing values across entire cleaned dataframe: {total_nulls}")

    # 11. Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset successfully saved to: {output_path}")

    # Return summary statistics
    summary = {
        "original_row_count": original_row_count,
        "cleaned_row_count": len(df),
        "columns_remaining": list(df.columns),
        "missing_values_after_cleaning": df.isnull().sum().to_dict(),
        "duplicate_timestamps": dup_count,
        "min_demand": min_demand,
        "max_demand": max_demand,
        "min_datetime": str(df["datetime"].min()),
        "max_datetime": str(df["datetime"].max()),
        "irregular_gaps_count": gap_count,
        "omitted_intervals_count": total_missing_steps,
    }
    return summary

if __name__ == "__main__":
    summary = preprocess_delhi_power_demand()
    print("\n=== PREPROCESSING SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
