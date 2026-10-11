"""
Dataset Preparation and Chronological Train-Test Split
Project: AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System

Input: feature_engineered_delhi_powerdemand_5min_2021_to_2024.csv
Outputs:
- train_delhi_powerdemand.csv (80% chronological training data)
- test_delhi_powerdemand.csv (20% chronological testing data)

Candidate Input Features (17):
1. Calendar: year, month, day, hour, minute, day_of_week, is_weekend (7)
2. Historical Demand Lags: demand_lag_5min, demand_lag_1hour, demand_lag_24hour, demand_lag_7day (4)
3. Weather: temp, dwpt, rhum, wdir, wspd, pres (6)

Target: Power demand (MW)
"""

import os
import json
import pandas as pd
import numpy as np

def prepare_and_split_dataset(
    input_path: str = "feature_engineered_delhi_powerdemand_5min_2021_to_2024.csv",
    train_output_path: str = "train_delhi_powerdemand.csv",
    test_output_path: str = "test_delhi_powerdemand.csv",
    train_ratio: float = 0.80
):
    print(f"Loading feature-engineered data from: {input_path}")
    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    initial_count = len(df)
    print(f"Total raw rows: {initial_count}")

    # 1. Filter rows where any of the 4 demand lag features is missing
    lag_cols = ["demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day"]
    df_valid = df.dropna(subset=lag_cols).copy()
    
    # 2. Ensure strict chronological order
    df_valid = df_valid.sort_values("datetime").reset_index(drop=True)
    total_valid = len(df_valid)
    dropped_count = initial_count - total_valid
    print(f"Valid rows with complete lag history: {total_valid} ({total_valid / initial_count * 100:.2f}%)")
    print(f"Rows dropped due to incomplete lag history: {dropped_count} ({dropped_count / initial_count * 100:.2f}%)")

    # 3. Define feature columns
    calendar_features = ["year", "month", "day", "hour", "minute", "day_of_week", "is_weekend"]
    lag_features = ["demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day"]
    weather_features = ["temp", "dwpt", "rhum", "wdir", "wspd", "pres"]
    all_input_features = calendar_features + lag_features + weather_features
    target_col = "Power demand"

    # Select columns to retain in output datasets (including datetime for tracking and plotting)
    columns_to_save = ["datetime", target_col] + all_input_features
    df_valid = df_valid[columns_to_save]

    # Verify no missing values exist across any feature
    null_counts = df_valid.isnull().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Unexpected missing values in valid dataset:\n{null_counts[null_counts > 0]}")

    # 4. Chronological 80/20 Train-Test Split (NO SHUFFLING)
    train_size = int(total_valid * train_ratio)
    test_size = total_valid - train_size

    train_df = df_valid.iloc[:train_size].reset_index(drop=True)
    test_df = df_valid.iloc[train_size:].reset_index(drop=True)

    # 5. Chronological Verification
    train_start = train_df["datetime"].min()
    train_end = train_df["datetime"].max()
    test_start = test_df["datetime"].min()
    test_end = test_df["datetime"].max()

    is_chronological = test_start > train_end
    print(f"\nChronological Split Verification:")
    print(f"  Training set : {len(train_df)} rows ({len(train_df)/total_valid*100:.2f}%) | {train_start} to {train_end}")
    print(f"  Testing set  : {len(test_df)} rows ({len(test_df)/total_valid*100:.2f}%) | {test_start} to {test_end}")
    print(f"  Test strictly after Train: {is_chronological}")

    if not is_chronological:
        raise ValueError("Chronological split failed! Overlap detected between train and test sets.")

    # 6. Save Separate Training and Testing CSVs
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    print(f"\nSaved training set to: {train_output_path}")
    print(f"Saved testing set to : {test_output_path}")

    # Summary dictionary
    summary = {
        "total_valid_observations_before_split": total_valid,
        "rows_dropped_due_to_missing_lags": dropped_count,
        "training_row_count": len(train_df),
        "testing_row_count": len(test_df),
        "train_percentage": round(len(train_df) / total_valid * 100, 2),
        "test_percentage": round(len(test_df) / total_valid * 100, 2),
        "training_start_datetime": str(train_start),
        "training_end_datetime": str(train_end),
        "testing_start_datetime": str(test_start),
        "testing_end_datetime": str(test_end),
        "chronological_verification_passed": bool(is_chronological),
        "target_feature": target_col,
        "input_features_count": len(all_input_features),
        "all_input_features": all_input_features,
        "calendar_features": calendar_features,
        "historical_demand_lags": lag_features,
        "weather_features": weather_features,
        "note_on_datetime": "datetime is preserved in CSV for reference/tracking but excluded from numerical feature matrix"
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/dataset_split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    summary = prepare_and_split_dataset()
    print("\nDataset preparation and chronological split completed successfully!")
