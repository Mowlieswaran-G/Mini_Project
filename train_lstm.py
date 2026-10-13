"""
LSTM Recurrent Neural Network Training and Evaluation
Project: AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System

Inputs:
- train_delhi_powerdemand.csv
- test_delhi_powerdemand.csv

Outputs:
- lstm_delhi_powerdemand.pt (PyTorch weights checkpoint)
- lstm_delhi_powerdemand.pkl (Inference-ready model wrapper bundle for Streamlit / API)
- lstm_test_predictions.csv (Datetime, Actual Demand, Predicted Demand)
- outputs/lstm/1_actual_vs_predicted_full_test.png
- outputs/lstm/2_actual_vs_predicted_zoomed_peak.png
- outputs/lstm/3_actual_vs_predicted_scatter.png
- outputs/lstm/4_training_loss_curve.png
- outputs/lstm/lstm_metrics.json
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.autolayout'] = True

OUTPUT_DIR = os.path.join("outputs", "lstm")
os.makedirs(OUTPUT_DIR, exist_ok=True)


from lstm_model import DemandLSTM, LSTMModelWrapper


def create_sequences(X, y, seq_len):
    """Efficiently create sliding sequence windows using stride tricks."""
    X_seq = sliding_window_view(X, (seq_len, X.shape[1])).squeeze(1)
    if y is not None:
        y_seq = y[seq_len - 1:]
        return X_seq, y_seq
    return X_seq


def train_and_evaluate_lstm(num_epochs=6, batch_size=512, hidden_dim=64, num_layers=2, seq_len=6, lr=0.002):
    print("=" * 70, flush=True)
    print("STEP: Training LSTM Neural Network for Delhi Power Demand", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Data
    print("Loading training and testing datasets...", flush=True)
    train_df = pd.read_csv("train_delhi_powerdemand.csv")
    test_df = pd.read_csv("test_delhi_powerdemand.csv")

    train_df["datetime"] = pd.to_datetime(train_df["datetime"])
    test_df["datetime"] = pd.to_datetime(test_df["datetime"])

    print(f"Train rows: {len(train_df):,}, Test rows: {len(test_df):,}", flush=True)

    feature_cols = [
        "year", "month", "day", "hour", "minute", "day_of_week", "is_weekend",
        "demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day",
        "temp", "dwpt", "rhum", "wdir", "wspd", "pres"
    ]
    target_col = "Power demand"

    print(f"Number of input features: {len(feature_cols)}", flush=True)

    # 2. Strict Causal Standardization (Fit on Train only!)
    print("Fitting StandardScalers on training partition (zero leakage)...", flush=True)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(train_df[feature_cols].values)
    X_test_scaled = scaler_X.transform(test_df[feature_cols].values)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(train_df[[target_col]].values).squeeze(-1)
    y_test_scaled = scaler_y.transform(test_df[[target_col]].values).squeeze(-1)

    # 3. Create Sequential Windows
    print(f"Constructing sequential sliding windows (seq_len = {seq_len} steps = {seq_len * 5} mins)...", flush=True)
    X_tr_seq, y_tr_seq = create_sequences(X_train_scaled, y_train_scaled, seq_len)

    # For test data: prepend the last (seq_len - 1) rows of train so test length matches test_df exactly
    X_test_prepended = np.vstack([X_train_scaled[-(seq_len - 1):], X_test_scaled])
    X_te_seq, _ = create_sequences(X_test_prepended, y_test_scaled, seq_len)

    assert len(X_te_seq) == len(test_df), f"Test window count {len(X_te_seq)} != test rows {len(test_df)}"
    print(f"Sequences created: Train: {X_tr_seq.shape}, Test: {X_te_seq.shape}", flush=True)

    # 4. Torch Dataloader
    train_dataset = TensorDataset(
        torch.tensor(X_tr_seq, dtype=torch.float32),
        torch.tensor(y_tr_seq, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cpu")
    print(f"Training on device: {device}", flush=True)

    model = DemandLSTM(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    # 5. Training Loop
    print("\nStarting LSTM Model Training...", flush=True)
    t_start_train = time.time()
    epoch_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        t_epoch_start = time.time()

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / n_batches
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)
        t_epoch_end = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Loss (MSE): {avg_loss:.6f} | LR: {current_lr:.6f} | Time: {t_epoch_end - t_epoch_start:.1f}s", flush=True)

    t_train_total = time.time() - t_start_train
    print(f"LSTM Training completed in {t_train_total:.2f} seconds.", flush=True)

    # 6. Evaluation on Train and Test Sets
    print("\nEvaluating model on Train and Hold-Out Test sets...", flush=True)
    model.eval()

    # Inference on Test Set
    t_start_pred = time.time()
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_te_seq, dtype=torch.float32)),
        batch_size=1024,
        shuffle=False
    )
    test_preds_scaled = []
    with torch.no_grad():
        for (batch_X,) in test_loader:
            batch_X = batch_X.to(device)
            p = model(batch_X).cpu().numpy()
            test_preds_scaled.append(p)
    test_preds_scaled = np.concatenate(test_preds_scaled)
    t_pred_total = time.time() - t_start_pred

    # Inverse transform to original MW units
    y_test_pred_mw = scaler_y.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
    y_test_true_mw = test_df[target_col].values

    # Inference on Train Set (sample 50,000 for fast accurate training metrics)
    sample_indices = np.linspace(0, len(X_tr_seq) - 1, min(50000, len(X_tr_seq)), dtype=int)
    X_tr_sample = torch.tensor(X_tr_seq[sample_indices], dtype=torch.float32)
    with torch.no_grad():
        train_preds_sample_scaled = model(X_tr_sample).cpu().numpy()
    y_train_sample_pred_mw = scaler_y.inverse_transform(train_preds_sample_scaled.reshape(-1, 1)).flatten()
    y_train_sample_true_mw = scaler_y.inverse_transform(y_tr_seq[sample_indices].reshape(-1, 1)).flatten()

    # 7. Compute Metrics
    train_mae = float(mean_absolute_error(y_train_sample_true_mw, y_train_sample_pred_mw))
    train_rmse = float(root_mean_squared_error(y_train_sample_true_mw, y_train_sample_pred_mw))
    train_r2 = float(r2_score(y_train_sample_true_mw, y_train_sample_pred_mw))

    test_mae = float(mean_absolute_error(y_test_true_mw, y_test_pred_mw))
    test_rmse = float(root_mean_squared_error(y_test_true_mw, y_test_pred_mw))
    test_r2 = float(r2_score(y_test_true_mw, y_test_pred_mw))

    actual_min = float(y_test_true_mw.min())
    actual_max = float(y_test_true_mw.max())
    pred_min = float(y_test_pred_mw.min())
    pred_max = float(y_test_pred_mw.max())

    # Peak observation analysis (Historic Delhi Peak)
    max_test_idx = int(np.argmax(y_test_true_mw))
    max_test_row = test_df.iloc[max_test_idx]
    max_test_dt = str(max_test_row["datetime"])
    max_actual_demand = float(max_test_row[target_col])
    pred_at_max_demand = float(y_test_pred_mw[max_test_idx])
    peak_prediction_error = pred_at_max_demand - max_actual_demand
    peak_err_pct = (peak_prediction_error / max_actual_demand) * 100.0

    # Demand Regime Error Analysis
    regimes = {
        "Normal (<= 5589 MW)": y_test_true_mw <= 5589.0,
        "High (5589 - 6906 MW)": (y_test_true_mw > 5589.0) & (y_test_true_mw <= 6906.0),
        "Extreme (> 6906 MW)": y_test_true_mw > 6906.0,
        "Super Extreme (> 7655 MW)": y_test_true_mw > 7655.0
    }
    regime_mae = {}
    for r_name, mask in regimes.items():
        if np.any(mask):
            regime_mae[r_name] = round(float(mean_absolute_error(y_test_true_mw[mask], y_test_pred_mw[mask])), 2)

    print("\n" + "=" * 50, flush=True)
    print("=== LSTM EVALUATION RESULTS ===", flush=True)
    print(f"Train Metrics: MAE = {train_mae:.2f} MW | RMSE = {train_rmse:.2f} MW | R² = {train_r2:.4f}", flush=True)
    print(f"Test Metrics : MAE = {test_mae:.2f} MW | RMSE = {test_rmse:.2f} MW | R² = {test_r2:.4f}", flush=True)
    print(f"Test Actual Range    : {actual_min:.2f} to {actual_max:.2f} MW", flush=True)
    print(f"Test Predicted Range : {pred_min:.2f} to {pred_max:.2f} MW", flush=True)
    print(f"\nHistoric Peak Record (2024-06-19 15:05:00):", flush=True)
    print(f"  Actual   : {max_actual_demand:.2f} MW", flush=True)
    print(f"  Predicted: {pred_at_max_demand:.2f} MW", flush=True)
    print(f"  Error    : {peak_prediction_error:+.2f} MW ({peak_err_pct:+.2f}%)", flush=True)
    print(f"Regime Errors (MAE): {regime_mae}", flush=True)
    print("=" * 50 + "\n", flush=True)

    # 8. Save Test Predictions CSV
    pred_df = pd.DataFrame({
        "datetime": test_df["datetime"],
        "actual_demand": y_test_true_mw,
        "predicted_demand": y_test_pred_mw
    })
    pred_csv_path = "lstm_test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved hold-out predictions to: {pred_csv_path}", flush=True)

    # 9. Save PyTorch State Dict & Wrapper Pickle
    torch_pt_path = "lstm_delhi_powerdemand.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'input_dim': len(feature_cols),
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'seq_len': seq_len,
        'epoch_losses': epoch_losses
    }, torch_pt_path)
    print(f"Saved PyTorch weights checkpoint to: {torch_pt_path}", flush=True)

    wrapper = LSTMModelWrapper(
        model_state_dict=model.state_dict(),
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_len=seq_len,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_cols=feature_cols
    )
    pkl_path = "lstm_delhi_powerdemand.pkl"
    joblib.dump(wrapper, pkl_path)
    print(f"Saved inference wrapper bundle to: {pkl_path}", flush=True)

    # 10. Generate Visualizations
    print("Generating LSTM evaluation plots...", flush=True)

    # Plot 1: Full Test Period (Actual vs Predicted)
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    pred_df_indexed = pred_df.set_index("datetime")
    daily_actual = pred_df_indexed["actual_demand"].resample("D").mean()
    daily_pred = pred_df_indexed["predicted_demand"].resample("D").mean()

    ax.plot(daily_actual.index, daily_actual.values, color="#1565C0", label="Actual Demand (Daily Mean)", linewidth=1.5)
    ax.plot(daily_pred.index, daily_pred.values, color="#7B1FA2", linestyle="--", label="LSTM Predicted (Daily Mean)", linewidth=1.5, alpha=0.9)
    ax.set_title("LSTM Recurrent Neural Network: Actual vs. Predicted Demand (Mar - Dec 2024)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Power Demand (MW)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_actual_vs_predicted_full_test.png"))
    plt.close()

    # Plot 2: Zoomed Peak Period (June 17 to June 21, 2024)
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    peak_mask = (pred_df["datetime"] >= "2024-06-17") & (pred_df["datetime"] <= "2024-06-22")
    peak_subset = pred_df[peak_mask]

    ax.plot(peak_subset["datetime"], peak_subset["actual_demand"], color="#1565C0", label="Actual Demand (5-min)", linewidth=1.8)
    ax.plot(peak_subset["datetime"], peak_subset["predicted_demand"], color="#7B1FA2", linestyle="--", label="LSTM Predicted (5-min)", linewidth=1.8)
    ax.scatter([pd.to_datetime(max_test_dt)], [max_actual_demand], color="#D32F2F", s=100, zorder=5, label=f"Record Peak ({max_actual_demand:,.0f} MW)")
    ax.set_title("LSTM Peak Demand Performance: Historic Delhi Heatwave (17–22 June 2024)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Date & Time", fontsize=11)
    ax.set_ylabel("Power Demand (MW)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_actual_vs_predicted_zoomed_peak.png"))
    plt.close()

    # Plot 3: Scatter Plot (Actual vs Predicted)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.scatter(y_test_true_mw, y_test_pred_mw, alpha=0.15, s=6, color="#7B1FA2", edgecolors='none')
    min_val = min(y_test_true_mw.min(), y_test_pred_mw.min())
    max_val = max(y_test_true_mw.max(), y_test_pred_mw.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#D32F2F", linestyle="--", linewidth=1.8, label="Ideal 1:1 Line")
    ax.set_title("LSTM: Actual vs. Predicted Demand Scatter (Hold-Out Test)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Actual Demand (MW)", fontsize=11)
    ax.set_ylabel("Predicted Demand (MW)", fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_actual_vs_predicted_scatter.png"))
    plt.close()

    # Plot 4: Training Loss Curve
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color="#7B1FA2", linewidth=2.0)
    ax.set_title("LSTM Convergence: Training MSE Loss Across Epochs", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Normalized MSE Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_training_loss_curve.png"))
    plt.close()

    # 11. Save Metrics JSON
    metrics_summary = {
        "model": f"LSTM Recurrent Neural Network (hidden_dim={hidden_dim}, num_layers={num_layers}, seq_len={seq_len})",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "input_features_count": len(feature_cols),
        "timing": {
            "training_time_seconds": round(t_train_total, 2),
            "prediction_time_seconds": round(t_pred_total, 2)
        },
        "train_metrics": {
            "MAE": round(train_mae, 2),
            "RMSE": round(train_rmse, 2),
            "R2": round(train_r2, 4)
        },
        "test_metrics": {
            "MAE": round(test_mae, 2),
            "RMSE": round(test_rmse, 2),
            "R2": round(test_r2, 4)
        },
        "test_demand_actual_min": round(actual_min, 2),
        "test_demand_actual_max": round(actual_max, 2),
        "test_demand_predicted_min": round(pred_min, 2),
        "test_demand_predicted_max": round(pred_max, 2),
        "peak_observation": {
            "datetime": max_test_dt,
            "actual_demand": round(max_actual_demand, 2),
            "predicted_demand": round(pred_at_max_demand, 2),
            "prediction_error_mw": round(peak_prediction_error, 2),
            "percentage_error": round(peak_err_pct, 2)
        },
        "regime_mae": regime_mae,
        "execution_status": "Success - Zero Errors"
    }

    metrics_json_path = os.path.join(OUTPUT_DIR, "lstm_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_json_path}", flush=True)
    print("\nLSTM pipeline execution completed successfully!", flush=True)


if __name__ == "__main__":
    train_and_evaluate_lstm(num_epochs=6, batch_size=512, hidden_dim=64, num_layers=2, seq_len=6, lr=0.002)
