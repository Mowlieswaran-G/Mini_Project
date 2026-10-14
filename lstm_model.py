"""
LSTM Neural Network Architecture and Model Wrapper for Delhi Power Demand Forecasting.
Provides the PyTorch model and scikit-learn compatible inference wrapper with safe import fallback.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    from typing import Any
    torch: Any = None
    nn: Any = None
    HAS_TORCH = False

if HAS_TORCH:
    class DemandLSTM(nn.Module):
        """Deep LSTM Network for Sequential Electricity Demand Projection."""
        def __init__(self, input_dim=17, hidden_dim=64, num_layers=2, dropout=0.1):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out.squeeze(-1)
else:
    class DemandLSTM:
        """Placeholder when PyTorch is not installed in the local environment."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is not installed in the current environment. "
                "Please run 'pip install torch' to use the LSTM architecture."
            )

        def load_state_dict(self, *args, **kwargs):
            pass

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for LSTM inference.")

class LSTMModelWrapper:
    """
    Scikit-learn compatible inference wrapper for the trained PyTorch LSTM model.
    Enables direct .predict(X) calls from scripts, benchmarks, and Streamlit app.
    """
    def __init__(self, model_state_dict, input_dim=17, hidden_dim=64, num_layers=2,
                 seq_len=6, scaler_X=None, scaler_y=None, feature_cols=None):
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is required to execute the LSTMModelWrapper. "
                "Please install PyTorch in your Python environment."
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.feature_cols = feature_cols

        # Reconstruct torch model
        self.model: Any = DemandLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0
        )
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

    def predict(self, X):
        """
        Accepts numpy array or pandas DataFrame of shape (N, 17) or (N, seq_len, 17).
        Returns 1D numpy array of predictions in original MW units.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to run LSTM inference.")

        if isinstance(X, pd.DataFrame):
            if self.feature_cols is not None:
                X_vals = X[self.feature_cols].values
            else:
                X_vals = X.values
        else:
            X_vals = np.array(X)

        self.model.eval()
        with torch.no_grad():
            if X_vals.ndim == 2:
                n_samples, n_features = X_vals.shape
                # Scale features
                if self.scaler_X is not None:
                    X_scaled = self.scaler_X.transform(X_vals)
                else:
                    X_scaled = X_vals

                if n_samples == 1:
                    # Single-instance interactive what-if input
                    x_seq = np.tile(X_scaled, (1, self.seq_len, 1))
                    t_in = torch.tensor(x_seq, dtype=torch.float32)
                    y_pred_scaled = self.model(t_in).numpy()
                else:
                    # Batch of samples: tile each row along sequence dimension
                    x_seq = np.tile(X_scaled[:, np.newaxis, :], (1, self.seq_len, 1))
                    t_in = torch.tensor(x_seq, dtype=torch.float32)
                    y_pred_scaled = self.model(t_in).numpy()
            elif X_vals.ndim == 3:
                orig_shape = X_vals.shape
                flat_X = X_vals.reshape(-1, orig_shape[-1])
                if self.scaler_X is not None:
                    flat_X = self.scaler_X.transform(flat_X)
                X_scaled = flat_X.reshape(orig_shape)
                t_in = torch.tensor(X_scaled, dtype=torch.float32)
                y_pred_scaled = self.model(t_in).numpy()
            else:
                raise ValueError(f"Unexpected input dimension: {X_vals.ndim}")

            if y_pred_scaled.ndim == 0:
                y_pred_scaled = y_pred_scaled.reshape(1)

            # Inverse scale back to MW
            if self.scaler_y is not None:
                y_pred_mw = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            else:
                y_pred_mw = y_pred_scaled.flatten()

            return y_pred_mw
