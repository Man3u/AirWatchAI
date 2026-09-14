"""
AirWatchAI -- Step 4: export the trained LSTM's weights to a plain .npz
file, and provide a pure-NumPy re-implementation of its forward pass.

Why: the deployed Streamlit app should be as light and fast-starting as
possible (free-tier hosting, viewers who click a link expect it to load
in seconds, not spin up a multi-hundred-MB torch install). The trained
network is tiny -- one 32-unit LSTM layer plus a linear head, four weight
matrices total -- small enough to hand-implement the exact same forward
pass in NumPy with no meaningful loss of precision. This also sidesteps
this sandbox's own torch<->numpy ABI mismatch entirely for the deployed
app (a different, cleaner environment than the one used for training).

This is NOT a re-trained or approximated model -- it is the identical
trained weights, running through hand-written versions of the same LSTM
equations PyTorch itself uses. Verified numerically against the live
torch model below before being trusted for anything.

Run:
    python3 aq_step4_export_weights_numpy.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SEQ_LEN, HORIZON = 12, 4
FEATURE_COLS = ["no2_log", "co_log", "aer_ai_mean", "precip_mm", "doy_sin", "doy_cos"]
NO2_COL_IDX = FEATURE_COLS.index("no2_log")
CO_COL_IDX = FEATURE_COLS.index("co_log")


class NO2Forecaster(nn.Module):
    def __init__(self, n_features, hidden_size=32, horizon=HORIZON, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n.squeeze(0))
        return self.head(h)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def numpy_lstm_forecast(window, weights):
    """window: (SEQ_LEN, n_features) numpy array, already window-centered.
    weights: dict from the exported .npz. Returns (HORIZON,) centered
    prediction -- exactly mirrors NO2Forecaster.forward() with dropout
    disabled (as it is at eval time)."""
    W_ih, W_hh = weights["weight_ih_l0"], weights["weight_hh_l0"]  # (128,6), (128,32)
    b_ih, b_hh = weights["bias_ih_l0"], weights["bias_hh_l0"]      # (128,), (128,)
    hidden = W_hh.shape[1]
    h = np.zeros(hidden, dtype=np.float32)
    c = np.zeros(hidden, dtype=np.float32)

    # PyTorch stacks the 4 gates in order [input, forget, cell(g), output],
    # each a (hidden, *) block.
    for t in range(window.shape[0]):
        x_t = window[t]
        z = W_ih @ x_t + b_ih + W_hh @ h + b_hh  # (128,)
        i_t = sigmoid(z[0 * hidden:1 * hidden])
        f_t = sigmoid(z[1 * hidden:2 * hidden])
        g_t = np.tanh(z[2 * hidden:3 * hidden])
        o_t = sigmoid(z[3 * hidden:4 * hidden])
        c = f_t * c + i_t * g_t
        h = o_t * np.tanh(c)

    pred = weights["head_weight"] @ h + weights["head_bias"]  # (HORIZON,)
    return pred


def verify(model, weights, df_norm):
    """Run both the torch model and the numpy re-implementation on several
    real windows and confirm they agree to floating precision -- the
    discipline this whole project has followed: verify, don't assume."""
    g = df_norm[df_norm["city"] == "Delhi"].sort_values("period_start").reset_index(drop=True)
    feats = g[FEATURE_COLS].values.astype(np.float32)

    max_diff = 0.0
    n_checked = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(g) - SEQ_LEN - HORIZON + 1, 17):  # sample every 17th window
            window = feats[start:start + SEQ_LEN].copy()
            no2_anchor = window[:, NO2_COL_IDX].mean()
            co_anchor = window[:, CO_COL_IDX].mean()
            window[:, NO2_COL_IDX] -= no2_anchor
            window[:, CO_COL_IDX] -= co_anchor

            torch_pred = model(torch.tensor(window.tolist(), dtype=torch.float32).unsqueeze(0)).squeeze(0).tolist()
            numpy_pred = numpy_lstm_forecast(window, weights)

            diff = np.max(np.abs(np.array(torch_pred) - numpy_pred))
            max_diff = max(max_diff, diff)
            n_checked += 1

    print(f"Verified {n_checked} windows -- max abs difference (torch vs numpy): {max_diff:.2e}")
    assert max_diff < 1e-4, "NumPy re-implementation diverges from the trained torch model -- do not ship this."
    print("PASS -- numpy forward pass is numerically identical to the trained model.")


def main():
    sd = torch.load(os.path.join(MODELS_DIR, "no2_forecaster.pt"))
    # .tolist() rather than .numpy() -- this sandbox's torch build has a
    # known ABI mismatch with the installed numpy (see aq_step2_train_lstm.py
    # for the full explanation); routing through plain Python lists
    # sidesteps it identically here.
    weights = {
        "weight_ih_l0": np.array(sd["lstm.weight_ih_l0"].tolist(), dtype=np.float32),
        "weight_hh_l0": np.array(sd["lstm.weight_hh_l0"].tolist(), dtype=np.float32),
        "bias_ih_l0": np.array(sd["lstm.bias_ih_l0"].tolist(), dtype=np.float32),
        "bias_hh_l0": np.array(sd["lstm.bias_hh_l0"].tolist(), dtype=np.float32),
        "head_weight": np.array(sd["head.weight"].tolist(), dtype=np.float32),
        "head_bias": np.array(sd["head.bias"].tolist(), dtype=np.float32),
    }
    np.savez(os.path.join(MODELS_DIR, "no2_forecaster_weights.npz"), **weights)
    print(f"Saved {os.path.join(MODELS_DIR, 'no2_forecaster_weights.npz')}")

    # -- verification --
    model = NO2Forecaster(n_features=len(FEATURE_COLS))
    model.load_state_dict(sd)
    model.eval()

    df = pd.read_csv(os.path.join(OUT_DIR, "aq_combined_weekly.csv"), parse_dates=["period_start"])
    with open(os.path.join(MODELS_DIR, "normalization_stats.json")) as f:
        stats = json.load(f)
    df_norm = df.copy()
    for col in ["no2_log", "co_log", "aer_ai_mean", "precip_mm"]:
        m, s = stats[col]["mean"], stats[col]["std"]
        df_norm[col] = (df_norm[col] - m) / s

    verify(model, weights, df_norm)


if __name__ == "__main__":
    main()
