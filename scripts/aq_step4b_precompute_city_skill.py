"""
AirWatchAI -- Step 4b: precompute per-city skill (model vs. persistence,
evaluated on each city's own held-out split) using the verified NumPy
forward pass, so the deployed Streamlit app never needs torch at all.

This mirrors aq_step2c_export_qgis.py's per_city_skill() exactly (same
windowing, same window-relative centering, same held-out-split rule), but
runs on the NumPy weights instead of the live torch model -- the two are
already verified numerically identical (aq_step4_export_weights_numpy.py,
max diff 2.4e-07), so this is not a second, divergent code path in spirit,
just the deployment-safe implementation of the same one.

Run (pure NumPy + pandas, no torch, no internet needed):
    python3 aq_step4b_precompute_city_skill.py
"""

import os
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SEQ_LEN, HORIZON = 12, 4
FEATURE_COLS = ["no2_log", "co_log", "aer_ai_mean", "precip_mm", "doy_sin", "doy_cos"]
NO2_COL_IDX = FEATURE_COLS.index("no2_log")
CO_COL_IDX = FEATURE_COLS.index("co_log")
TEST_CITIES = {"Bangkok", "Tehran", "Karachi", "Johannesburg"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def numpy_lstm_forecast(window, weights):
    W_ih, W_hh = weights["weight_ih_l0"], weights["weight_hh_l0"]
    b_ih, b_hh = weights["bias_ih_l0"], weights["bias_hh_l0"]
    hidden = W_hh.shape[1]
    h = np.zeros(hidden, dtype=np.float32)
    c = np.zeros(hidden, dtype=np.float32)
    for t in range(window.shape[0]):
        x_t = window[t]
        z = W_ih @ x_t + b_ih + W_hh @ h + b_hh
        i_t = sigmoid(z[0 * hidden:1 * hidden])
        f_t = sigmoid(z[1 * hidden:2 * hidden])
        g_t = np.tanh(z[2 * hidden:3 * hidden])
        o_t = sigmoid(z[3 * hidden:4 * hidden])
        c = f_t * c + i_t * g_t
        h = o_t * np.tanh(c)
    return weights["head_weight"] @ h + weights["head_bias"]


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "aq_combined_weekly.csv"), parse_dates=["period_start"])
    with open(os.path.join(MODELS_DIR, "normalization_stats.json")) as f:
        stats = json.load(f)
    weights = dict(np.load(os.path.join(MODELS_DIR, "no2_forecaster_weights.npz")))

    df_norm = df.copy()
    for col in ["no2_log", "co_log", "aer_ai_mean", "precip_mm"]:
        m, s = stats[col]["mean"], stats[col]["std"]
        df_norm[col] = (df_norm[col] - m) / s

    city_means = df.groupby("city")["no2_mean"].mean().sort_values()
    rank = {city: i + 1 for i, city in enumerate(city_means.index)}

    results = {}
    for city, g in df_norm.groupby("city"):
        g = g.sort_values("period_start").reset_index(drop=True)
        feats = g[FEATURE_COLS].values.astype(np.float32)
        target = g["no2_log"].values.astype(np.float32)
        splits = g["split"].values

        model_sq_err, persist_sq_err, n = 0.0, 0.0, 0
        for start in range(0, len(g) - SEQ_LEN - HORIZON + 1):
            tgt_slice = slice(start + SEQ_LEN, start + SEQ_LEN + HORIZON)
            tgt_splits = np.unique(splits[tgt_slice])
            if len(tgt_splits) != 1 or tgt_splits[0] == "train":
                continue
            window = feats[start:start + SEQ_LEN].copy()
            no2_anchor = window[:, NO2_COL_IDX].mean()
            co_anchor = window[:, CO_COL_IDX].mean()
            last_centered = window[-1, NO2_COL_IDX] - no2_anchor
            window[:, NO2_COL_IDX] -= no2_anchor
            window[:, CO_COL_IDX] -= co_anchor

            pred_centered = numpy_lstm_forecast(window, weights)
            y_centered = target[tgt_slice] - no2_anchor

            model_sq_err += np.sum((pred_centered - y_centered) ** 2)
            persist_sq_err += np.sum((last_centered - y_centered) ** 2)
            n += HORIZON

        if n == 0:
            results[city] = {"skill_pct": None, "n_eval_windows": 0, "held_out_split": None}
            continue
        model_mse = model_sq_err / n
        persist_mse = persist_sq_err / n
        skill = (persist_mse - model_mse) / persist_mse * 100
        held_out = "test (never trained on)" if city in TEST_CITIES else "val (time-holdout)"
        results[city] = {
            "role": "test" if city in TEST_CITIES else "train",
            "held_out_split": held_out,
            "skill_pct": round(float(skill), 1),
            "model_mse": round(float(model_mse), 4),
            "persistence_mse": round(float(persist_mse), 4),
            "n_eval_windows": int(n / HORIZON),
            "relative_pollution_rank_1to12": rank[city],
            "mean_no2_column_density": round(float(city_means[city]), 8),
        }
        print(f"  {city:15s} {results[city]['held_out_split']:22s} skill={skill:+.1f}%")

    out_path = os.path.join(MODELS_DIR, "city_skill_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
