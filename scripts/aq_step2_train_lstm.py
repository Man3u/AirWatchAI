"""
AirWatchAI -- Step 2: train an LSTM to forecast NO2 4 weeks ahead, jointly
across 8 cities, validated on entirely unseen cities.

Architecture, and why:
  - Input: 12 consecutive weeks of 6 features (no2_log, co_log, aer_ai,
    precip_mm -- all normalized using TRAIN-only stats from step 1b -- plus
    doy_sin/doy_cos for seasonality).
  - An LSTM reads the 12-week sequence one week at a time, updating an
    internal hidden state it carries forward -- this is what lets it learn
    things like "NO2 has been climbing for 3 weeks and rain hasn't broken
    the trend" rather than only ever looking at the most recent value.
  - The LSTM's final hidden state (a compressed summary of the whole
    12-week window) feeds a small feedforward head that outputs 4 numbers:
    the predicted (normalized) NO2 level for each of the next 4 weeks.
  - Deliberately NO per-city embedding/lookup. A model that memorizes "city
    #3 behaves like X" cannot be evaluated on a city it has never seen --
    the embedding for a novel city would just be random noise. Every
    prediction here comes only from the 12-week pattern of the 6 shared
    features, which is the only way "the model generalizes to unseen
    cities" is a claim that actually means something.

Validation discipline (same standard as every earlier project):
  - TEST cities (Bangkok, Tehran, Karachi, Johannesburg) are NEVER touched
    during training, validation-based early stopping, or hyperparameter
    choice -- used exactly once, at the end, to report genuine
    generalization.
  - A naive persistence baseline (repeat the last observed value for all
    4 future weeks) is computed the same way, and the model has to beat it
    on the TRAIN cities' held-out time AND on the entirely-unseen TEST
    cities, or that's reported honestly, not hidden.

Run (pure PyTorch, CPU, no internet needed):
    pip install torch pandas numpy matplotlib
    python3 aq_step2_train_lstm.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SEQ_LEN = 12   # weeks of history the model sees
HORIZON = 4    # weeks ahead it forecasts
FEATURE_COLS = ["no2_log", "co_log", "aer_ai_mean", "precip_mm", "doy_sin", "doy_cos"]
TARGET_COL = "no2_log"

torch.manual_seed(42)
np.random.seed(42)


def load_data():
    df = pd.read_csv(os.path.join(OUT_DIR, "aq_combined_weekly.csv"), parse_dates=["period_start"])
    with open(os.path.join(MODELS_DIR, "normalization_stats.json")) as f:
        stats = json.load(f)
    return df, stats


def normalize(df, stats):
    df = df.copy()
    for col in ["no2_log", "co_log", "aer_ai_mean", "precip_mm"]:
        m, s = stats[col]["mean"], stats[col]["std"]
        df[col] = (df[col] - m) / s
    # doy_sin/doy_cos are already in [-1, 1], no normalization needed
    return df


def build_windows(df):
    """For every city, slide a (SEQ_LEN + HORIZON)-week window across its
    chronologically-sorted rows. A window belongs to whichever split its
    HORIZON target weeks fall in -- windows whose targets straddle a
    train/val boundary are dropped (rare, and ambiguous by construction)."""
    X, Y, splits, cities, target_dates = [], [], [], [], []
    for city, g in df.groupby("city"):
        g = g.sort_values("period_start").reset_index(drop=True)
        feats = g[FEATURE_COLS].values
        target = g[TARGET_COL].values
        n = len(g)
        for start in range(0, n - SEQ_LEN - HORIZON + 1):
            hist_slice = slice(start, start + SEQ_LEN)
            tgt_slice = slice(start + SEQ_LEN, start + SEQ_LEN + HORIZON)
            tgt_splits = g["split"].iloc[tgt_slice].unique()
            if len(tgt_splits) != 1:
                continue  # straddles a split boundary -- skip
            X.append(feats[hist_slice])
            Y.append(target[tgt_slice])
            splits.append(tgt_splits[0])
            cities.append(city)
            target_dates.append(g["period_start"].iloc[tgt_slice].tolist())
    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(splits),
        np.array(cities),
        target_dates,
    )


class NO2Forecaster(nn.Module):
    def __init__(self, n_features, hidden_size=32, horizon=HORIZON, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n: (1, batch, hidden_size)
        h = self.dropout(h_n.squeeze(0))
        return self.head(h)


def persistence_baseline(X, Y):
    """Repeat the last observed (normalized) no2_log for all HORIZON steps.
    X's last timestep, feature index 0, is no2_log (see FEATURE_COLS)."""
    last_val = X[:, -1, 0]  # (n,)
    pred = np.repeat(last_val[:, None], Y.shape[1], axis=1)
    mse = float(np.mean((pred - Y) ** 2))
    return mse


def train():
    df, stats = load_data()
    df_norm = normalize(df, stats)
    X, Y, splits, cities, target_dates = build_windows(df_norm)

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    print(f"Windows: train={train_mask.sum()}  val={val_mask.sum()}  test={test_mask.sum()}")

    # NOTE: torch.tensor()/from_numpy() on a numpy array can hard-fail with
    # "Numpy is not available" when torch's compiled numpy-interop layer and
    # the installed numpy version have an ABI mismatch (hit exactly this in
    # this environment: torch 2.2.2 vs numpy 2.2.6). Routing through a plain
    # Python list sidesteps torch's numpy C-bridge entirely -- slower for
    # huge arrays, irrelevant at our size (thousands of small windows).
    to_tensor = lambda a: torch.tensor(a.tolist(), dtype=torch.float32)
    X_train, Y_train = to_tensor(X[train_mask]), to_tensor(Y[train_mask])
    X_val, Y_val = to_tensor(X[val_mask]), to_tensor(Y[val_mask])
    X_test, Y_test = to_tensor(X[test_mask]), to_tensor(Y[test_mask])

    baseline_val_mse = persistence_baseline(X[val_mask], Y[val_mask])
    baseline_test_mse = persistence_baseline(X[test_mask], Y[test_mask])
    print(f"Persistence baseline -- val MSE: {baseline_val_mse:.4f}  test MSE: {baseline_test_mse:.4f}")

    model = NO2Forecaster(n_features=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    BATCH_SIZE = 64
    MAX_EPOCHS = 200
    PATIENCE = 15

    n_train = X_train.shape[0]
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_train[idx], Y_train[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, Y_val).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}"
                  f"  (best={best_val_loss:.4f}, no_improve={epochs_no_improve})")

        if epochs_no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, Y_test).item()

    val_skill = (baseline_val_mse - best_val_loss) / baseline_val_mse * 100
    test_skill = (baseline_test_mse - test_loss) / baseline_test_mse * 100

    print(f"\nFinal results:")
    print(f"  Val   -- model MSE: {best_val_loss:.4f}  persistence MSE: {baseline_val_mse:.4f}  skill: {val_skill:+.1f}%")
    print(f"  Test  -- model MSE: {test_loss:.4f}  persistence MSE: {baseline_test_mse:.4f}  skill: {test_skill:+.1f}%")
    print(f"  (Test = {sorted(set(cities[test_mask]))}, cities the model NEVER saw during training)")

    # -- save everything needed by the app later --
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "no2_forecaster.pt"))
    metrics = {
        "seq_len": SEQ_LEN, "horizon": HORIZON, "feature_cols": FEATURE_COLS,
        "val_mse": best_val_loss, "test_mse": test_loss,
        "baseline_val_mse": baseline_val_mse, "baseline_test_mse": baseline_test_mse,
        "val_skill_pct": val_skill, "test_skill_pct": test_skill,
        "test_cities": sorted(set(cities[test_mask].tolist())),
        "train_cities": sorted(set(cities[train_mask].tolist())),
        "epochs_trained": len(history["train_loss"]),
    }
    with open(os.path.join(MODELS_DIR, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model -> models/no2_forecaster.pt")
    print(f"Saved metrics -> models/training_metrics.json")

    # -- training curve plot --
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_loss"], label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (normalized)"); ax.legend(); ax.set_title("Training curve")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=120)
    plt.close(fig)

    # -- forecast-vs-actual example plots: one train city, one test city --
    plot_example(model, df_norm, stats, city=sorted(set(cities[train_mask]))[0], tag="train_example")
    plot_example(model, df_norm, stats, city=sorted(set(cities[test_mask]))[0], tag="test_example_UNSEEN")

    return model, metrics


def plot_example(model, df_norm, stats, city, tag):
    g = df_norm[df_norm["city"] == city].sort_values("period_start").reset_index(drop=True)
    feats = g[FEATURE_COLS].values.astype(np.float32)
    dates = g["period_start"].values
    m, s = stats["no2_log"]["mean"], stats["no2_log"]["std"]

    preds, pred_dates, actual_denorm = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(g) - SEQ_LEN - HORIZON + 1, HORIZON):
            x = torch.tensor(feats[start:start + SEQ_LEN].tolist(), dtype=torch.float32).unsqueeze(0)
            # .tolist() instead of .numpy() -- same broken torch<->numpy
            # bridge, this time going tensor -> array; a plain Python list
            # sidesteps it in both directions.
            pred_norm = model(x).squeeze(0).tolist()
            pred_denorm = [v * s + m for v in pred_norm]
            preds.extend(pred_denorm)
            pred_dates.extend(dates[start + SEQ_LEN:start + SEQ_LEN + HORIZON].tolist())

    actual_denorm = (g["no2_log"].values * s + m)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, actual_denorm, label="actual (log NO2)", color="#2b6cb0", linewidth=1)
    ax.plot(pred_dates, preds, label="model forecast (4wk ahead, stitched)", color="#e03131",
             linewidth=1, alpha=0.8)
    ax.set_title(f"{city} -- log NO2, actual vs. rolling 4-week forecast" +
                 (" [NEVER seen in training]" if "UNSEEN" in tag else ""))
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"forecast_{tag}_{city.replace(' ', '_')}.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    train()
