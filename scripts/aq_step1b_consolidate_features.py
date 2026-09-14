"""
AirWatchAI -- Step 1b: consolidate the 12 per-city CSVs into one modeling-
ready dataset.

Good news from step 1: there turned out to be no missing weeks at all (0
null values across all 12 cities) -- weekly aggregation over a whole AOI
box is robust enough that some valid pixels were always found, even in
persistently cloudy/hazy cities like Jakarta. So there is no gap-filling
to do. What this script does instead, uniformly across every city:

  1. Flags (does not drop) low-confidence weeks: LOW_CONF_THRESHOLD applied
     identically to no2_valid_frac / co_valid_frac / aer_ai_valid_frac for
     every city. A flagged week stays in the dataset; the model and the
     app are both free to weight it differently, but nothing is deleted or
     silently excluded.
  2. Log-transforms NO2 and CO (strictly positive, right-skewed column
     densities -- standard practice, gives the network a better-behaved
     signal than raw skewed values). Aerosol Index can be negative, so it's
     left as-is and standardized instead.
  3. Adds day-of-year seasonality features (sin/cos encoding) -- pollution
     has strong yearly cycles (heating season, monsoon washout, etc.) and
     an LSTM benefits from an explicit seasonal signal rather than having
     to infer the calendar from 400 raw timestamps.
  4. Assigns a split: TEST cities are held out ENTIRELY (never touched
     during training or normalization fitting); TRAIN cities are further
     split by TIME into train/val (last 15% of weeks = validation) so we
     also catch overfitting to a specific time period, not just to cities.
  5. Fits normalization (mean/std) using ONLY the train split, and saves
     those stats to disk -- fitting on val/test data would leak information
     the model shouldn't have and would make the "held-out city" test
     meaningless.

Run (pure pandas, no internet/Earth Engine needed):
    pip install pandas numpy
    python3 aq_step1b_consolidate_features.py
"""

import os
import json
import glob
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LOW_CONF_THRESHOLD = 0.5  # same threshold, every city, every product

TEST_CITIES = {"Bangkok", "Tehran", "Karachi", "Johannesburg"}
VAL_TIME_FRAC = 0.15  # last 15% of weeks, within TRAIN cities only


def load_all_cities():
    frames = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "*_aq_timeseries.csv"))):
        df = pd.read_csv(path)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["period_start"] = pd.to_datetime(combined["period_start"])
    combined = combined.sort_values(["city", "period_start"]).reset_index(drop=True)
    return combined


def add_features(df):
    df["no2_log"] = np.log(df["no2_mean"].clip(lower=1e-7))
    df["co_log"] = np.log(df["co_mean"].clip(lower=1e-7))

    doy = df["period_start"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    df["low_confidence"] = (
        (df["no2_valid_frac"] < LOW_CONF_THRESHOLD)
        | (df["co_valid_frac"] < LOW_CONF_THRESHOLD)
        | (df["aer_ai_valid_frac"] < LOW_CONF_THRESHOLD)
    )
    return df


def assign_split(df):
    df["split"] = "train"
    df.loc[df["city"].isin(TEST_CITIES), "split"] = "test"

    for city in sorted(set(df["city"]) - TEST_CITIES):
        city_mask = df["city"] == city
        n = city_mask.sum()
        n_val = max(1, int(n * VAL_TIME_FRAC))
        city_idx = df.index[city_mask].sort_values()
        val_idx = city_idx[-n_val:]  # most RECENT weeks -> validation
        df.loc[val_idx, "split"] = "val"
    return df


FEATURE_COLS = ["no2_log", "co_log", "aer_ai_mean", "precip_mm"]


def fit_normalization(df):
    train_only = df[df["split"] == "train"]
    stats = {}
    for col in FEATURE_COLS:
        stats[col] = {"mean": float(train_only[col].mean()), "std": float(train_only[col].std())}
    return stats


def main():
    df = load_all_cities()
    df = add_features(df)
    df = assign_split(df)
    stats = fit_normalization(df)

    print(f"Total rows: {len(df)}  ({df['city'].nunique()} cities)")
    print(f"Split counts:\n{df['split'].value_counts()}")
    print(f"Low-confidence weeks: {df['low_confidence'].sum()} / {len(df)} "
          f"({df['low_confidence'].mean()*100:.1f}%)")
    print(f"Test cities (never in train/val): {sorted(TEST_CITIES)}")

    out_csv = os.path.join(OUT_DIR, "aq_combined_weekly.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    stats_path = os.path.join(MODELS_DIR, "normalization_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved {stats_path} (fit on TRAIN split only)")


if __name__ == "__main__":
    main()
