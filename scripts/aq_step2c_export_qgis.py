"""
AirWatchAI -- Step 2c: export a QGIS-ready GeoPackage.

This project's other outputs (metrics, plots) are non-spatial in the
sense that they don't show WHERE the 12 cities sit relative to each
other, what patch of Earth each city's AOI actually covers, or how NO2
evolved week-by-week across the whole 7-year record in one animated
view. QGIS is the right tool for that -- same pattern that worked
cleanly in the earlier NZ/India project (unlike the kepler.gl/deck.gl
path, which never rendered).

Pure local computation -- no Earth Engine needed, runs entirely on
already-downloaded CSVs + the trained model.

Produces one GeoPackage (outputs/airwatch_qgis.gpkg) with three layers:

  1. city_aoi       -- the actual 1x1 degree extraction box per city
                        (polygon). Shows the real spatial footprint the
                        satellite data was averaged over, not just a
                        point on a map.
  2. city_summary    -- one point per city (static attributes): role
                        (train/test), the model's skill vs. persistence
                        on that city's OWN held-out weeks (val split for
                        train cities, test split for test cities -- same
                        metric, computed identically for all 12, so a
                        map reader can see at a glance where the model
                        generalizes well and where it doesn't), and a
                        relative pollution rank among these 12 cities
                        (NOT a WHO/EPA AQI category -- this is satellite
                        column density, a different physical quantity
                        from ground-level AQI; ranking cities against
                        each other is honest, converting to an AQI
                        color scale would not be).
  3. city_weekly     -- one point per city PER WEEK (2019-01-01 through
                        the latest extracted week), with a proper date
                        field (period_start) so QGIS's Temporal
                        Controller can animate NO2/CO/aerosol index
                        evolving across all 12 cities over time.

Run (pure Python, no internet needed):
    pip install geopandas shapely pyogrio pandas torch
    python3 aq_step2c_export_qgis.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Must match aq_step1_extract_features.py exactly -- same cities, same AOI.
CITIES = {
    "Delhi":        (28.6139, 77.2090),
    "Beijing":      (39.9042, 116.4074),
    "Los Angeles":  (34.0522, -118.2437),
    "Mexico City":  (19.4326, -99.1332),
    "Cairo":        (30.0444, 31.2357),
    "Lagos":        (6.5244, 3.3792),
    "Jakarta":      (-6.2088, 106.8456),
    "London":       (51.5074, -0.1278),
    "Bangkok":      (13.7563, 100.5018),
    "Tehran":       (35.6892, 51.3890),
    "Karachi":      (24.8607, 67.0011),
    "Johannesburg": (-26.2041, 28.0473),
}
AOI_HALF_WIDTH_DEG = 0.5

SEQ_LEN, HORIZON = 12, 4
FEATURE_COLS = ["no2_log", "co_log", "aer_ai_mean", "precip_mm", "doy_sin", "doy_cos"]
TARGET_COL = "no2_log"
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


def load_data_and_model():
    df = pd.read_csv(os.path.join(OUT_DIR, "aq_combined_weekly.csv"), parse_dates=["period_start"])
    with open(os.path.join(MODELS_DIR, "normalization_stats.json")) as f:
        stats = json.load(f)
    model = NO2Forecaster(n_features=len(FEATURE_COLS))
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "no2_forecaster.pt")))
    model.eval()
    return df, stats, model


def normalize(df, stats):
    df = df.copy()
    for col in ["no2_log", "co_log", "aer_ai_mean", "precip_mm"]:
        m, s = stats[col]["mean"], stats[col]["std"]
        df[col] = (df[col] - m) / s
    return df


def per_city_skill(df_norm, model):
    """Same window-relative-centering methodology as training (must match
    exactly, or these numbers won't mean what they claim to). For every
    city, evaluate on its OWN held-out split -- val for train cities, test
    for test cities -- never on rows the model's weights were fit against."""
    loss_fn = nn.MSELoss()
    results = {}
    for city, g in df_norm.groupby("city"):
        g = g.sort_values("period_start").reset_index(drop=True)
        feats = g[FEATURE_COLS].values
        target = g[TARGET_COL].values
        splits = g["split"].values
        X, Y = [], []
        for start in range(0, len(g) - SEQ_LEN - HORIZON + 1):
            tgt_slice = slice(start + SEQ_LEN, start + SEQ_LEN + HORIZON)
            tgt_splits = np.unique(splits[tgt_slice])
            if len(tgt_splits) != 1 or tgt_splits[0] == "train":
                continue  # only evaluate on this city's OWN held-out weeks
            window = feats[start:start + SEQ_LEN].copy()
            no2_anchor = window[:, NO2_COL_IDX].mean()
            co_anchor = window[:, CO_COL_IDX].mean()
            window[:, NO2_COL_IDX] -= no2_anchor
            window[:, CO_COL_IDX] -= co_anchor
            X.append(window)
            Y.append(target[tgt_slice] - no2_anchor)
        if not X:
            results[city] = {"skill_pct": None, "n_eval_windows": 0, "held_out_split": None}
            continue
        Xt = torch.tensor(np.array(X).tolist(), dtype=torch.float32)
        Yt = torch.tensor(np.array(Y).tolist(), dtype=torch.float32)
        with torch.no_grad():
            pred = model(Xt)
            model_mse = loss_fn(pred, Yt).item()
        last_val = Xt[:, -1, NO2_COL_IDX]
        persist_pred = last_val.unsqueeze(1).repeat(1, Yt.shape[1])
        persist_mse = loss_fn(persist_pred, Yt).item()
        skill = (persist_mse - model_mse) / persist_mse * 100
        held_out = "val" if city not in TEST_CITIES else "test"
        results[city] = {
            "skill_pct": round(skill, 1), "model_mse": round(model_mse, 4),
            "persistence_mse": round(persist_mse, 4), "n_eval_windows": len(X),
            "held_out_split": held_out,
        }
    return results


TEST_CITIES = {"Bangkok", "Tehran", "Karachi", "Johannesburg"}


def build_city_aoi_layer():
    rows = []
    for city, (lat, lon) in CITIES.items():
        geom = box(lon - AOI_HALF_WIDTH_DEG, lat - AOI_HALF_WIDTH_DEG,
                   lon + AOI_HALF_WIDTH_DEG, lat + AOI_HALF_WIDTH_DEG)
        rows.append({"city": city, "role": "test" if city in TEST_CITIES else "train", "geometry": geom})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def build_city_summary_layer(df, skills):
    # Relative pollution rank among these 12 cities -- explicitly NOT a
    # WHO/EPA AQI category. Column density (mol/m^2) is a different
    # physical quantity from ground-level ug/m^3 AQI; presenting this as
    # an AQI color scale would misrepresent what was actually measured.
    city_means = df.groupby("city")["no2_mean"].mean().sort_values()
    rank = {city: i + 1 for i, city in enumerate(city_means.index)}  # 1 = cleanest of the 12

    rows = []
    for city, (lat, lon) in CITIES.items():
        s = skills.get(city, {})
        rows.append({
            "city": city,
            "role": "test (never trained on)" if city in TEST_CITIES else "train",
            "skill_pct_vs_persistence": s.get("skill_pct"),
            "held_out_split_used": s.get("held_out_split"),
            "n_eval_windows": s.get("n_eval_windows", 0),
            "mean_no2_column_density": round(float(city_means[city]), 8),
            "relative_pollution_rank_1to12": rank[city],
            "rank_note": "1 = cleanest of these 12 monitored cities, 12 = most polluted -- NOT a WHO/EPA AQI category",
            "geometry": Point(lon, lat),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def build_city_weekly_layer(df):
    rows = []
    for city, (lat, lon) in CITIES.items():
        g = df[df["city"] == city]
        for _, r in g.iterrows():
            rows.append({
                "city": city,
                "role": "test" if city in TEST_CITIES else "train",
                "period_start": r["period_start"],
                "no2_mean": r["no2_mean"],
                "co_mean": r["co_mean"],
                "aer_ai_mean": r["aer_ai_mean"],
                "precip_mm": r["precip_mm"],
                "low_confidence": bool(r["low_confidence"]),
                "geometry": Point(lon, lat),
            })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def main():
    df, stats, model = load_data_and_model()
    df_norm = normalize(df, stats)

    print("Evaluating per-city skill on each city's OWN held-out split...")
    skills = per_city_skill(df_norm, model)
    for city, s in skills.items():
        print(f"  {city:15s} split={s.get('held_out_split')!s:6}  skill={s.get('skill_pct')}")

    aoi_layer = build_city_aoi_layer()
    summary_layer = build_city_summary_layer(df, skills)
    weekly_layer = build_city_weekly_layer(df)

    out_path = os.path.join(OUT_DIR, "airwatch_qgis.gpkg")
    if os.path.exists(out_path):
        os.remove(out_path)
    aoi_layer.to_file(out_path, layer="city_aoi", driver="GPKG")
    summary_layer.to_file(out_path, layer="city_summary", driver="GPKG")
    weekly_layer.to_file(out_path, layer="city_weekly", driver="GPKG")

    print(f"\nSaved {out_path}")
    print(f"  city_aoi:     {len(aoi_layer)} polygons (1x1 deg extraction box per city)")
    print(f"  city_summary: {len(summary_layer)} points (per-city model performance)")
    print(f"  city_weekly:  {len(weekly_layer)} points ({weekly_layer['period_start'].nunique()} weeks x 12 cities, for Temporal Controller animation)")


if __name__ == "__main__":
    main()
