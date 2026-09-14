"""
AirWatchAI -- Step 1c: patch precip_mm in the existing 12 city CSVs using
ERA5-Land instead of CHIRPS, without re-fetching NO2/CO/Aerosol Index
(those were already good -- no need to redo ~40 minutes of work for a
rainfall-source swap).

Run:
    python3 aq_step1c_refetch_precip.py
"""

import os
import glob
import ee
import pandas as pd

EE_PROJECT = "floodmapping-506505"
ee.Initialize(project=EE_PROJECT)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

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
ERA5_LAND_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
PRECIP_BAND = "total_precipitation_sum"


def city_aoi(lat, lon):
    return ee.Geometry.Rectangle([
        lon - AOI_HALF_WIDTH_DEG, lat - AOI_HALF_WIDTH_DEG,
        lon + AOI_HALF_WIDTH_DEG, lat + AOI_HALF_WIDTH_DEG,
    ])


def era5_weekly_precip_mm_list(aoi, period_starts):
    def one_week(period_start_str):
        period_start = ee.Date(period_start_str)
        period_end = period_start.advance(7, "day")
        coll = ee.ImageCollection(ERA5_LAND_COLLECTION).filterBounds(aoi).filterDate(period_start, period_end)
        daily_precip_m = coll.select(PRECIP_BAND).map(lambda img: img.max(0))
        total_precip_mm = daily_precip_m.sum().clip(aoi).multiply(1000).rename("PRECIP")
        stats = total_precip_mm.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=11132, maxPixels=1e9, bestEffort=True,
        )
        return ee.Feature(None, {"period_start": period_start.format("YYYY-MM-dd"), "precip_mm": stats.get("PRECIP")})

    BATCH = 8
    all_rows = []
    for i in range(0, len(period_starts), BATCH):
        batch = period_starts[i:i + BATCH]
        fc = ee.FeatureCollection(ee.List(batch).map(one_week))
        for r in fc.getInfo()["features"]:
            all_rows.append(r["properties"])
    return pd.DataFrame(all_rows)


def main():
    for name, (lat, lon) in CITIES.items():
        path = os.path.join(OUT_DIR, f"{name.lower().replace(' ', '_')}_aq_timeseries.csv")
        df = pd.read_csv(path)
        aoi = city_aoi(lat, lon)
        print(f"=== {name} ===")
        new_precip = era5_weekly_precip_mm_list(aoi, df["period_start"].tolist())
        merged = df.drop(columns=["precip_mm"]).merge(new_precip, on="period_start", how="left")
        n_null = merged["precip_mm"].isna().sum()
        merged.to_csv(path, index=False)
        print(f"  Patched {path} -- precip_mm nulls now: {n_null}")

    print("\nDone. Re-run aq_step1b_consolidate_features.py next.")


if __name__ == "__main__":
    main()
