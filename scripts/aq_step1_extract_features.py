"""
AirWatchAI -- Step 1: multi-city air quality feature extraction.

Pulls Sentinel-5P (NO2, CO, Aerosol Index) + CHIRPS rainfall, weekly,
2019-01-01 through the most recent complete week, for every city in
CITIES below -- with EXACTLY the same code path for every city. This is
a deliberate discipline (see project README): no per-city thresholds, no
per-city exclusions. If a city's data looks different, that has to be
something the downstream model learns to handle, not something this
script quietly special-cases.

Data quality, handled uniformly (not dropped):
  - Real band names were confirmed with aq_step0_inspect_bands.py before
    writing this, not assumed. NO2 exposes a "cloud_fraction" band (filtered
    at <= 0.3, standard practice for this product); CO and Aerosol Index
    expose no additional per-pixel quality band in this GEE collection --
    the L3 gridding has already discarded low-quality retrievals before
    publishing. That's a difference between PRODUCTS, not between cities --
    every city gets identical treatment within each product's own rule.
  - Every row gets both a value AND a valid_frac (fraction of the city's
    AOI with a usable retrieval that week). Low-confidence weeks are NOT
    dropped here -- they're written to the CSV with their true valid_frac
    so the next step (cleaning script) can interpolate gaps explicitly,
    with the gap-filling itself visible and auditable, rather than this
    script silently deciding what counts as "good enough" per city.

AOI: every city gets an identical 1.0 x 1.0 degree box centered on its
coordinates (roughly 90-110km wide depending on latitude) -- one uniform
rule applied to all 12 cities, not sized per-city.

Run (needs a real internet connection + `earthengine authenticate` done
locally -- this will NOT work in a network-sandboxed environment):
    pip install earthengine-api pandas
    python3 aq_step1_extract_features.py
"""

import os
import time
import datetime
import ee
import pandas as pd

EE_PROJECT = "floodmapping-506505"
ee.Initialize(project=EE_PROJECT)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# name -> (lat, lon). Split into TRAIN vs TEST is a modeling decision made
# later (step 3) -- this script treats all 12 identically.
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

AOI_HALF_WIDTH_DEG = 0.5  # every city: (lat +/- 0.5, lon +/- 0.5), no exceptions

START_DATE = "2019-01-01"
END_DATE = "2026-08-31"  # most recent fully-complete week as of extraction time

# Real band names, confirmed by running aq_step0_inspect_bands.py against
# the live collections (NOT guessed -- a first guess assuming a uniform
# "qa_value" band across all three S5P products was wrong; that band only
# exists for NO2, and even there it's actually called differently than
# expected). Ground truth:
#   NO2:    has a "cloud_fraction" band -> filter on that (standard practice
#           for this specific GEE-hosted product, which does not expose the
#           swath-level qa_value used in some other S5P access methods).
#   CO:     no cloud/quality band exposed at all in this L3 collection --
#           the L3 gridding already discards low-quality retrievals before
#           publishing, so there is nothing further to mask on.
#   AER_AI: same as CO -- no quality band exposed, use the value as-is.
# This differs by PRODUCT (a data reality), not by CITY -- every city still
# gets identical treatment within each product's rule.
PRODUCTS = {
    "no2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_NO2",
        "value_band": "tropospheric_NO2_column_number_density",
        "qa_band": "cloud_fraction",
        "qa_keep_if_lte": 0.3,  # keep pixels with cloud_fraction <= 0.3
    },
    "co": {
        "collection": "COPERNICUS/S5P/OFFL/L3_CO",
        "value_band": "CO_column_number_density",
        "qa_band": None,
    },
    "aer_ai": {
        "collection": "COPERNICUS/S5P/OFFL/L3_AER_AI",
        "value_band": "absorbing_aerosol_index",
        "qa_band": None,
    },
}

CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"


def city_aoi(lat, lon):
    return ee.Geometry.Rectangle([
        lon - AOI_HALF_WIDTH_DEG, lat - AOI_HALF_WIDTH_DEG,
        lon + AOI_HALF_WIDTH_DEG, lat + AOI_HALF_WIDTH_DEG,
    ])


def s5p_weekly_value(product_key, aoi, start, end):
    """Mean of the product's value band over its AOI for one week, plus the
    fraction of the AOI with a usable (unmasked) retrieval -- combined
    reducer so numerator and denominator share one pixel grid (same
    denominator-bug-avoidance pattern used throughout this portfolio).
    Quality masking is applied only where the product actually has a
    quality band (see PRODUCTS above); otherwise valid_frac reflects the
    data provider's own native masking of missing/invalid retrievals."""
    spec = PRODUCTS[product_key]
    coll = ee.ImageCollection(spec["collection"]).filterBounds(aoi).filterDate(start, end)

    if spec["qa_band"] is not None:
        def mask_by_qa(img):
            qa = img.select(spec["qa_band"])
            return img.updateMask(qa.lte(spec["qa_keep_if_lte"]))
        coll = coll.map(mask_by_qa)

    masked = coll.select(spec["value_band"])
    composite = masked.mean().clip(aoi).rename("VALUE")
    total_band = ee.Image.constant(1).rename("TOTAL")
    combined = composite.addBands(total_band)

    combined_reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.count(), sharedInputs=True)
    stats = combined.reduceRegion(
        reducer=combined_reducer, geometry=aoi, scale=7000, maxPixels=1e9, bestEffort=True,
    )
    valid = ee.Number(stats.get("VALUE_count"))
    total = ee.Number(stats.get("TOTAL_count"))
    valid_frac = ee.Algorithms.If(total.gt(0), valid.divide(total), 0)
    return stats.get("VALUE_mean"), valid_frac


def chirps_weekly_sum(aoi, start, end):
    coll = ee.ImageCollection(CHIRPS_COLLECTION).filterBounds(aoi).filterDate(start, end)
    total_precip = coll.sum().clip(aoi).rename("PRECIP")
    # Single (non-combined) reducer -> output key is just the band name, no
    # "_mean" suffix (that suffix only appears when reducers are .combine()'d,
    # as in s5p_weekly_value above).
    stats = total_precip.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=5000, maxPixels=1e9, bestEffort=True,
    )
    return stats.get("PRECIP")


def period_feature(period_start_str, aoi):
    period_start = ee.Date(period_start_str)
    period_end = period_start.advance(7, "day")

    no2_mean, no2_valid_frac = s5p_weekly_value("no2", aoi, period_start, period_end)
    co_mean, co_valid_frac = s5p_weekly_value("co", aoi, period_start, period_end)
    aer_mean, aer_valid_frac = s5p_weekly_value("aer_ai", aoi, period_start, period_end)
    precip = chirps_weekly_sum(aoi, period_start, period_end)

    return ee.Feature(None, {
        "period_start": period_start.format("YYYY-MM-dd"),
        "no2_mean": no2_mean,
        "no2_valid_frac": no2_valid_frac,
        "co_mean": co_mean,
        "co_valid_frac": co_valid_frac,
        "aer_ai_mean": aer_mean,
        "aer_ai_valid_frac": aer_valid_frac,
        "precip_mm": precip,
    })


def weekly_starts(start_date, end_date):
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    starts = []
    d = start
    while d <= end:
        starts.append(d.isoformat())
        d += datetime.timedelta(days=7)
    return starts


BATCH_SIZE = 3  # kept small deliberately -- each period now does 4 separate
# product pulls (NO2, CO, AER_AI, CHIRPS) instead of 1, so the same
# "too many concurrent aggregations" risk documented in the earlier NZ/India
# project applies here at roughly 4x the load per period.
MAX_RETRIES = 4


def fetch_batch_with_retry(period_list, aoi):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            fc = ee.FeatureCollection(period_list.map(lambda p: period_feature(p, aoi)))
            return fc.getInfo()["features"]
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 5 * attempt
            print(f"    batch failed ({e}), retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)


def process_city(name, lat, lon):
    print(f"\n=== {name} ({lat:.4f}, {lon:.4f}) ===")
    aoi = city_aoi(lat, lon)
    all_starts = weekly_starts(START_DATE, END_DATE)

    all_rows = []
    for i in range(0, len(all_starts), BATCH_SIZE):
        batch = all_starts[i:i + BATCH_SIZE]
        period_list = ee.List(batch)
        rows = fetch_batch_with_retry(period_list, aoi)
        for r in rows:
            props = r["properties"]
            props["city"] = name
            all_rows.append(props)
        print(f"  {batch[0]} .. {batch[-1]}: {len(rows)} weeks fetched")

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, f"{name.lower().replace(' ', '_')}_aq_timeseries.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(df)} rows)")
    return df


def main():
    for name, (lat, lon) in CITIES.items():
        process_city(name, lat, lon)
    print("\nDone. Next: aq_step1b_clean_interpolate.py to handle gaps explicitly,")
    print("then aq_step2_train_lstm.py to build and train the forecasting model.")


if __name__ == "__main__":
    main()
