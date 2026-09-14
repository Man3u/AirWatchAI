"""
AirWatchAI -- Step 0: print the real band names for every collection we
plan to use, before writing any masking logic against guessed band names.
Cheap, fast (metadata only, no heavy computation) -- run this whenever a
"Band pattern ... did not match" error shows up rather than guessing again.
"""

import ee

ee.Initialize(project="floodmapping-506505")

COLLECTIONS = [
    "ECMWF/ERA5/DAILY",
    "ECMWF/ERA5_LAND/DAILY_AGGR",
]

# Also check whether each collection actually has recent (2026) data --
# some ERA5 products lag behind by months or were discontinued in favor of
# a newer product, and we need coverage through mid-2026.
for coll_id in COLLECTIONS:
    print(f"{coll_id}:")
    img = ee.ImageCollection(coll_id).filterDate("2023-01-01", "2023-01-15").first()
    bands = img.bandNames().getInfo()
    print(f"  bands: {bands}")

    recent = ee.ImageCollection(coll_id).filterDate("2026-06-01", "2026-08-01")
    recent_count = recent.size().getInfo()
    print(f"  images available 2026-06-01 to 2026-08-01: {recent_count}\n")
