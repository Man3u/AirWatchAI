"""
AirWatchAI -- Step 0: print the real band names for every collection we
plan to use, before writing any masking logic against guessed band names.
Cheap, fast (metadata only, no heavy computation) -- run this whenever a
"Band pattern ... did not match" error shows up rather than guessing again.
"""

import ee

ee.Initialize(project="floodmapping-506505")

COLLECTIONS = [
    "COPERNICUS/S5P/OFFL/L3_NO2",
    "COPERNICUS/S5P/OFFL/L3_CO",
    "COPERNICUS/S5P/OFFL/L3_AER_AI",
    "UCSB-CHG/CHIRPS/DAILY",
]

for coll_id in COLLECTIONS:
    img = ee.ImageCollection(coll_id).filterDate("2023-01-01", "2023-01-15").first()
    bands = img.bandNames().getInfo()
    print(f"{coll_id}:")
    print(f"  {bands}\n")
