# Opening AirWatchAI in QGIS

File: `airwatch_qgis.gpkg` (3 layers, all in EPSG:4326 / WGS84)

## 1. Open it
`Layer -> Add Layer -> Add Vector Layer` -> pick `airwatch_qgis.gpkg` -> select
all three layers (`city_aoi`, `city_summary`, `city_weekly`) -> Add.
Add an OpenStreetMap basemap first (`XYZ Tiles -> OpenStreetMap` in the
Browser panel) so the cities have geographic context.

## 2. city_aoi -- the real satellite footprint
This is the actual 1deg x 1deg box (~100km wide) that NO2/CO/aerosol/rain
were averaged over for each city -- not just a point, the real extraction
area. Style: `No brush` fill, colored outline, so the basemap shows through.
Symbology -> Categorized -> field `role` -> two colors (train vs. test) makes
it obvious at a glance which 4 cities were never touched during training.

## 3. city_summary -- where the model actually generalizes
One point per city with `skill_pct_vs_persistence`: how much better (or
worse) the model is than "just repeat last week's value," evaluated on
that city's OWN held-out weeks (val split for the 8 training cities, test
split for the 4 the model never saw at all -- same metric, same method,
computed identically for all 12, so this is a fair map).

Symbology -> Graduated -> field `skill_pct_vs_persistence` -> a
red-to-green ramp centered at 0 shows immediately: 11 of 12 cities beat
the naive baseline, Johannesburg (-11%) is the one honest weak spot.
Label points with `city` (Layer Properties -> Labels).

Note: `relative_pollution_rank_1to12` ranks these 12 cities against each
other by average NO2 column density -- it is deliberately NOT styled as a
WHO/EPA AQI color scale, because satellite column density (mol/m^2) is a
different physical quantity from ground-level AQI (ug/m^3). Presenting it
as an AQI scale would misrepresent what was actually measured; the field
is there for relative comparison only.

## 4. city_weekly -- animate 7 years of pollution across 12 cities
This is the one to demo live. 4,800 points: 400 weekly snapshots x 12
cities, 2019-01-01 through the most recent extracted week.

1. Right-click `city_weekly` -> Properties -> Temporal -> enable
   "Single Field" -> field `period_start`.
2. Symbology -> Graduated -> field `no2_mean` -> a sequential color ramp
   (e.g. YlOrRd), sized by the same field for extra effect.
3. Open the Temporal Controller (`View -> Panels -> Temporal Controller`),
   set the step to 1 week, hit play.

You'll watch NO2 breathe with the seasons across all 12 cities at once --
winter heating spikes in Beijing/Delhi/Tehran, monsoon washout in
Jakarta/Bangkok, the visible dip during COVID lockdowns in early 2020.
That's the same underlying data the LSTM was trained on, just seen
spatially instead of as a per-city line chart.
