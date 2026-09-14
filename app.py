"""
AirWatchAI -- interactive NO2 forecasting dashboard.

Deployed app entry point (Streamlit Community Cloud runs this file
directly). Deliberately has NO torch dependency -- inference runs through
a hand-verified NumPy re-implementation of the trained LSTM's forward
pass (see scripts/aq_step4_export_weights_numpy.py; verified against the
live torch model to 2.4e-07 max absolute difference). That keeps this app
light and fast to cold-start on free hosting.

What this shows, honestly:
  - A world map of the 12 cities in the study, colored by how well the
    model generalizes to that city's own held-out weeks -- NOT by
    pollution level, because that would misleadingly imply the model is
    always more "sure" about dirtier cities.
  - Per-city: 7+ years of actual NO2 (Sentinel-5P tropospheric column
    density), the model's stitched 4-week-ahead forecast through that
    whole history, and a live forecast for the next 4 weeks from the
    most recent available data.
  - Explicit, unhidden caveats: this is satellite column density, not
    ground-level AQI; the model has no input for sudden unprecedented
    events (wildfires, policy shocks) and will not anticipate them;
    Johannesburg is the one city where this model currently does WORSE
    than just repeating last week's value, and that's shown, not hidden.
"""

import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(APP_DIR, "outputs")
MODELS_DIR = os.path.join(APP_DIR, "models")

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
TEST_CITIES = {"Bangkok", "Tehran", "Karachi", "Johannesburg"}

SEQ_LEN, HORIZON = 12, 4
FEATURE_COLS = ["no2_log", "co_log", "aer_ai_mean", "precip_mm", "doy_sin", "doy_cos"]
NO2_COL_IDX = FEATURE_COLS.index("no2_log")
CO_COL_IDX = FEATURE_COLS.index("co_log")


# ---------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(OUT_DIR, "aq_combined_weekly.csv"), parse_dates=["period_start"])
    return df


@st.cache_data
def load_stats():
    with open(os.path.join(MODELS_DIR, "normalization_stats.json")) as f:
        return json.load(f)


@st.cache_resource
def load_weights():
    npz = np.load(os.path.join(MODELS_DIR, "no2_forecaster_weights.npz"))
    return {k: npz[k] for k in npz.files}


@st.cache_data
def load_city_skill():
    with open(os.path.join(MODELS_DIR, "city_skill_summary.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------
# NumPy LSTM forward pass -- identical equations to the trained
# PyTorch model (nn.LSTM + Linear head), verified to 2.4e-07 max
# absolute difference in scripts/aq_step4_export_weights_numpy.py.
# ---------------------------------------------------------------------
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


def normalize_df(df, stats):
    out = df.copy()
    for col in ["no2_log", "co_log", "aer_ai_mean", "precip_mm"]:
        m, s = stats[col]["mean"], stats[col]["std"]
        out[col] = (out[col] - m) / s
    return out


def forecast_window(window_raw_norm, stats, weights):
    """window_raw_norm: (SEQ_LEN, n_features) already globally-normalized.
    Applies the same window-relative centering used in training, runs the
    NumPy LSTM, de-centers and de-normalizes back to raw no2_mean units.
    Returns (HORIZON,) array of raw no2_mean predictions."""
    window = window_raw_norm.copy()
    no2_anchor = window[:, NO2_COL_IDX].mean()
    co_anchor = window[:, CO_COL_IDX].mean()
    window[:, NO2_COL_IDX] -= no2_anchor
    window[:, CO_COL_IDX] -= co_anchor

    pred_centered = numpy_lstm_forecast(window, weights)
    pred_norm = pred_centered + no2_anchor
    m, s = stats["no2_log"]["mean"], stats["no2_log"]["std"]
    pred_log = pred_norm * s + m
    return np.exp(pred_log)


@st.cache_data
def rolling_historical_forecast(city):
    """Stitched 4-week-ahead forecasts through the city's whole history --
    same method as scripts/aq_step2_train_lstm.py's plot_example(), just
    re-run here in NumPy for any city on demand."""
    df = load_data()
    stats = load_stats()
    weights = load_weights()

    g = df[df["city"] == city].sort_values("period_start").reset_index(drop=True)
    g_norm = normalize_df(g, stats)
    feats = g_norm[FEATURE_COLS].values.astype(np.float32)
    dates = g["period_start"]

    rows = []
    for start in range(0, len(g) - SEQ_LEN - HORIZON + 1, HORIZON):
        pred = forecast_window(feats[start:start + SEQ_LEN], stats, weights)
        tgt_dates = dates.iloc[start + SEQ_LEN:start + SEQ_LEN + HORIZON].tolist()
        for d, p in zip(tgt_dates, pred):
            rows.append({"date": d, "forecast_no2": p})
    return pd.DataFrame(rows)


@st.cache_data
def next_forecast(city):
    """Forecast the 4 weeks immediately after the most recent available
    data for this city -- the live, forward-looking prediction."""
    df = load_data()
    stats = load_stats()
    weights = load_weights()

    g = df[df["city"] == city].sort_values("period_start").reset_index(drop=True)
    g_norm = normalize_df(g, stats)
    feats = g_norm[FEATURE_COLS].values.astype(np.float32)
    last_window = feats[-SEQ_LEN:]
    pred = forecast_window(last_window, stats, weights)

    last_date = g["period_start"].iloc[-1]
    future_dates = [last_date + timedelta(weeks=i) for i in range(1, HORIZON + 1)]
    return pd.DataFrame({"date": future_dates, "forecast_no2": pred})


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
st.set_page_config(page_title="AirWatchAI", page_icon="\U0001F32B️", layout="wide")

df = load_data()
stats = load_stats()
city_skill = load_city_skill()

st.title("AirWatchAI \U0001F32B️")
st.caption(
    "A deep-learning NO2 forecaster trained on 8 cities and tested on 4 it never saw at all. "
    "Built on Sentinel-5P satellite data (2019-2026) -- no ground sensors required, works anywhere on Earth."
)

# --- Map ---
map_rows = []
for city, (lat, lon) in CITIES.items():
    s = city_skill.get(city, {})
    map_rows.append({
        "city": city, "lat": lat, "lon": lon,
        "role": s.get("role", "train"),
        "skill_pct": s.get("skill_pct"),
        "held_out_split": s.get("held_out_split", ""),
    })
map_df = pd.DataFrame(map_rows)


def skill_color(skill):
    if skill is None:
        return [150, 150, 150, 180]
    if skill >= 30:
        return [43, 138, 62, 200]     # strong green -- beats persistence clearly
    if skill >= 0:
        return [240, 140, 0, 200]     # amber -- marginal
    return [201, 42, 42, 200]         # red -- currently loses to persistence


map_df["color"] = map_df["skill_pct"].apply(skill_color)
map_df["radius"] = map_df["role"].apply(lambda r: 45000 if r == "test" else 32000)

col_map, col_legend = st.columns([3, 1])
with col_map:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )
    view_state = pdk.ViewState(latitude=15, longitude=20, zoom=1.1, pitch=0)
    st.pydeck_chart(pdk.Deck(
        layers=[layer], initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip={"text": "{city}\nRole: {role}\nHeld out on: {held_out_split}\nSkill vs. persistence: {skill_pct}%"},
    ))
with col_legend:
    st.markdown("**Map color = generalization skill**")
    st.markdown(
        "\U0001F7E2 Beats naive baseline by 30%+\n\n"
        "\U0001F7E0 Marginal (0-30%)\n\n"
        "\U0001F534 Currently loses to baseline\n\n"
        "Small dot = trained on this city (evaluated on its own held-out time window). "
        "Large dot = **entirely unseen** during training -- the real generalization test."
    )

st.divider()

# --- City selector ---
train_cities = sorted(c for c in CITIES if c not in TEST_CITIES)
test_cities = sorted(TEST_CITIES)
city = st.selectbox(
    "Choose a city",
    options=test_cities + train_cities,
    format_func=lambda c: f"{c}  ({'never trained on -- generalization test' if c in TEST_CITIES else 'trained on'})",
)

s = city_skill.get(city, {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Role", "Test (unseen)" if city in TEST_CITIES else "Train")
c2.metric("Skill vs. persistence", f"{s.get('skill_pct', 'n/a')}%", help="Held out on: " + s.get("held_out_split", "n/a"))
c3.metric("Relative pollution rank", f"{s.get('relative_pollution_rank_1to12', '?')} / 12",
          help="1 = cleanest of these 12 cities, 12 = most polluted. NOT a WHO/EPA AQI category -- "
               "this is satellite column density (mol/m^2), a different physical quantity from ground AQI.")
g_city = df[df["city"] == city].sort_values("period_start")
c4.metric("Data through", g_city["period_start"].max().strftime("%Y-%m-%d"))

# --- Forecast chart ---
hist_forecast = rolling_historical_forecast(city)
future = next_forecast(city)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=g_city["period_start"], y=g_city["no2_mean"] * 1e6,
    name="Actual NO2", mode="lines", line=dict(color="#2b6cb0", width=1.5),
))
fig.add_trace(go.Scatter(
    x=hist_forecast["date"], y=hist_forecast["forecast_no2"] * 1e6,
    name="Model forecast (4wk ahead, historical)", mode="lines",
    line=dict(color="#e03131", width=1.2, dash="dot"), opacity=0.75,
))
fig.add_trace(go.Scatter(
    x=[g_city["period_start"].max()] + future["date"].tolist(),
    y=[g_city["no2_mean"].iloc[-1] * 1e6] + (future["forecast_no2"] * 1e6).tolist(),
    name="Next 4 weeks (live forecast)", mode="lines+markers",
    line=dict(color="#e03131", width=3),
    marker=dict(size=7, symbol="diamond"),
))
fig.update_layout(
    title=f"{city} -- NO2 column density (tropospheric, Sentinel-5P)",
    yaxis_title="NO2 (µmol/m²)", xaxis_title=None,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=460, hovermode="x unified", margin=dict(t=70),
)
st.plotly_chart(fig, width="stretch")

low_conf = g_city["low_confidence"].sum()
st.caption(
    f"{low_conf} of {len(g_city)} weeks flagged low-confidence (>50% cloud/quality-masked pixels in the AOI) -- "
    "kept in the data and the chart, not dropped; weight this city's chart accordingly during those stretches."
)

with st.expander("What this app is honest about"):
    st.markdown(
        "- **This is satellite column density, not ground-level AQI.** NO2 tropospheric column "
        "density (mol/m²) correlates with surface pollution but is not the same physical quantity "
        "as the PM2.5/AQI numbers on a phone weather app. Treat this as a research-grade trend "
        "signal, not a substitute for a local air quality monitor.\n"
        "- **No input for sudden, unprecedented events.** The model only sees 12 weeks of its own "
        "recent trajectory -- it has no news feed, no wildfire alerts, no policy calendar. It will "
        "not anticipate a lockdown, a wildfire, or a sudden industrial shutdown; it reacts only "
        "after such an event shows up in the data. (See the project's event-evaluation notebook for "
        "exactly this check against COVID-19 and the January 2025 LA wildfires.)\n"
        "- **Johannesburg currently loses to the naive baseline** (-11% skill). Shown on the map in "
        "red, not hidden. Every other city beats simply repeating last week's value.\n"
        "- **4 cities were never touched during training at all** (Bangkok, Tehran, Karachi, "
        "Johannesburg) -- their skill numbers are a genuine test of whether the model learned "
        "transferable pollution *dynamics*, not just memorized 8 specific places."
    )

st.divider()
st.caption(
    "Data: Copernicus Sentinel-5P TROPOMI (NO2, CO, Aerosol Index) + ECMWF ERA5-Land (precipitation), "
    "via Google Earth Engine. Model: single-layer LSTM, window-relative normalized, trained on 8 cities, "
    "held out on 4. Built by Manu Chauhan Mudavath."
)
