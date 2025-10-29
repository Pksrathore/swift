# streamlit_pydeck_hub_spoke.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
from pathlib import Path

st.set_page_config(page_title="Hub-Spoke Map (Pydeck)", layout="wide")
st.title("📍 Hub-and-Spoke Map — pydeck")

# ----------------- Helpers -----------------
def compute_zoom_for_bounds(lat_series, lon_series):
    lat_span = lat_series.max() - lat_series.min()
    lon_span = lon_series.max() - lon_series.min()
    span = max(lat_span, lon_span)
    if span <= 0.02: return 13
    if span <= 0.05: return 12
    if span <= 0.15: return 11
    if span <= 0.5: return 10
    if span <= 1.5: return 9
    if span <= 3.0: return 8
    if span <= 7.0: return 7
    if span <= 15: return 6
    return 5

def validate_and_prepare_df(df):
    required = [
        'hub_id','spoke_id','hub_name','spoke_name','hub_city','spoke_city','mega_city',
        'hub_gmv','spoke_gmv','distance_bw_hub_spoke','hub_locality_type','spoke_locality_type',
        'hub_lat','hub_lon','spoke_lat','spoke_lon'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    # India bounding box guard (same as earlier)
    min_lat, max_lat = 6.5, 35.5
    min_lon, max_lon = 68.0, 97.5
    df = df[
        df['spoke_lat'].between(min_lat, max_lat) & df['spoke_lon'].between(min_lon, max_lon) &
        df['hub_lat'].between(min_lat, max_lat) & df['hub_lon'].between(min_lon, max_lon)
    ].copy()

    df['hub_gmv'] = pd.to_numeric(df['hub_gmv'], errors='coerce').fillna(0)
    df['spoke_gmv'] = pd.to_numeric(df['spoke_gmv'], errors='coerce').fillna(0)
    df['distance_bw_hub_spoke'] = pd.to_numeric(df['distance_bw_hub_spoke'], errors='coerce').fillna(0)
    # ensure hub_id/spoke_id are strings (so text input matches)
    df['hub_id'] = df['hub_id'].astype(str)
    df['spoke_id'] = df['spoke_id'].astype(str)
    df['hub_locality_type'] = df['hub_locality_type'].astype(str).fillna("")
    df['spoke_locality_type'] = df['spoke_locality_type'].astype(str).fillna("")
    df['mega_city'] = df['mega_city'].astype(str).fillna("Unknown")
    return df

def gmv_to_marker_size(gmv, min_size=6, max_size=25):
    gmv = np.array(gmv, dtype=float)
    if gmv.size == 0:
        return np.array([])
    if gmv.max() == gmv.min():
        return np.clip(np.ones_like(gmv) * min_size, min_size, max_size)
    return min_size + (gmv - gmv.min()) / (gmv.max() - gmv.min()) * (max_size - min_size)

# ----------------- Config / Data load -----------------
LOCAL_CSV = "hub_spoke_data.csv"
HTML_OUTFILE = "hub_spoke_map_snapshot.html"

# Sidebar: data upload / local
st.sidebar.markdown("## Data")
uploaded = st.sidebar.file_uploader("Upload hub_spoke CSV (must contain required columns)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif Path(LOCAL_CSV).exists():
    if st.sidebar.checkbox(f"Load local CSV `{LOCAL_CSV}`", value=True):
        df = pd.read_csv(LOCAL_CSV)
    else:
        df = None
else:
    df = None

if df is None:
    st.warning("No dataframe loaded yet. Upload a CSV file or place a local CSV named `hub_spoke_data.csv`.")
    st.stop()

# Validate & prepare
try:
    df = validate_and_prepare_df(df)
except Exception as e:
    st.error(f"Dataframe validation error: {e}")
    st.stop()

# ----------------- Sidebar filters -----------------
st.sidebar.markdown("## Filters / Controls")

# Tabs icons (optional UI nicety)
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Overall View",
    "🏢 Hub View",
    "📌 Spoke View",
    "🔗 Hub-Spoke Mapping"
])

# Mega city
mega_cities = ["All India"] + sorted(df['mega_city'].dropna().unique().tolist())
sel_mega = st.sidebar.selectbox("Mega city", mega_cities, index=0)

# Spoke GMV range
min_gmv = float(df['spoke_gmv'].min())
max_gmv = float(df['spoke_gmv'].max())
spoke_gmv_range = st.sidebar.slider("Spoke GMV range", min_value=min_gmv, max_value=max_gmv, value=(min_gmv, max_gmv))

# Hub name multiselect (user requested earlier)
hub_name_options = sorted(df['hub_name'].dropna().unique().tolist())
sel_hubs = st.sidebar.multiselect("Select hub(s) to filter by name (leave empty = all)", options=hub_name_options, default=[])

# Hub ID as comma-separated text (user requested)
hub_id_text = st.sidebar.text_input(
    "Filter by Hub IDs (comma-separated, leave blank = all)",
    value=""
)
# Parse and normalize
if hub_id_text.strip():
    sel_hub_ids = [x.strip() for x in hub_id_text.split(",") if x.strip()]
else:
    sel_hub_ids = []

# Option: show hub labels toggle
show_hubs = st.sidebar.checkbox("Show hub labels (text on map)", value=False)

# Map style & export
map_style = st.sidebar.selectbox("Map style", ["mapbox/light-v10", "mapbox/dark-v10", "road", "satellite"], index=0)
export_html = st.sidebar.checkbox("Save HTML snapshot on render", value=False)
if st.sidebar.button("Save snapshot HTML (now)"):
    export_html = True

# ----------------- Apply filters -----------------
dsub = df.copy()

if sel_mega != "All India":
    dsub = dsub[dsub['mega_city'] == sel_mega]

dsub = dsub[(dsub['spoke_gmv'] >= spoke_gmv_range[0]) & (dsub['spoke_gmv'] <= spoke_gmv_range[1])]

if sel_hubs:
    dsub = dsub[dsub['hub_name'].isin(sel_hubs)]

if sel_hub_ids:
    # hub_id column is string, so compare strings
    sel_hub_ids_str = [str(x) for x in sel_hub_ids]
    # optionally warn about bad ids
    invalid_ids = [hid for hid in sel_hub_ids_str if hid not in dsub['hub_id'].unique() and hid not in df['hub_id'].unique()]
    if invalid_ids:
        st.sidebar.warning(f"Hub ID(s) not found in data (will be ignored): {', '.join(invalid_ids)}")
    dsub = dsub[dsub['hub_id'].isin(sel_hub_ids_str)]

if dsub.empty:
    st.write("No data for selected filters.")
    st.stop()

# ----------------- Colors & sizes -----------------
HUB_COLOR = [220, 50, 50]     # red-ish for hubs
SPOKE_COLOR = [40, 140, 220]  # blue-ish for spokes
LINE_ALPHA = 200

spoke_sizes = gmv_to_marker_size(dsub['spoke_gmv'], min_size=6, max_size=20)
hubs_unique = dsub.drop_duplicates(subset=['hub_id']).reset_index(drop=True)
hub_sizes = gmv_to_marker_size(hubs_unique['hub_gmv'], min_size=12, max_size=42)

# ----------------- Prepare pydeck data -----------------
# spokes_data includes hub pivot info so tooltip can show hub details
spokes_data = dsub.assign(
    _position = dsub.apply(lambda r: [r['spoke_lon'], r['spoke_lat']], axis=1),
    _size = spoke_sizes,
    hub_id = dsub['hub_id'],
    hub_name = dsub['hub_name'],
    hub_lat = dsub['hub_lat'],
    hub_lon = dsub['hub_lon'],
    spoke_gmv = dsub['spoke_gmv'],
    spoke_name = dsub['spoke_name'],
    distance_km = dsub['distance_bw_hub_spoke'],
    hub_locality_type = dsub['hub_locality_type'],
    spoke_locality_type = dsub['spoke_locality_type']
).to_dict(orient='records')

hubs_data = hubs_unique.assign(
    _position = hubs_unique.apply(lambda r: [r['hub_lon'], r['hub_lat']], axis=1),
    _size = hub_sizes,
    hub_gmv = hubs_unique['hub_gmv'],
    hub_name = hubs_unique['hub_name'],
    hub_id = hubs_unique['hub_id'],
    hub_locality_type = hubs_unique['hub_locality_type']
).to_dict(orient='records')

# lines: each spoke -> its hub, colored by HUB_COLOR (could be extended per-hub)
lines = []
for _, r in dsub.iterrows():
    # color field uses RGBA
    color = HUB_COLOR + [LINE_ALPHA]
    lines.append({
        "source": [r['hub_lon'], r['hub_lat']],
        "target": [r['spoke_lon'], r['spoke_lat']],
        "hub_id": str(r['hub_id']),
        "hub_name": r['hub_name'],
        "spoke_id": str(r['spoke_id']),
        "spoke_name": r['spoke_name'],
        "spoke_gmv": float(r['spoke_gmv']),
        "distance_km": float(r['distance_bw_hub_spoke']),
        "color": color
    })

# ----------------- View state -----------------
lat_vals = pd.concat([dsub['hub_lat'], dsub['spoke_lat']])
lon_vals = pd.concat([dsub['hub_lon'], dsub['spoke_lon']])
center_lat = float(lat_vals.mean())
center_lon = float(lon_vals.mean())
zoom = compute_zoom_for_bounds(lat_vals, lon_vals)

view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=40)

# ----------------- Mapbox token -----------------
MAPBOX_TOKEN = os.getenv("MAPBOX_API_KEY", None)
if MAPBOX_TOKEN:
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN
else:
    st.sidebar.info("Set MAPBOX_API_KEY env var to enable Mapbox styles (optional).")

# ----------------- Layers -----------------
layers = []

# Line layer: hub -> spoke, pickable for hover
line_layer = pdk.Layer(
    "LineLayer",
    data=lines,
    get_source_position="source",
    get_target_position="target",
    get_color="color",
    get_width=2,
    pickable=True,
    auto_highlight=True
)
layers.append(line_layer)

# Spoke layer
spoke_layer = pdk.Layer(
    "ScatterplotLayer",
    data=spokes_data,
    get_position="_position",
    get_radius="_size",
    radius_units="pixels",
    get_fill_color=SPOKE_COLOR,
    get_line_color=[0, 0, 0],
    pickable=True,
    opacity=0.9,
    stroked=True,
    auto_highlight=True
)
layers.append(spoke_layer)

# Hub layer
hub_layer = pdk.Layer(
    "ScatterplotLayer",
    data=hubs_data,
    get_position="_position",
    get_radius="_size",
    radius_units="pixels",
    get_fill_color=HUB_COLOR,
    get_line_color=[0, 0, 0],
    pickable=True,
    opacity=0.95,
    stroked=True,
    auto_highlight=True
)
layers.append(hub_layer)

# Optional text labels when show_hubs True (user toggle)
if show_hubs:
    text_data = hubs_unique.assign(
        _position = hubs_unique.apply(lambda r: [r['hub_lon'], r['hub_lat']], axis=1),
        label = hubs_unique['hub_name']
    ).to_dict(orient='records')

    text_layer = pdk.Layer(
        "TextLayer",
        data=text_data,
        get_position="_position",
        get_text="label",
        get_size=14,
        get_color=[0, 0, 0],
        get_angle=0,
        get_text_anchor="'start'",
        get_alignment_baseline="'center'",
        pickable=False
    )
    layers.append(text_layer)

# ----------------- Tooltip (hover-only labels/details) -----------------
# The HTML references keys that exist on the hovered object's data.
tooltip = {
    "html": (
        "<div style='font-size:13px;line-height:1.25'>"
        "<b>Layer:</b> {__deck_object_type__} <br/>"
        "<b>Hub:</b> {hub_name} (ID: {hub_id})<br/>"
        "<b>Spoke:</b> {spoke_name} (ID: {spoke_id})<br/>"
        "<b>Hub GMV:</b> {hub_gmv} <br/>"
        "<b>Spoke GMV:</b> {spoke_gmv} <br/>"
        "<b>Distance (km):</b> {distance_km} <br/>"
        "<b>Hub locality:</b> {hub_locality_type} <br/>"
        "<b>Spoke locality:</b> {spoke_locality_type}"
        "</div>"
    ),
    "style": {"backgroundColor": "rgba(30,30,30,0.95)", "color": "white", "padding": "8px", "borderRadius": "6px"}
}

# ----------------- Deck & Render -----------------
deck = pdk.Deck(
    initial_view_state=view_state,
    layers=layers,
    tooltip=tooltip,
    map_style=map_style if (MAPBOX_TOKEN or map_style.startswith("mapbox/")) else "road"
)

# Place map in first tab (Overall View) for a clean UI; you can move if desired
with tab1:
    st.subheader("Overall View")
    st.pydeck_chart(deck, use_container_width=True)

# Simple per-tab placeholders for now — user can extend individually
with tab2:
    st.subheader("Hub View")
    st.write("Use the filters on the left to focus on one or more hubs. (Hub-only summary plots can be added here.)")
with tab3:
    st.subheader("Spoke View")
    st.write("Spoke-level charts or tables can go here.")
with tab4:
    st.subheader("Hub-Spoke Mapping")
    st.write("Detailed hub↔spoke mapping controls and exports can be added here.")

# ----------------- Export HTML snapshot -----------------
if export_html:
    try:
        deck.to_html(HTML_OUTFILE, notebook_display=False, iframe_height=700)
        st.sidebar.success(f"Saved snapshot: {HTML_OUTFILE}")
        st.sidebar.markdown(f"[Open snapshot](./{HTML_OUTFILE})")
    except Exception as e:
        st.sidebar.error(f"Failed to save HTML: {e}")

# ----------------- Data table for inspection -----------------
with st.expander("Show data table (first 200 rows)"):
    st.dataframe(dsub.head(200))

st.caption(
    "Hover over hubs, spokes, or connecting lines to see labels and details. "
    "Use Mega city, GMV range, Hub name multiselect, and Hub ID comma-separated input to filter."
)
