# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from io import BytesIO

st.set_page_config(page_title="Locality Mapping — Hub-Spoke", layout="wide")
st.title("📍 Locality Mapping — Hub & Spoke")

# ---------- CONFIG ----------
LOCAL_CSV = r"C:\Users\Admin\Desktop\Excel Files\Locality_mapping\New_cluster_loc_mapping_lat_lon.csv"
HTML_OUTFILE = "hub_spoke_map_snapshot.html"

# ---------- helper functions ----------
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


def parse_hub_ids(text):
    """Parse comma/semicolon/space separated hub ids into list of strings or ints"""
    if not text:
        return []
    parts = [p.strip() for p in text.replace(';', ',').split(',') if p.strip()]
    parsed = []
    for p in parts:
        # keep as original string if non-numeric
        try:
            parsed.append(int(p))
        except Exception:
            parsed.append(p)
    return parsed


def validate_and_prepare_df(df):
 
    df = df.copy()
    minimal_required = [
        'spoke_id','spoke_name','hub_id','hub_name','spoke_city','hub_city','mega_city',
        'distance_bw_hub_spoke','hub_lat','hub_lon','spoke_lat','spoke_lon'
    ]

    # Full expected schema (for downstream features)
    full_expected = [
        'spoke_id','spoke_name','hub_id','hub_name','spoke_city','hub_city','mega_city',
        'hub_gmv','spoke_gmv','distance_bw_hub_spoke','hub_locality_type','spoke_locality_type',
        'hub_lat','hub_lon','spoke_lat','spoke_lon','loc_type','loc_classification',
        'old_cluster_loc_id','old_loc_bucket'
    ]

    missing_minimal = [c for c in minimal_required if c not in df.columns]
    missing_full = [c for c in full_expected if c not in df.columns]

    # If any minimal columns are missing — we cannot plot. Return error via exception
    if missing_minimal:
        raise ValueError(f"Cannot plot — missing minimal required columns: {missing_minimal}")

    # For any full expected columns that were missing, create them with defaults so the rest of the app can run
    filled_cols = []
    for c in missing_full:
        filled_cols.append(c)
        if c in ['hub_gmv','spoke_gmv','distance_bw_hub_spoke']:
            df[c] = 0
        elif c in ['hub_lat','hub_lon','spoke_lat','spoke_lon']:
            # these shouldn't be missing because of minimal check; just in case
            df[c] = np.nan
        else:
            df[c] = ""

    # Coerce numeric types safely
    df['hub_gmv'] = pd.to_numeric(df.get('hub_gmv', 0), errors='coerce').fillna(0)
    df['spoke_gmv'] = pd.to_numeric(df.get('spoke_gmv', 0), errors='coerce').fillna(0)
    df['distance_bw_hub_spoke'] = pd.to_numeric(df.get('distance_bw_hub_spoke', 0), errors='coerce').fillna(0)

    # Keep hub_id/spoke_id as strings for flexible filtering
    df['hub_id'] = df['hub_id'].astype(str)
    df['spoke_id'] = df['spoke_id'].astype(str)

    # Fill/normalize string columns
    for s in ['hub_locality_type','spoke_locality_type','mega_city','loc_type','loc_classification','old_loc_bucket']:
        if s in df.columns:
            df[s] = df[s].astype(str).fillna("")
        else:
            df[s] = ""

    # Ensure lat/lon numeric and drop rows with invalid coordinates (can't plot these)
    for coord in ['hub_lat','hub_lon','spoke_lat','spoke_lon']:
        df[coord] = pd.to_numeric(df[coord], errors='coerce')

    # Keep rows where valid lat–lon pairs exist
    has_valid_hub_coords = df['hub_lat'].notna() & df['hub_lon'].notna()
    has_valid_spoke_coords = df['spoke_lat'].notna() & df['spoke_lon'].notna()

    # Keep rows where either hub pair or spoke pair (or both) exist
    df = df[has_valid_hub_coords | has_valid_spoke_coords].copy()


    return df, filled_cols


def gmv_to_marker_size(gmv, min_size=6, max_size=25):
    gmv = np.array(gmv)
    if gmv.size == 0:
        return np.array([])
    if gmv.max() == gmv.min():
        return np.clip(np.ones_like(gmv) * min_size, min_size, max_size)
    return min_size + (gmv - gmv.min()) / (gmv.max() - gmv.min()) * (max_size - min_size)


# ---------- data source selection ----------
st.sidebar.header("Data source")
source = st.sidebar.radio("Load data from:", ("Upload file (csv / xlsx)", "BigQuery", "Local CSV on server"))

# We'll keep the loaded dataframe in session state under 'df' for robust clearing/reset behavior
if 'df' not in st.session_state:
    st.session_state['df'] = None

df = None

# ---------- BQ caching helpers ----------
import hashlib
import pickle
from pathlib import Path
import time
CACHE_DIR = Path("bq_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY_SECONDS = 3600  # 1 hour expiry


def _cache_path_for_query(query: str, project_id: str):
    key = (query or "") + "__" + (project_id or "")
    h = hashlib.md5(key.encode('utf-8')).hexdigest()
    return CACHE_DIR / f"bq_{h}.pkl"


def _is_cache_fresh(path: Path):
    try:
        mtime = path.stat().st_mtime
        age = time.time() - mtime
        return age <= CACHE_EXPIRY_SECONDS
    except Exception:
        return False


def list_cache_files():
    return sorted([p for p in CACHE_DIR.glob("bq_*.pkl")], key=lambda p: p.stat().st_mtime, reverse=True)


def load_bq_cache(query: str, project_id: str):
    path = _cache_path_for_query(query, project_id)
    if path.exists():
        # check expiry
        if not _is_cache_fresh(path):
            try:
                st.sidebar.info(f"Found cached results but cache is older than 1 hour (expired).")
            except Exception:
                pass
            return None
        try:
            return pd.read_pickle(path)
        except Exception:
            return None
    return None


def load_bq_cache_from_path(path: Path):
    try:
        if not path.exists():
            return None
        if not _is_cache_fresh(path):
            st.info(f"Selected cache '{path.name}' is expired (older than 1 hour). Consider refreshing from BigQuery.")
        return pd.read_pickle(path)
    except Exception as e:
        st.error(f"Failed to load cache file: {e}")
        return None


def save_bq_cache(df_in: pd.DataFrame, query: str, project_id: str):
    path = _cache_path_for_query(query, project_id)
    try:
        df_in.to_pickle(path)
        return True
    except Exception:
        return False


def delete_bq_cache(query: str = None, project_id: str = None):
    """Delete one cache (if query provided) or all caches when no query provided."""
    if query:
        path = _cache_path_for_query(query, project_id)
        if path.exists():
            try:
                path.unlink()
                return True
            except Exception:
                return False
        return False
    else:
        # delete all
        success = True
        for p in CACHE_DIR.glob("bq_*.pkl"):
            try:
                p.unlink()
            except Exception:
                success = False
        return success


# ---------- CLEAR FILTERS BUTTON ----------
if st.sidebar.button("🧹Reset Data" ,key="clear_filters_btn"):
    filter_keys = [
        'hub_ids_text',
        'sel_hub_name',
        'sel_mega',
        'sel_spoke',
        'sel_loc_type',
        'sel_loc_class',
        'sel_old_loc_bucket',
        'only_independent',
        'show_hubs',
        'map_style',
        'spoke_gmv_range',
    ]

    for key in filter_keys:
        if key in st.session_state:
            del st.session_state[key]

    # Also remove the loaded dataframe and any cache indicators so the UI truly resets
    for k in ['df', 'loaded_cache_name', 'selected_cache_idx', 'selected_cache_path']:
        if k in st.session_state:
            del st.session_state[k]

    st.sidebar.success("✅ All filters have been cleared. Please reapply filters if needed.")
    st.rerun()


if source.startswith("Upload"):
    uploaded = st.sidebar.file_uploader("Upload file (csv or xlsx). Columns must match the expected schema.", type=["csv", "xlsx"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
                st.session_state['df'] = df
            else:
                df = pd.read_excel(uploaded)
            # store into session state for persistence
            st.session_state['df'] = df
            if 'loaded_cache_name' in st.session_state:
                del st.session_state['loaded_cache_name']
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded file: {e}")

elif source == "BigQuery":
    st.sidebar.subheader("BigQuery options")
    project_id = st.sidebar.text_input("GCP Project ID (optional)")
    bq_query = st.sidebar.text_area("BigQuery SQL upload", value='''---Enter Your Query---''', height=200)

    st.sidebar.markdown("---")
    st.sidebar.caption("Caching behaviour: query results are cached locally (in `bq_cache/`) to avoid re-running heavy queries repeatedly. Cache expires after 1 hour.")
    force_refresh = st.sidebar.checkbox("Force BQ refresh (ignore cache)", value=False)
    run_bq_btn = st.sidebar.button("Load / Run BigQuery")
    clear_cache_btn = st.sidebar.button("Clear all BQ cache")

    if clear_cache_btn:
        ok = delete_bq_cache()
        if ok:
            st.sidebar.success("Cleared all BQ cache files.")
        else:
            st.sidebar.warning("Tried to clear cache but some files may remain.")

    # --- MAIN DASHBOARD: cached-load button & selector (duplicate of sidebar) ---
    st.markdown("### Quick cache loader")
    st.caption("Load a previously cached BQ result from local cache files.")
    cache_files = list_cache_files()
    if cache_files:
        cache_names = [f"{p.name} — {int((time.time()-p.stat().st_mtime)//60)}m" for p in cache_files]
        selected_cache_idx = st.selectbox("Select cache file to load (main)", options=list(range(len(cache_files))), format_func=lambda i: cache_names[i])
        if st.button("Load selected cache (main)"):
            path = cache_files[selected_cache_idx]
            cached_df = load_bq_cache_from_path(path)
            if cached_df is not None:
                st.success(f"Loaded cached file: {path.name} ({len(cached_df):,} rows)")
                st.session_state['df'] = cached_df
                st.session_state['loaded_cache_name'] = path.name
                # remember which cache index selected so Clear All can reset it
                st.session_state['selected_cache_idx'] = selected_cache_idx
                st.session_state['selected_cache_path'] = str(path)
    else:
        st.info("No cached BigQuery files found in bq_cache/. Run and cache a BigQuery query first (sidebar).")

    if run_bq_btn:
        if not bq_query or bq_query.strip() == "" or 'Enter Your Query' in bq_query:
            st.sidebar.warning("Please paste your BigQuery SQL into the textbox before running.")
        else:
            # Attempt to load from cache first (unless forced)
            cached_df = None
            cache_path = _cache_path_for_query(bq_query, project_id)
            cache_exists = cache_path.exists()
            cache_fresh = cache_exists and _is_cache_fresh(cache_path)

            if cache_exists:
                # show cache age
                try:
                    age = time.time() - cache_path.stat().st_mtime
                    age_minutes = int(age // 60)
                    st.sidebar.caption(f"Cache file: {cache_path.name} (age: {age_minutes} minutes)")
                except Exception:
                    pass

            if not force_refresh and cache_fresh:
                cached_df = load_bq_cache(bq_query, project_id)
                if cached_df is not None:
                    st.sidebar.success(f"Loaded cached BigQuery results ({len(cached_df):,} rows).")
                    st.session_state['df'] = cached_df
                    st.session_state['loaded_cache_name'] = cache_path.name

            # If cache wasn't used, run query once and cache it
            if st.session_state.get('df') is None:
                with st.spinner("Running your query to fetch data ..."):
                    try:
                        from pandas_gbq import read_gbq
                        kwargs = {}
                        if project_id:
                            kwargs['project_id'] = project_id
                        df_bq = read_gbq(bq_query, **kwargs)
                        saved = save_bq_cache(df_bq, bq_query, project_id)
                        if saved:
                            st.sidebar.success(f"BigQuery finished and cached ({len(df_bq):,} rows).")
                        else:
                            st.sidebar.warning("BigQuery finished but caching failed.")
                        st.session_state['df'] = df_bq
                        st.session_state['loaded_cache_name'] = _cache_path_for_query(bq_query, project_id).name
                    except Exception as e:
                        st.sidebar.error(f"BigQuery read failed: {e}")

else:
    if os.path.exists(LOCAL_CSV):
        if st.sidebar.checkbox(f"Load local CSV `{LOCAL_CSV}`", value=True):
            try:
                df_local = pd.read_csv(LOCAL_CSV)
                st.session_state['df'] = df_local
                if 'loaded_cache_name' in st.session_state:
                    del st.session_state['loaded_cache_name']
            except Exception as e:
                st.sidebar.error(f"Failed to load local CSV: {e}")

# If df is still None, show message
if st.session_state.get('df') is None:
    st.info("No data loaded yet. Please choose a data source and load data (or load cached BQ results).")
    st.stop()

# use the dataframe from session state
df = st.session_state.get('df')

# ---------- validate & prepare ----------
if df is None:
    st.info("No data loaded yet. Please choose a data source and load data (or load cached BQ results).")
    st.stop()

# ---------- validate & prepare ----------
# try:
#     df, filled_cols = validate_and_prepare_df(df)
#     if filled_cols:
#         st.warning(f"The following expected columns were missing in the provided data and were filled with defaults: {filled_cols}")
# except Exception as e:
#     st.error(f"Data validation error: {e}")
#     st.stop()

# ---------- CLEAR FILTERS BUTTON ----------
if st.sidebar.button("🧹 Clear All Filters"):
    filter_keys = [
        'hub_ids_text',
        'sel_hub_name',
        'sel_mega',
        'sel_spoke',
        'sel_loc_type',
        'sel_loc_class',
        'sel_old_loc_bucket',
        'only_independent',
        'show_hubs',
        'map_style',
        'spoke_gmv_range',
    ]

    for key in filter_keys:
        if key in st.session_state:
            del st.session_state[key]

    st.sidebar.success("✅ All filters have been cleared. Please reapply filters if needed.")
    st.rerun()

# ---------- Sidebar filters ----------
st.sidebar.markdown("### Filters / Controls")
# hub ids comma separated
hub_ids_text_key = 'hub_ids_text'
if hub_ids_text_key not in st.session_state:
    st.session_state[hub_ids_text_key] = ""
hub_ids_text = st.sidebar.text_input("Hub IDs (comma-separated). Example: 11331,15985,778", value=st.session_state[hub_ids_text_key], key=hub_ids_text_key)
hub_ids = parse_hub_ids(hub_ids_text)

# dynamic multiselects (guard for empty columns)
def safe_unique(col):
    if col in df.columns:
        vals = df[col].dropna().unique().tolist()
        return sorted(vals)
    return []

hub_names = ["All"] + safe_unique('hub_name')
mega_cities = ["All"] + safe_unique('mega_city')
spokes = ["All"] + safe_unique('spoke_name')
loc_types = ["All"] + safe_unique('loc_type')
loc_classifications = ["All"] + safe_unique('loc_classification')
old_loc_buckets = ["All"] + safe_unique('old_loc_bucket')

# ensure session_state defaults so selections persist across reruns
if 'sel_hub_name' not in st.session_state:
    st.session_state['sel_hub_name'] = "All"
if 'sel_mega' not in st.session_state:
    st.session_state['sel_mega'] = "All"
if 'sel_spoke' not in st.session_state:
    st.session_state['sel_spoke'] = "All"
if 'sel_loc_type' not in st.session_state:
    st.session_state['sel_loc_type'] = []
if 'sel_loc_class' not in st.session_state:
    st.session_state['sel_loc_class'] = []
if 'sel_old_loc_bucket' not in st.session_state:
    st.session_state['sel_old_loc_bucket'] = []
if 'only_independent' not in st.session_state:
    st.session_state['only_independent'] = False
if 'show_hubs' not in st.session_state:
    st.session_state['show_hubs'] = True
if 'map_style' not in st.session_state:
    st.session_state['map_style'] = "carto-positron"

sel_hub_name = st.sidebar.selectbox("Hub Name", hub_names, index=hub_names.index(st.session_state['sel_hub_name']) if st.session_state['sel_hub_name'] in hub_names else 0, key='sel_hub_name')
sel_mega = st.sidebar.selectbox("Mega City", mega_cities, index=mega_cities.index(st.session_state['sel_mega']) if st.session_state['sel_mega'] in mega_cities else 0, key='sel_mega')
sel_spoke = st.sidebar.selectbox("Spoke (spoke_name)", spokes, index=spokes.index(st.session_state['sel_spoke']) if st.session_state['sel_spoke'] in spokes else 0, key='sel_spoke')
sel_loc_type = st.sidebar.multiselect("Loc Type", options=loc_types[1:], default=st.session_state['sel_loc_type'], key='sel_loc_type')
sel_loc_class = st.sidebar.multiselect("Loc Classification", options=loc_classifications[1:], default=st.session_state['sel_loc_class'], key='sel_loc_class')
sel_old_loc_bucket = st.sidebar.multiselect("Old Loc Bucket", options=old_loc_buckets[1:], default=st.session_state['sel_old_loc_bucket'], key='sel_old_loc_bucket')

only_independent = st.sidebar.checkbox('Show only loc_type == "independent"', value=st.session_state['only_independent'], key='only_independent')

min_gmv = float(df['spoke_gmv'].min()) if 'spoke_gmv' in df.columns else 0.0
max_gmv = float(df['spoke_gmv'].max()) if 'spoke_gmv' in df.columns else 1.0
# ensure session state for gmv range
if 'spoke_gmv_range' not in st.session_state:
    st.session_state['spoke_gmv_range'] = (min_gmv, max_gmv)
spoke_gmv_range = st.sidebar.slider("Spoke GMV range", min_value=min_gmv, max_value=max_gmv, value=st.session_state['spoke_gmv_range'], key='spoke_gmv_range')

show_hubs = st.sidebar.checkbox("Show hub labels", value=st.session_state['show_hubs'], key='show_hubs')
map_style = st.sidebar.selectbox("Map style", ["carto-positron", "open-street-map", "carto-darkmatter", "stamen-terrain"], index=["carto-positron", "open-street-map", "carto-darkmatter", "stamen-terrain"].index(st.session_state['map_style']) if st.session_state['map_style'] in ["carto-positron", "open-street-map", "carto-darkmatter", "stamen-terrain"] else 0, key='map_style')
export_html = st.sidebar.checkbox("Save HTML snapshot on render", value=False, key='export_html')

# ---------- apply filters ----------
dsub = df.copy()

#__________________________________________________________________________
try:
    st.write("loc_type value_counts (raw):")
    st.write(df['loc_type'].value_counts(dropna=False).head(20))
except Exception as e:
    st.write("loc_type not present or error:", e)
#__________________________________________________________________________


# hub_ids (accept ints or strings)
if hub_ids:
    hub_ids_str = [str(x) for x in hub_ids]
    dsub = dsub[dsub['hub_id'].isin(hub_ids_str)]

if sel_hub_name and sel_hub_name != "All":
    dsub = dsub[dsub['hub_name'] == sel_hub_name]

if sel_mega and sel_mega != "All":
    dsub = dsub[dsub['mega_city'] == sel_mega]

if sel_spoke and sel_spoke != "All":
    dsub = dsub[dsub['spoke_name'] == sel_spoke]

if sel_loc_type:
    dsub = dsub[dsub['loc_type'].isin(sel_loc_type)]

if sel_loc_class:
    dsub = dsub[dsub['loc_classification'].isin(sel_loc_class)]

if sel_old_loc_bucket:
    dsub = dsub[dsub['old_loc_bucket'].isin(sel_old_loc_bucket)]

if only_independent:
    dsub = dsub[dsub['loc_type'].str.lower() == 'independent'] 
     



# GMV range
dsub = dsub[(dsub['spoke_gmv'] >= spoke_gmv_range[0]) & (dsub['spoke_gmv'] <= spoke_gmv_range[1])]

if dsub.empty:
    st.write("No data for selected filters.")
    st.stop()

# ---------- prepare marker sizes and unique hubs ----------
spoke_sizes = gmv_to_marker_size(dsub['spoke_gmv'])
hubs_unique = dsub.drop_duplicates(subset=['hub_id']).dropna(subset = ["hub_lat" , "hub_lon"] ).reset_index(drop=True)
# hubs_unique = dsub[dsub['loc_type'] == "hubs"]
hub_sizes = gmv_to_marker_size(hubs_unique['hub_gmv'], min_size=15, max_size=40)

# ---------- lines connecting hub -> spoke ----------
## here we need to ensure that all coords [hub_lat , hub_lon , spoke_lat spoke_lon --> exists]

dsub_lines = dsub[ dsub['spoke_lat'].notna() & dsub['spoke_lon'].notna() & dsub['hub_lat'].notna() & dsub['hub_lon'].notna() ].copy()

line_lats, line_lons = [], []
for hub_id, group in dsub_lines.groupby('hub_id'):
    hub_lat = group['hub_lat'].iloc[0]
    hub_lon = group['hub_lon'].iloc[0]
    for _, r in group.iterrows():
        line_lats += [r['spoke_lat'], hub_lat, None]
        line_lons += [r['spoke_lon'], hub_lon, None]

line_trace = go.Scattermapbox(
    lat=line_lats,
    lon=line_lons,
    mode='lines',
    line=dict(width=2, color='green' ),
    hoverinfo='none',
    showlegend=False,
)

# ---------- hubs trace ----------

hub_trace = go.Scattermapbox(
    lat=hubs_unique['hub_lat'],
    lon=hubs_unique['hub_lon'],
    mode='markers+text' if show_hubs else 'markers',
    marker=dict(size=hub_sizes,color= "red", opacity=0.85),
    text=hubs_unique['hub_name'] if show_hubs else None,
    textposition='top right',
    customdata=np.column_stack([
        hubs_unique['hub_gmv'],
        hubs_unique['hub_id'],
        hubs_unique.get('hub_locality_type', pd.Series(['']*len(hubs_unique)))
    ]),
    hovertemplate=(
        "<b>%{text}</b><br>" +
        "Hub GMV: %{customdata[0]:,.0f}<br>" +
        "Hub ID: %{customdata[1]}<br>" +
        "Hub locality type: %{customdata[2]}<br>" +
        "<extra></extra>"
    ),
    name='Hubs'
)

# ---------- spokes trace ----------
dsub_spokes = dsub[dsub['loc_type'] == "mapped"]

spoke_trace = go.Scattermapbox(
    lat=dsub_spokes['spoke_lat'],
    lon=dsub_spokes['spoke_lon'],
    mode='markers',
    marker=dict(size=spoke_sizes, color = "cyan", opacity=0.9, symbol='circle'),
    text=dsub_spokes['spoke_name'],
    customdata=np.column_stack([
        dsub_spokes['spoke_gmv'],
        dsub_spokes['hub_id'],
        dsub_spokes['spoke_id'],
        dsub_spokes['distance_bw_hub_spoke'],
        dsub_spokes['hub_locality_type'],
        dsub_spokes['spoke_locality_type'],
        dsub_spokes.get('old_loc_bucket', pd.Series(['']*len(dsub)))
    ]),
    hovertemplate=(
        "<b>%{text}</b><br>" +
        "Spoke GMV: %{customdata[0]:,.0f}<br>" +
        "Hub ID: %{customdata[1]}<br>" +
        "Spoke ID: %{customdata[2]}<br>" +
        "Distance (km): %{customdata[3]:.2f}<br>" +
        "Hub locality type: %{customdata[4]}<br>" +
        "Spoke locality type: %{customdata[5]}<br>" +
        "Old loc bucket: %{customdata[6]}<br>" +
        "<extra></extra>"
    ),
    name='Spokes'
)

indep_df = dsub[dsub['loc_type'] == "independent"]

independent_loc_trace = go.Scattermapbox(
    lat=indep_df['spoke_lat'],
    lon=indep_df['spoke_lon'],
    mode='markers',
    marker=dict(size=spoke_sizes, color='grey', opacity=0.9 , symbol = "circle"),
    text=indep_df['spoke_name'],
    customdata=np.column_stack([
        indep_df['spoke_gmv'], 
        indep_df['spoke_id'],
        indep_df.get('old_loc_bucket', pd.Series(['']*len(dsub)))

    ]),
    hovertemplate=(
        "<b>%{text}</b><br>" +
        "Locality GMV: %{customdata[0]:,.0f}<br>" +
        "Independent Loc ID: %{customdata[1]}<br>" +
        "Old loc bucket: %{customdata[2]}<br>" +
        "<extra></extra>"
    ),
    name='Independent Localities'
)



# ---------- build figure ----------
lat_vals = pd.concat([dsub['hub_lat'], dsub['spoke_lat']])
lon_vals = pd.concat([dsub['hub_lon'], dsub['spoke_lon']])
fig = go.Figure([line_trace, hub_trace, spoke_trace, independent_loc_trace])
fig.update_layout(
    mapbox=dict(
        style=map_style,
        center={'lat': float(lat_vals.mean()), 'lon': float(lon_vals.mean())},
        zoom=compute_zoom_for_bounds(lat_vals, lon_vals)
    ),
    margin=dict(r=0,t=30,l=0,b=0),
    legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.01),
    title=f"Locality map — {sel_mega if sel_mega!='All' else 'All India'}" + (f" — Hub: {sel_hub_name}" if sel_hub_name != 'All' else "")
)

# ---------- display ----------
st.plotly_chart(fig, use_container_width=True)

# optional HTML snapshot
if export_html or st.sidebar.button("Save snapshot HTML"):
    try:
        fig.write_html(HTML_OUTFILE, include_plotlyjs="cdn")
        st.sidebar.success(f"Saved snapshot: {HTML_OUTFILE}")
        st.sidebar.markdown(f"[Open snapshot](./{HTML_OUTFILE})")
    except Exception as e:
        st.sidebar.error(f"Failed to save HTML: {e}")

# ---------- small table & download ----------
with st.expander("Show data table (first 200 rows)"):
    st.dataframe(dsub.head(200))

# download filtered CSV or excel
@st.cache_data
def to_csv_bytes(df_in):
    return df_in.to_csv(index=False).encode('utf-8')

if not dsub.empty:
    st.download_button("Download filtered CSV", data=to_csv_bytes(dsub), file_name="locality_mapping_filtered.csv", mime="text/csv")

# ---------- summary metrics ----------
col1, col2, col3 = st.columns(3)
col1.metric("Rows (filtered)", len(dsub))
col2.metric("Unique hubs (filtered)", dsub['hub_id'].nunique())
col3.metric("Total spoke GMV (sum)", f"{dsub['spoke_gmv'].sum():,.0f}")

st.caption(
    "Filters available: Hub IDs (comma-separated), Hub Name, Mega City, Spoke, Loc Type, Loc Classification, Old Loc Bucket." 
    "Set 'Show only loc_type == independent' to isolate independent localities (no external independent file required)."
)

# End of script
