# streamlit_app.py
"""
Streamlit MVP Dashboard - Phase 3 skeleton

Place this file at the project root (SIH_25/). It expects the unified dataset at:
  data/processed/unified.parquet

If `unified.parquet` is missing you can (optionally) run your ETL scripts
from the app (button-driven). For local dev you can also run ETL manually first.

Run locally:
  streamlit run streamlit_app.py
"""
import os
import subprocess
import io
import json
import re
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="SIH MVP Dashboard", layout="wide")

# -----------------------------
# Config: adjust paths 
# -----------------------------
UNIFIED_PATH = "data/processed/unified.parquet"
SAMPLES_CLEANED = "data/processed/cleaned_samples.csv"
ASV_PATH = "data/synthetic/asv_table.tsv"
IMAGES_DIR = "data/synthetic/images"
NC_PATH = "data/synthetic/env_data.nc"
ETL_PHASE1 = "etl_phase1.py"
ETL_PHASE2 = "etl_phase2_integration.py"


# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(ttl=3600)
def load_unified(path=UNIFIED_PATH):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        # Ensure numeric lat/lon & depth
        for c in ("latitude", "longitude", "depth_m"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # ensure collection_date parsed
        if "collection_date" in df.columns:
            df["collection_date_parsed"] = pd.to_datetime(df["collection_date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Failed to load unified dataset ({path}): {e}")
        return None

def run_etl_from_app():
    """Runs ETL scripts. Returns True on success, False on error."""
    try:
        with st.spinner("Running ETL Phase 1 (ingest & validate)..."):
            subprocess.run(
                ["python", ETL_PHASE1,
                 "--input", "data/samples.csv",
                 "--asv", ASV_PATH,
                 "--images", IMAGES_DIR,
                 "--nc", NC_PATH,
                 "--output", "data/processed",
                 "--sqlite", "data/processed/mvp.db"],
                check=True,
            )
        with st.spinner("Running ETL Phase 2 (integration)..."):
            subprocess.run(
                ["python", ETL_PHASE2,
                 "--samples", SAMPLES_CLEANED,
                 "--asv", ASV_PATH,
                 "--images", IMAGES_DIR,
                 "--nc", NC_PATH,
                 "--output", "data/processed",
                 "--sqlite", "data/processed/mvp.db"],
                check=True,
            )
        st.success("ETL finished successfully — reloading dataset.")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"ETL failed (exit code {e.returncode}). See logs in terminal.")
        return False
    except FileNotFoundError as e:
        st.error(f"ETL script not found: {e}. Ensure ETL scripts are in repo root.")
        return False
    except Exception as e:
        st.error(f"Unexpected error when running ETL: {e}")
        return False

def image_exists(path):
    return isinstance(path, str) and path and os.path.exists(path)

def top_n_asvs_from_row(row, asv_prefix="ASV", n=10):
    # detect ASV columns
    asv_cols = [c for c in row.index if re.match(rf"^{asv_prefix}", c, flags=re.IGNORECASE)]
    if not asv_cols:
        return pd.Series(dtype=float)
   

    # convert values safely to numbers
    counts = pd.to_numeric(row[asv_cols], errors="coerce").fillna(0)

    # filter out zero-only ASVs
    counts = counts[counts > 0]

    if counts.empty:
        return pd.Series(dtype=float)

    top = counts.sort_values(ascending=False).head(n)
    return top


# -----------------------------
# UI: sidebar controls
# -----------------------------
st.sidebar.title("SIH MVP")
st.sidebar.markdown("**Navigation**")
page = st.sidebar.radio("Go to", ["Overview", "Sample List", "Map", "Sample Detail", "About"])

# Load data (cached)
df = load_unified()

if df is None:
    st.sidebar.warning("Unified dataset not found.")
    if st.sidebar.button("Run ETL now (generates unified.parquet)"):
        ok = run_etl_from_app()
        if ok:
            # clear cache and reload
            load_unified.clear()
            df = load_unified()
    st.sidebar.info("Alternatively, run ETL locally:\n`python etl_phase1.py` then `python etl_phase2_integration.py`")
else:
    st.sidebar.success(f"Loaded {len(df)} samples")

# Filters (appear if dataset loaded)
species_list = []
if df is not None and "species" in df.columns:
    species_list = sorted(df["species"].dropna().unique().tolist())

st.sidebar.markdown("----")
st.sidebar.header("Filters")
selected_species = st.sidebar.multiselect("Species", options=species_list, default=[])
# depth slider
if df is not None and "depth_m" in df.columns:
    min_depth = int(np.nanmin(df["depth_m"].fillna(0)))
    max_depth = int(np.nanmax(df["depth_m"].fillna(0)))
    sel_depth = st.sidebar.slider("Depth (m)", min_depth, max_depth, (min_depth, max_depth))
else:
    sel_depth = None

# date range filter
if df is not None and "collection_date_parsed" in df.columns:
    min_date = df["collection_date_parsed"].min().date()
    max_date = df["collection_date_parsed"].max().date()
    sel_date = st.sidebar.date_input("Collection date range", (min_date, max_date))
else:
    sel_date = None

# location filter (lat/lon ranges)
if df is not None and "latitude" in df.columns and "longitude" in df.columns:
    lat_min = float(np.nanmin(df["latitude"].fillna(0)))
    lat_max = float(np.nanmax(df["latitude"].fillna(0)))
    lon_min = float(np.nanmin(df["longitude"].fillna(0)))
    lon_max = float(np.nanmax(df["longitude"].fillna(0)))
    sel_lat = st.sidebar.slider("Latitude", lat_min, lat_max, (lat_min, lat_max))
    sel_lon = st.sidebar.slider("Longitude", lon_min, lon_max, (lon_min, lon_max))
else:
    sel_lat = sel_lon = None

search_text = st.sidebar.text_input("Search sample_id or species")

st.sidebar.markdown("----")
st.sidebar.markdown("Data export")
if df is not None:
    st.sidebar.download_button("Download current dataset (CSV)", df.to_csv(index=False), file_name="unified.csv")

# -----------------------------
# filtering function
# -----------------------------
def apply_filters(df):
    if df is None:
        return df
    out = df.copy()
    if selected_species:
        out = out[out["species"].isin(selected_species)]
    if sel_depth:
        out = out[(out["depth_m"] >= sel_depth[0]) & (out["depth_m"] <= sel_depth[1])]
    if sel_date and "collection_date_parsed" in out:
        start, end = sel_date
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        out = out[(out["collection_date_parsed"] >= start_dt) & (out["collection_date_parsed"] <= end_dt)]
    if sel_lat and sel_lon:
        out = out[(out["latitude"] >= sel_lat[0]) & (out["latitude"] <= sel_lat[1]) &
                  (out["longitude"] >= sel_lon[0]) & (out["longitude"] <= sel_lon[1])]
    if search_text:
        mask = out["sample_id"].astype(str).str.contains(search_text, case=False, na=False) | \
               out.get("species", pd.Series([""]*len(out))).astype(str).str.contains(search_text, case=False, na=False)
        out = out[mask]
    return out

filtered = apply_filters(df)

# -----------------------------
# Pages
# -----------------------------
if page == "Overview":
    st.title("Overview")
    if df is None:
        st.info("Dataset not available. Run ETL or upload processed files.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total samples", len(df))
    col2.metric("With images", int(df.get("image_available", pd.Series([False]*len(df))).sum()))
    col3.metric("With ASV", int(df.get("has_asv", pd.Series([False]*len(df))).sum()))
    if "depth_m" in df.columns:
        col4.metric("Avg depth (m)", f"{df['depth_m'].mean():.1f}")

    st.markdown("#### Acceptance checks (quick)")
    checks = []
    total = len(df)
    checks.append(("Ingestion: non-empty rows", total > 0))
    checks.append(("Images available >= 20%", (df.get("image_available", pd.Series([False]*len(df))).sum() / max(1, total)) >= 0.2))
    checks.append(("ASV present >= 80%", (df.get("has_asv", pd.Series([False]*len(df))).sum() / max(1, total)) >= 0.8))
    for label, ok in checks:
        st.write(f"- {label}: {'✅' if ok else '❗️'}")

    st.markdown("#### Map preview (filtered)")
    if filtered is not None and not filtered.empty:
        map_df = filtered.dropna(subset=["latitude", "longitude"])  # st.map expects lat/lon
        if not map_df.empty:
            st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]].head(1000))
        else:
            st.info("No geolocated samples in filtered set.")
    else:
        st.info("No samples to show for the current filters.")

elif page == "Sample List":
    st.title("Sample List")
    if filtered is None:
        st.info("Load dataset first.")
        st.stop()

    st.write(f"{len(filtered)} samples match current filters.")
    # Pagination controls
    page_size = st.number_input("Rows per page", min_value=10, max_value=200, value=25, step=5)
    page_idx = st.number_input("Page # (starts at 0)", min_value=0, value=0)
    start = int(page_idx * page_size)
    end = start + int(page_size)
    st.dataframe(filtered.reset_index(drop=True).iloc[start:end])

    # Allow quick selection to view details
    sel_id = st.selectbox("Open sample detail for:", options=filtered["sample_id"].tolist())
    if sel_id:
        st.button("Go to Sample Detail")  # UI hint — user can switch to Sample Detail tab manually
        st.write("Selected sample:", sel_id)

elif page == "Sample Detail":
    st.title("Sample Detail")
    if df is None:
        st.info("No dataset loaded.")
        st.stop()

    # selection box populated with filtered values if available
    options = filtered["sample_id"].tolist() if filtered is not None else df["sample_id"].tolist()
    sel = st.selectbox("Choose sample", options)
    if not sel:
        st.info("Choose a sample to view details.")
        st.stop()

    row = df[df["sample_id"] == sel].iloc[0]

    # Left: image + metadata ; Right: ASV + env
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Otolith image")
        thumb = row.get("thumbnail_path") or row.get("image_ref")
        if image_exists(thumb):
            try:
                st.image(thumb, caption=sel, width=300)
            except Exception as e:
                st.warning(f"Failed to render image: {e}")
                st.write("Path tried:", thumb)
        else:
            st.info("No image available for this sample.")
        st.markdown("**Metadata**")
        meta_cols = ["sample_id", "collection_date", "latitude", "longitude", "depth_m", "species"]
        meta = {c: row[c] for c in meta_cols if c in row.index}
        st.table(pd.DataFrame([meta]).T.rename(columns={0: "value"}))

    with right:
        st.subheader("Top ASVs")
        top_asv = top_n_asvs_from_row(row, asv_prefix="ASV", n=10)
        if not top_asv.empty:
            # convert to dataframe for plotting
            asv_df = pd.DataFrame({"asv": top_asv.index, "count": top_asv.values}).set_index("asv")
            st.bar_chart(asv_df)
            st.write(asv_df)
        else:
            st.info("No ASV data available for this sample.")

        st.subheader("Environmental context")
        # preferred: env_temperature column extracted in ETL
        if "env_temperature" in row.index and not pd.isna(row["env_temperature"]):
            st.write(f"Temperature at collection: **{row['env_temperature']}**")
        else:
            st.info("No pre-extracted environmental value; attempting to preview NetCDF (if present).")
            if os.path.exists(NC_PATH):
                try:
                    import xarray as xr
                    ds = xr.open_dataset(NC_PATH)
                    varnames = [v for v in ds.data_vars]
                    st.write("NetCDF variables:", varnames)
                    # quick plot of first time step & first variable
                    v = varnames[0]
                    arr = ds[v].isel(time=0) if "time" in ds.dims else ds[v]
                    fig, ax = plt.subplots(figsize=(5,3))
                    im = ax.pcolormesh(arr.lon.values, arr.lat.values, arr.values, shading="auto")
                    ax.set_title(f"{v} (time index 0)")
                    fig.colorbar(im, ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Failed to read NetCDF for preview: {e}")
            else:
                st.info("No NetCDF file found in repo (skip).")

elif page == "Map":
    st.title("Map view")
    if df is None:
        st.info("Dataset not loaded.")
        st.stop()

    map_df = df.dropna(subset=["latitude", "longitude"])
    if map_df.empty:
        st.info("No geolocated samples.")
    else:
        # color by env value if present
        if "env_temperature" in map_df.columns:
            # show points on a map colored by temperature (approx)
            st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]].head(2000))
            st.write("Map shows sample locations (use filters to refine).")
        else:
            st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]].head(2000))

elif page == "About":
    st.title("About this MVP")
    st.markdown("""
    **SIH MVP** — integrates morphological (otolith), molecular (ASV), and environmental (NetCDF) data.
    - ETL scripts: `etl_phase1.py`, `etl_phase2_integration.py`
    - Data inputs: `data/samples.csv`, `data/asv_table.tsv`, `data/images/`, `data/env_data.nc`
    - Processed outputs: `data/processed/unified.parquet`
    """)
    st.markdown("**Usage**: use the sidebar filters, select a sample to view details. Download filtered CSV from sidebar.")

# End of app
