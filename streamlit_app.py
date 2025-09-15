# streamlit_app.py
"""
Enhanced Streamlit MVP Dashboard - Marine Data Analytics Platform

A comprehensive dashboard for analyzing marine biological data including:
- Otolith morphological data
- ASV (Amplicon Sequence Variant) molecular data  
- Environmental data from NetCDF files
- Integrated data visualization and analytics

Deployment ready for Streamlit Cloud with enhanced features:
- Advanced data visualizations
- Interactive filtering and search
- Data export capabilities
- Performance optimizations
- Error handling and user feedback

Run locally: streamlit run streamlit_app.py
Deploy: Push to GitHub and deploy via Streamlit Cloud
"""

import os
import subprocess
import io
import json
import re
import warnings
from datetime import datetime, timedelta
import traceback
import logging
from typing import Optional, Dict, List, Tuple, Any

import xarray as xr
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.getcwd()   # <-- instead of os.path.dirname(__file__)

# Page configuration with enhanced settings
st.set_page_config(
    page_title="Marine Data Analytics Platform",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "Marine Data Analytics Platform v2.0"
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Configuration and Constants
# -----------------------------
BASE_DIR = os.getcwd()

# Data paths
UNIFIED_PATH = "data/processed/unified.parquet"
SAMPLES_CLEANED = "data/processed/cleaned_samples.csv"
ASV_PATH = "data/asv_table.tsv"
IMAGES_DIR = "data/images"
NC_PATH = "data/env_data.nc"
ETL_PHASE1 = "etl_phase1.py"
ETL_PHASE2 = "etl_phase2_integration.py"

# App configuration
APP_VERSION = "2.0.0"
MAX_SAMPLES_DISPLAY = 1000
CACHE_TTL = 3600  # 1 hour

# Color schemes for visualizations
COLOR_PALETTE = px.colors.qualitative.Set3
MARINE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'last_etl_run' not in st.session_state:
    st.session_state.last_etl_run = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None


# -----------------------------
# Enhanced Utility Functions
# -----------------------------

@st.cache_data(ttl=CACHE_TTL)
def load_unified(path=UNIFIED_PATH):
    """Load and preprocess the unified dataset with enhanced error handling."""
    if not os.path.exists(path):
        logger.warning(f"Unified dataset not found at {path}")
        return None
    
    try:
        logger.info(f"Loading unified dataset from {path}")
        df = pd.read_parquet(path)
        
        # Data preprocessing
        df = preprocess_dataframe(df)
        
        logger.info(f"Successfully loaded {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load unified dataset: {str(e)}")
        st.error(f"❌ Failed to load unified dataset ({path}): {e}")
        return None

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe with data type conversions and cleaning."""
    try:
        # Ensure numeric columns
        numeric_cols = ["latitude", "longitude", "depth_m", "env_temperature"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Parse dates
        if "collection_date" in df.columns:
            df["collection_date_parsed"] = pd.to_datetime(df["collection_date"], errors="coerce")
        
        # Add derived columns
        if "latitude" in df.columns and "longitude" in df.columns:
            df["has_coordinates"] = df["latitude"].notna() & df["longitude"].notna()
        
        if "image_available" not in df.columns:
            df["image_available"] = False
        
        if "has_asv" not in df.columns:
            asv_cols = [c for c in df.columns if c.startswith("ASV")]
            df["has_asv"] = df[asv_cols].notna().any(axis=1) if asv_cols else False
        
        return df
        
    except Exception as e:
        logger.error(f"Error preprocessing dataframe: {str(e)}")
        return df

@st.cache_data(ttl=CACHE_TTL)
def load_environmental_data(path=NC_PATH):
    """Load environmental NetCDF data with error handling."""
    if not os.path.exists(path):
        return None
    
    try:
        ds = xr.open_dataset(path)
        logger.info(f"Loaded environmental data with variables: {list(ds.data_vars)}")
        return ds
    except Exception as e:
        logger.error(f"Failed to load environmental data: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data summary statistics."""
    if df is None or df.empty:
        return {}
    
    summary = {
        "total_samples": len(df),
        "samples_with_images": int(df.get("image_available", pd.Series([False]*len(df))).sum()),
        "samples_with_asv": int(df.get("has_asv", pd.Series([False]*len(df))).sum()),
        "samples_with_coordinates": int(df.get("has_coordinates", pd.Series([False]*len(df))).sum()),
        "unique_species": df["species"].nunique() if "species" in df.columns else 0,
        "date_range": None,
        "depth_range": None,
        "geographic_bounds": None
    }
    
    # Date range
    if "collection_date_parsed" in df.columns:
        dates = df["collection_date_parsed"].dropna()
        if not dates.empty:
            summary["date_range"] = (dates.min().date(), dates.max().date())
    
    # Depth range
    if "depth_m" in df.columns:
        depths = df["depth_m"].dropna()
        if not depths.empty:
            summary["depth_range"] = (depths.min(), depths.max())
    
    # Geographic bounds
    if "latitude" in df.columns and "longitude" in df.columns:
        coords = df[["latitude", "longitude"]].dropna()
        if not coords.empty:
            summary["geographic_bounds"] = {
                "lat_min": coords["latitude"].min(),
                "lat_max": coords["latitude"].max(),
                "lon_min": coords["longitude"].min(),
                "lon_max": coords["longitude"].max()
            }
    
    return summary

def run_etl_from_app():
    """Enhanced ETL runner with better error handling and progress tracking."""
    try:
        # Check if ETL scripts exist
        if not os.path.exists(ETL_PHASE1) or not os.path.exists(ETL_PHASE2):
            st.error("❌ ETL scripts not found. Please ensure they are in the repository root.")
            return False
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Phase 1
        status_text.text("🔄 Running ETL Phase 1 (ingest & validate)...")
        progress_bar.progress(25)
        
        result1 = subprocess.run(
            [sys.executable, ETL_PHASE1,
             "--input", "data/samples.csv",
             "--asv", ASV_PATH,
             "--images", IMAGES_DIR,
             "--nc", NC_PATH,
             "--output", "data/processed",
             "--sqlite", "data/processed/mvp.db"],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Phase 2
        status_text.text("🔄 Running ETL Phase 2 (integration)...")
        progress_bar.progress(75)
        
        result2 = subprocess.run(
            [sys.executable, ETL_PHASE2,
             "--samples", SAMPLES_CLEANED,
             "--asv", ASV_PATH,
             "--images", IMAGES_DIR,
             "--nc", NC_PATH,
             "--output", "data/processed",
             "--sqlite", "data/processed/mvp.db"],
            check=True,
            capture_output=True,
            text=True
        )
        
        progress_bar.progress(100)
        status_text.text("✅ ETL completed successfully!")
        
        # Update session state
        st.session_state.last_etl_run = datetime.now()
        
        # Clear cache to force reload
        load_unified.clear()
        load_environmental_data.clear()
        get_data_summary.clear()
        
        st.success("🎉 ETL finished successfully — dataset will be reloaded.")
        return True
        
    except subprocess.CalledProcessError as e:
        st.error(f"❌ ETL failed (exit code {e.returncode})")
        st.error(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError as e:
        st.error(f"❌ ETL script not found: {e}")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error when running ETL: {e}")
        logger.error(f"ETL error: {traceback.format_exc()}")
        return False
    finally:
        # Clean up progress indicators
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def create_visualizations(df: pd.DataFrame, sample_id: str = None) -> Dict[str, Any]:
    """Create comprehensive visualizations for the data."""
    visualizations = {}
    
    if df is None or df.empty:
        return visualizations
    
    try:
        # Species distribution
        if "species" in df.columns:
            species_counts = df["species"].value_counts().head(10)
            fig_species = px.pie(
                values=species_counts.values,
                names=species_counts.index,
                title="Species Distribution",
                color_discrete_sequence=MARINE_COLORS
            )
            visualizations["species_distribution"] = fig_species
        
        # Depth distribution
        if "depth_m" in df.columns:
            depth_data = df["depth_m"].dropna()
            if not depth_data.empty:
                fig_depth = px.histogram(
                    depth_data,
                    title="Depth Distribution",
                    labels={"value": "Depth (m)", "count": "Number of Samples"},
                    color_discrete_sequence=MARINE_COLORS
                )
                visualizations["depth_distribution"] = fig_depth
        
        # Geographic distribution
        if "latitude" in df.columns and "longitude" in df.columns:
            geo_data = df[["latitude", "longitude"]].dropna()
            if not geo_data.empty:
                fig_map = px.scatter_mapbox(
                    geo_data,
                    lat="latitude",
                    lon="longitude",
                    title="Geographic Distribution of Samples",
                    mapbox_style="open-street-map",
                    zoom=1
                )
                visualizations["geographic_map"] = fig_map
        
        # Time series if available
        if "collection_date_parsed" in df.columns:
            date_data = df["collection_date_parsed"].dropna()
            if not date_data.empty:
                monthly_counts = date_data.dt.to_period('M').value_counts().sort_index()
                fig_time = px.line(
                    x=monthly_counts.index.astype(str),
                    y=monthly_counts.values,
                    title="Sample Collection Over Time",
                    labels={"x": "Month", "y": "Number of Samples"}
                )
                visualizations["time_series"] = fig_time
        
        # ASV analysis for specific sample
        if sample_id and sample_id in df["sample_id"].values:
            sample_row = df[df["sample_id"] == sample_id].iloc[0]
            asv_data = top_n_asvs_from_row(sample_row, n=15)
            if not asv_data.empty:
                fig_asv = px.bar(
                    x=asv_data.values,
                    y=asv_data.index,
                    orientation='h',
                    title=f"Top ASVs for Sample {sample_id}",
                    labels={"x": "Count", "y": "ASV ID"}
                )
                visualizations["sample_asv"] = fig_asv
    
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        st.warning(f"Some visualizations could not be created: {e}")
    
    return visualizations

def image_exists(path):
    """Check if image file exists and is accessible."""
    return isinstance(path, str) and path and os.path.exists(path)

def top_n_asvs_from_row(row, asv_prefix="ASV", n=10):
    """Extract top N ASVs from a sample row with enhanced error handling."""
    try:
        # Detect ASV columns
        asv_cols = [c for c in row.index if re.match(rf"^{asv_prefix}", c, flags=re.IGNORECASE)]
        if not asv_cols:
            return pd.Series(dtype=float)
        
        # Convert values safely to numbers
        counts = pd.to_numeric(row[asv_cols], errors="coerce").fillna(0)
        
        # Filter out zero-only ASVs
        counts = counts[counts > 0]
        
        if counts.empty:
            return pd.Series(dtype=float)
        
        top = counts.sort_values(ascending=False).head(n)
        return top
        
    except Exception as e:
        logger.error(f"Error extracting ASVs: {str(e)}")
        return pd.Series(dtype=float)

def export_data(df: pd.DataFrame, format_type: str = "csv") -> bytes:
    """Export data in various formats."""
    try:
        if format_type == "csv":
            return df.to_csv(index=False).encode('utf-8')
        elif format_type == "excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            return output.getvalue()
        elif format_type == "json":
            return df.to_json(orient='records', date_format='iso').encode('utf-8')
        else:
            return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        return b""

def create_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Create enhanced sidebar filters with better UX."""
    filters = {}
    
    if df is None or df.empty:
        return filters
    
    st.sidebar.markdown("### 🔍 Data Filters")
    
    # Species filter
    if "species" in df.columns:
        species_list = sorted(df["species"].dropna().unique().tolist())
        filters["species"] = st.sidebar.multiselect(
            "🐟 Species", 
            options=species_list, 
            default=[],
            help="Select one or more species to filter"
        )
    
    # Depth filter
    if "depth_m" in df.columns:
        depths = df["depth_m"].dropna()
        if not depths.empty:
            min_depth = int(depths.min())
            max_depth = int(depths.max())
            filters["depth"] = st.sidebar.slider(
                "🌊 Depth (m)", 
                min_depth, 
                max_depth, 
                (min_depth, max_depth),
                help="Select depth range in meters"
            )
    
    # Date range filter
    if "collection_date_parsed" in df.columns:
        dates = df["collection_date_parsed"].dropna()
        if not dates.empty:
            min_date = dates.min().date()
            max_date = dates.max().date()
            filters["date_range"] = st.sidebar.date_input(
                "📅 Collection Date Range", 
                (min_date, max_date),
                help="Select date range for sample collection"
            )
    
    # Geographic filters
    if "latitude" in df.columns and "longitude" in df.columns:
        coords = df[["latitude", "longitude"]].dropna()
        if not coords.empty:
            lat_min, lat_max = coords["latitude"].min(), coords["latitude"].max()
            lon_min, lon_max = coords["longitude"].min(), coords["longitude"].max()
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                filters["latitude"] = st.slider(
                    "📍 Latitude", 
                    lat_min, lat_max, 
                    (lat_min, lat_max),
                    help="Latitude range"
                )
            with col2:
                filters["longitude"] = st.slider(
                    "📍 Longitude", 
                    lon_min, lon_max, 
                    (lon_min, lon_max),
                    help="Longitude range"
                )
    
    # Search filter
    filters["search"] = st.sidebar.text_input(
        "🔍 Search", 
        placeholder="Search sample ID or species...",
        help="Search across sample IDs and species names"
    )
    
    # Data quality filters
    st.sidebar.markdown("### 📊 Data Quality")
    filters["has_images"] = st.sidebar.checkbox("📷 Has Images", value=False)
    filters["has_asv"] = st.sidebar.checkbox("🧬 Has ASV Data", value=False)
    filters["has_coordinates"] = st.sidebar.checkbox("📍 Has Coordinates", value=False)
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to dataframe with enhanced logic."""
    if df is None or df.empty:
        return df
    
    filtered_df = df.copy()
    
    try:
        # Species filter
        if filters.get("species"):
            filtered_df = filtered_df[filtered_df["species"].isin(filters["species"])]
        
        # Depth filter
        if filters.get("depth"):
            min_depth, max_depth = filters["depth"]
            filtered_df = filtered_df[
                (filtered_df["depth_m"] >= min_depth) & 
                (filtered_df["depth_m"] <= max_depth)
            ]
        
        # Date range filter
        if filters.get("date_range") and "collection_date_parsed" in filtered_df.columns:
            start_date, end_date = filters["date_range"]
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            filtered_df = filtered_df[
                (filtered_df["collection_date_parsed"] >= start_dt) & 
                (filtered_df["collection_date_parsed"] <= end_dt)
            ]
        
        # Geographic filters
        if filters.get("latitude") and filters.get("longitude"):
            lat_range = filters["latitude"]
            lon_range = filters["longitude"]
            filtered_df = filtered_df[
                (filtered_df["latitude"] >= lat_range[0]) & 
                (filtered_df["latitude"] <= lat_range[1]) &
                (filtered_df["longitude"] >= lon_range[0]) & 
                (filtered_df["longitude"] <= lon_range[1])
            ]
        
        # Search filter
        if filters.get("search"):
            search_term = filters["search"].lower()
            search_mask = (
                filtered_df["sample_id"].astype(str).str.contains(search_term, case=False, na=False) |
                filtered_df.get("species", pd.Series([""]*len(filtered_df))).astype(str).str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[search_mask]
        
        # Data quality filters
        if filters.get("has_images"):
            filtered_df = filtered_df[filtered_df.get("image_available", False)]
        
        if filters.get("has_asv"):
            filtered_df = filtered_df[filtered_df.get("has_asv", False)]
        
        if filters.get("has_coordinates"):
            filtered_df = filtered_df[filtered_df.get("has_coordinates", False)]
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error applying filters: {str(e)}")
        st.warning(f"Error applying some filters: {e}")
        return df


# -----------------------------
# Main Application UI
# -----------------------------

# Header
st.markdown('<h1 class="main-header">🐟 Marine Data Analytics Platform</h1>', unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center; color: #666; margin-bottom: 2rem;'>Version {APP_VERSION} | Advanced Marine Biological Data Analysis</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Select Page", 
    ["📊 Dashboard", "🔍 Data Explorer", "🗺️ Geographic View", "🔬 Sample Analysis", "📈 Analytics", "ℹ️ About"],
    index=0
)

# Load data with enhanced error handling
with st.spinner("🔄 Loading data..."):
    df = load_unified()

# Data loading status and ETL controls
if df is None:
    st.sidebar.markdown("### ⚠️ Data Status")
    st.sidebar.error("Unified dataset not found")
    
    st.sidebar.markdown("### 🔧 Data Processing")
    if st.sidebar.button("🚀 Run ETL Pipeline", type="primary"):
        with st.spinner("Running ETL pipeline..."):
            success = run_etl_from_app()
            if success:
                st.rerun()
    
    st.sidebar.info("💡 **Alternative**: Run ETL locally:\n```bash\npython etl_phase1.py\npython etl_phase2_integration.py\n```")
    
    # Show help for deployment
    st.sidebar.markdown("### 🚀 Deployment")
    st.sidebar.info("For Streamlit Cloud deployment, ensure all data files are in the repository.")
    
else:
    st.sidebar.markdown("### ✅ Data Status")
    summary = get_data_summary(df)
    st.sidebar.success(f"📊 {summary['total_samples']} samples loaded")
    
    # Quick stats in sidebar
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Images", summary['samples_with_images'])
    with col2:
        st.metric("ASV Data", summary['samples_with_asv'])

# Create filters
filters = create_sidebar_filters(df)

# Apply filters
if df is not None:
    filtered_df = apply_filters(df, filters)
    st.session_state.filtered_data = filtered_df
else:
    filtered_df = None

# Export options
if df is not None:
    st.sidebar.markdown("### 📤 Export Data")
    export_format = st.sidebar.selectbox("Format", ["CSV", "Excel", "JSON"])
    
    if st.sidebar.button("💾 Download Filtered Data"):
        if filtered_df is not None and not filtered_df.empty:
            data = export_data(filtered_df, export_format.lower())
            file_extension = {"csv": "csv", "excel": "xlsx", "json": "json"}[export_format.lower()]
            st.sidebar.download_button(
                f"📥 Download {export_format}",
                data,
                file_name=f"marine_data_filtered.{file_extension}",
                mime="application/octet-stream"
            )
        else:
            st.sidebar.warning("No data to export")

# -----------------------------
# Enhanced Page Content
# -----------------------------

if page == "📊 Dashboard":
    st.title("📊 Data Dashboard")
    
    if df is None:
        st.error("❌ Dataset not available. Please run ETL pipeline or ensure data files are present.")
        st.stop()
    
    # Key metrics
    summary = get_data_summary(df)
    
    # Main metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📊 Total Samples", 
            f"{summary['total_samples']:,}",
            help="Total number of samples in the dataset"
        )
    
    with col2:
        st.metric(
            "📷 With Images", 
            f"{summary['samples_with_images']:,}",
            f"{summary['samples_with_images']/summary['total_samples']*100:.1f}%",
            help="Samples with available otolith images"
        )
    
    with col3:
        st.metric(
            "🧬 With ASV Data", 
            f"{summary['samples_with_asv']:,}",
            f"{summary['samples_with_asv']/summary['total_samples']*100:.1f}%",
            help="Samples with ASV molecular data"
        )
    
    with col4:
        st.metric(
            "📍 With Coordinates", 
            f"{summary['samples_with_coordinates']:,}",
            f"{summary['samples_with_coordinates']/summary['total_samples']*100:.1f}%",
            help="Samples with geographic coordinates"
        )
    
    with col5:
        if "depth_m" in df.columns:
            avg_depth = df["depth_m"].mean()
            st.metric(
                "🌊 Avg Depth", 
                f"{avg_depth:.1f}m",
                help="Average collection depth"
            )
        else:
            st.metric("🌊 Avg Depth", "N/A")
    
    # Data quality checks
    st.markdown("### 🔍 Data Quality Assessment")
    
    quality_checks = []
    total = summary['total_samples']
    
    quality_checks.append(("✅ Data Ingestion", total > 0, "Dataset contains samples"))
    quality_checks.append(("📷 Image Coverage", summary['samples_with_images']/max(1, total) >= 0.2, "At least 20% have images"))
    quality_checks.append(("🧬 ASV Coverage", summary['samples_with_asv']/max(1, total) >= 0.8, "At least 80% have ASV data"))
    quality_checks.append(("📍 Geographic Coverage", summary['samples_with_coordinates']/max(1, total) >= 0.5, "At least 50% have coordinates"))
    
    for check_name, passed, description in quality_checks:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"{'✅' if passed else '❌'} **{check_name}**")
        with col2:
            st.markdown(f"{description}")
    
    # Quick visualizations
    st.markdown("### 📈 Quick Insights")
    
    if filtered_df is not None and not filtered_df.empty:
        # Create visualizations
        viz = create_visualizations(filtered_df)
        
        if "species_distribution" in viz:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(viz["species_distribution"], use_container_width=True)
            with col2:
                if "depth_distribution" in viz:
                    st.plotly_chart(viz["depth_distribution"], use_container_width=True)
        
        # Geographic preview
        if "geographic_map" in viz:
            st.plotly_chart(viz["geographic_map"], use_container_width=True)
        
        # Time series if available
        if "time_series" in viz:
            st.plotly_chart(viz["time_series"], use_container_width=True)
    
    else:
        st.info("No data available for the current filters. Adjust filters to see visualizations.")

elif page == "🔍 Data Explorer":
    st.title("🔍 Data Explorer")
    
    if filtered_df is None:
        st.error("❌ No data available. Please ensure dataset is loaded.")
        st.stop()
    
    st.markdown(f"### 📊 Showing {len(filtered_df):,} samples (filtered from {len(df):,} total)")
    
    # Display options
    col1, col2, col3 = st.columns(3)
    with col1:
        page_size = st.selectbox("Rows per page", [25, 50, 100, 200], index=1)
    with col2:
        show_columns = st.multiselect(
            "Select columns to display",
            options=filtered_df.columns.tolist(),
            default=["sample_id", "species", "collection_date", "latitude", "longitude", "depth_m"]
        )
    with col3:
        sort_by = st.selectbox("Sort by", ["sample_id", "species", "collection_date", "depth_m"], index=0)
        sort_asc = st.checkbox("Ascending", value=True)
    
    # Apply sorting and column selection
    if show_columns:
        display_df = filtered_df[show_columns].copy()
    else:
        display_df = filtered_df.copy()
    
    if sort_by in display_df.columns:
        display_df = display_df.sort_values(by=sort_by, ascending=sort_asc)
    
    # Pagination
    total_pages = (len(display_df) - 1) // page_size + 1
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
    
    start_idx = page_num * page_size
    end_idx = min(start_idx + page_size, len(display_df))
    
    # Display data
    st.dataframe(
        display_df.iloc[start_idx:end_idx].reset_index(drop=True),
        use_container_width=True,
        height=400
    )
    
    # Sample selection for detailed view
    st.markdown("### 🔬 Sample Selection")
    if not filtered_df.empty:
        sample_options = filtered_df["sample_id"].tolist()
        selected_sample = st.selectbox(
            "Select a sample for detailed analysis:",
            options=sample_options,
            help="Choose a sample to view detailed information"
        )
        
        if selected_sample:
            st.success(f"Selected: {selected_sample}")
            if st.button("🔬 View Sample Details", type="primary"):
                st.session_state.selected_sample = selected_sample
                st.rerun()

elif page == "🔬 Sample Analysis":
    st.title("🔬 Sample Analysis")
    
    if df is None:
        st.error("❌ No dataset loaded.")
        st.stop()
    
    # Sample selection
    sample_options = df["sample_id"].tolist()
    if st.session_state.get("selected_sample") and st.session_state.selected_sample in sample_options:
        default_idx = sample_options.index(st.session_state.selected_sample)
    else:
        default_idx = 0
    
    selected_sample = st.selectbox(
        "🔬 Choose a sample for detailed analysis:",
        options=sample_options,
        index=default_idx,
        help="Select a sample to view detailed information including images, ASV data, and environmental context"
    )
    
    if not selected_sample:
        st.info("Please select a sample to view details.")
        st.stop()
    
    # Get sample data
    sample_row = df[df["sample_id"] == selected_sample].iloc[0]
    
    # Sample overview
    st.markdown(f"### 📋 Sample Overview: {selected_sample}")
    
    # Key metrics for this sample
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Species", sample_row.get("species", "Unknown"))
    with col2:
        st.metric("Depth", f"{sample_row.get('depth_m', 'N/A')}m")
    with col3:
        st.metric("Has Image", "✅" if sample_row.get("image_available", False) else "❌")
    with col4:
        st.metric("Has ASV", "✅" if sample_row.get("has_asv", False) else "❌")
    
    # Main content in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image & Metadata", "🧬 ASV Analysis", "🌍 Environmental Data", "📊 Statistics"])
    
    with tab1:
        st.markdown("#### 🖼️ Otolith Image")
        
        # Image display
        image_path = sample_row.get("thumbnail_path") or sample_row.get("image_ref")
        if image_exists(image_path):
            try:
                st.image(image_path, caption=f"Otolith image for {selected_sample}", width=400)
            except Exception as e:
                st.warning(f"Failed to render image: {e}")
                st.write("Path tried:", image_path)
        else:
            st.info("No image available for this sample.")
        
        # Metadata table
        st.markdown("#### 📋 Sample Metadata")
        metadata_cols = ["sample_id", "species", "collection_date", "latitude", "longitude", "depth_m"]
        metadata = {col: sample_row[col] for col in metadata_cols if col in sample_row.index}
        
        metadata_df = pd.DataFrame(list(metadata.items()), columns=["Property", "Value"])
        st.dataframe(metadata_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("#### 🧬 ASV (Amplicon Sequence Variant) Analysis")
        
        # Get top ASVs
        top_asvs = top_n_asvs_from_row(sample_row, n=15)
        
        if not top_asvs.empty:
            # Create ASV visualization
            asv_fig = px.bar(
                x=top_asvs.values,
                y=top_asvs.index,
                orientation='h',
                title=f"Top ASVs for Sample {selected_sample}",
                labels={"x": "Count", "y": "ASV ID"},
                color=top_asvs.values,
                color_continuous_scale="viridis"
            )
            asv_fig.update_layout(height=500)
            st.plotly_chart(asv_fig, use_container_width=True)
            
            # ASV data table
            asv_df = pd.DataFrame({
                "ASV ID": top_asvs.index,
                "Count": top_asvs.values,
                "Relative Abundance": (top_asvs.values / top_asvs.values.sum() * 100).round(2)
            })
            st.dataframe(asv_df, use_container_width=True, hide_index=True)
            
            # ASV statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total ASVs", len(top_asvs))
            with col2:
                st.metric("Total Count", f"{top_asvs.sum():,.0f}")
            with col3:
                st.metric("Most Abundant", f"{top_asvs.iloc[0]:.0f}")
        else:
            st.info("No ASV data available for this sample.")
    
    with tab3:
        st.markdown("#### 🌍 Environmental Context")
        
        # Environmental data
        if "env_temperature" in sample_row.index and not pd.isna(sample_row["env_temperature"]):
            st.metric("🌡️ Temperature", f"{sample_row['env_temperature']:.2f}°C")
        else:
            st.info("No pre-extracted environmental temperature data available.")
        
        # NetCDF preview
        if os.path.exists(NC_PATH):
            try:
                ds_env = load_environmental_data()
                if ds_env is not None:
                    st.markdown("#### 📊 Environmental Data Preview")
                    varnames = list(ds_env.data_vars)
                    st.write(f"Available variables: {', '.join(varnames)}")
                    
                    # Show first variable as example
                    if varnames:
                        var_name = varnames[0]
                        st.write(f"**{var_name}** preview:")
                        
                        # Create a simple plot
                        if "time" in ds_env.dims:
                            data_slice = ds_env[var_name].isel(time=0)
                        else:
                            data_slice = ds_env[var_name]
                        
                        if "lat" in data_slice.dims and "lon" in data_slice.dims:
                            fig = px.imshow(
                                data_slice.values,
                                title=f"{var_name} (Environmental Data)",
                                color_continuous_scale="viridis"
                            )
                            st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Failed to load environmental data: {e}")
        else:
            st.info("No environmental NetCDF file found.")
    
    with tab4:
        st.markdown("#### 📊 Sample Statistics")
        
        # Calculate statistics
        if not top_asvs.empty:
            stats_data = {
                "Metric": [
                    "Total ASV Count",
                    "Number of ASVs",
                    "Shannon Diversity",
                    "Simpson Diversity",
                    "Most Abundant ASV",
                    "Least Abundant ASV"
                ],
                "Value": [
                    f"{top_asvs.sum():,.0f}",
                    f"{len(top_asvs)}",
                    f"{-sum((top_asvs/top_asvs.sum()) * np.log(top_asvs/top_asvs.sum())):.3f}",
                    f"{1 - sum((top_asvs/top_asvs.sum())**2):.3f}",
                    f"{top_asvs.iloc[0]:.0f}",
                    f"{top_asvs.iloc[-1]:.0f}"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No ASV data available for statistical analysis.")

elif page == "🗺️ Geographic View":
    st.title("🗺️ Geographic Distribution")
    
    if df is None:
        st.error("❌ Dataset not loaded.")
        st.stop()
    
    # Filter data for mapping
    map_df = df.dropna(subset=["latitude", "longitude"])
    
    if map_df.empty:
        st.warning("⚠️ No geolocated samples available for mapping.")
        st.info("Ensure your data contains valid latitude and longitude coordinates.")
        st.stop()
    
    st.markdown(f"### 📍 Showing {len(map_df):,} samples with coordinates")
    
    # Map visualization options
    col1, col2 = st.columns(2)
    with col1:
        color_by = st.selectbox(
            "Color by",
            ["None", "Species", "Depth", "Collection Date", "ASV Count"],
            help="Choose how to color the map points"
        )
    with col2:
        size_by = st.selectbox(
            "Size by",
            ["None", "Depth", "ASV Count"],
            help="Choose how to size the map points"
        )
    
    # Create map visualization
    if color_by == "None" and size_by == "None":
        # Simple map
        st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]].head(2000))
    else:
        # Enhanced map with Plotly
        fig = px.scatter_mapbox(
            map_df.head(2000),
            lat="latitude",
            lon="longitude",
            color=color_by.lower().replace(" ", "_") if color_by != "None" else None,
            size=size_by.lower().replace(" ", "_") if size_by != "None" else None,
            hover_data=["sample_id", "species", "depth_m", "collection_date"],
            mapbox_style="open-street-map",
            zoom=1,
            title="Sample Geographic Distribution"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic statistics
    st.markdown("### 📊 Geographic Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latitude Range", f"{map_df['latitude'].min():.2f}° to {map_df['latitude'].max():.2f}°")
    with col2:
        st.metric("Longitude Range", f"{map_df['longitude'].min():.2f}° to {map_df['longitude'].max():.2f}°")
    with col3:
        st.metric("Geographic Spread", f"{(map_df['latitude'].max() - map_df['latitude'].min()):.2f}° lat")
    with col4:
        st.metric("Longitudinal Spread", f"{(map_df['longitude'].max() - map_df['longitude'].min()):.2f}° lon")

elif page == "📈 Analytics":
    st.title("📈 Advanced Analytics")
    
    if df is None:
        st.error("❌ Dataset not loaded.")
        st.stop()
    
    if filtered_df is None or filtered_df.empty:
        st.warning("⚠️ No data available for analysis. Adjust filters to see data.")
        st.stop()
    
    st.markdown(f"### 📊 Analyzing {len(filtered_df):,} samples")
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Data Overview", "📊 Statistical Analysis", "🔬 ASV Analysis", "🌍 Environmental Analysis"])
    
    with tab1:
        st.markdown("#### 📋 Dataset Overview")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sample Distribution by Species**")
            if "species" in filtered_df.columns:
                species_counts = filtered_df["species"].value_counts()
                fig_species = px.pie(
                    values=species_counts.values,
                    names=species_counts.index,
                    title="Species Distribution"
                )
                st.plotly_chart(fig_species, use_container_width=True)
        
        with col2:
            st.markdown("**Depth Distribution**")
            if "depth_m" in filtered_df.columns:
                depth_data = filtered_df["depth_m"].dropna()
                if not depth_data.empty:
                    fig_depth = px.histogram(
                        depth_data,
                        title="Depth Distribution",
                        labels={"value": "Depth (m)", "count": "Number of Samples"}
                    )
                    st.plotly_chart(fig_depth, use_container_width=True)
    
    with tab2:
        st.markdown("#### 📊 Statistical Analysis")
        
        # Correlation analysis
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.markdown("**Correlation Matrix**")
            corr_matrix = filtered_df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix of Numeric Variables"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Summary statistics
        st.markdown("**Summary Statistics**")
        summary_stats = filtered_df.describe()
        st.dataframe(summary_stats, use_container_width=True)
    
    with tab3:
        st.markdown("#### 🔬 ASV Analysis")
        
        # ASV diversity analysis
        asv_cols = [c for c in filtered_df.columns if c.startswith("ASV")]
        if asv_cols:
            st.markdown("**ASV Diversity Metrics**")
            
            # Calculate diversity for each sample
            diversity_data = []
            for idx, row in filtered_df.iterrows():
                asv_counts = pd.to_numeric(row[asv_cols], errors="coerce").fillna(0)
                asv_counts = asv_counts[asv_counts > 0]
                
                if len(asv_counts) > 0:
                    shannon = -sum((asv_counts/asv_counts.sum()) * np.log(asv_counts/asv_counts.sum()))
                    simpson = 1 - sum((asv_counts/asv_counts.sum())**2)
                    
                    diversity_data.append({
                        "sample_id": row["sample_id"],
                        "shannon_diversity": shannon,
                        "simpson_diversity": simpson,
                        "asv_count": len(asv_counts),
                        "total_count": asv_counts.sum()
                    })
            
            if diversity_data:
                diversity_df = pd.DataFrame(diversity_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_shannon = px.histogram(
                        diversity_df,
                        x="shannon_diversity",
                        title="Shannon Diversity Distribution"
                    )
                    st.plotly_chart(fig_shannon, use_container_width=True)
                
                with col2:
                    fig_simpson = px.histogram(
                        diversity_df,
                        x="simpson_diversity",
                        title="Simpson Diversity Distribution"
                    )
                    st.plotly_chart(fig_simpson, use_container_width=True)
                
                # Diversity vs depth
                if "depth_m" in filtered_df.columns:
                    merged_df = diversity_df.merge(
                        filtered_df[["sample_id", "depth_m"]], 
                        on="sample_id"
                    )
                    
                    fig_depth_div = px.scatter(
                        merged_df,
                        x="depth_m",
                        y="shannon_diversity",
                        color="asv_count",
                        title="Shannon Diversity vs Depth",
                        labels={"depth_m": "Depth (m)", "shannon_diversity": "Shannon Diversity"}
                    )
                    st.plotly_chart(fig_depth_div, use_container_width=True)
        else:
            st.info("No ASV columns found for diversity analysis.")
    
    with tab4:
        st.markdown("#### 🌍 Environmental Analysis")
        
        if os.path.exists(NC_PATH):
            try:
                ds_env = load_environmental_data()
                if ds_env is not None:
                    st.markdown("**Environmental Data Variables**")
                    varnames = list(ds_env.data_vars)
                    
                    selected_var = st.selectbox("Select variable to analyze", varnames)
                    
                    if selected_var:
                        var_data = ds_env[selected_var]
                        st.write(f"**{selected_var}** - Shape: {var_data.shape}")
                        st.write(f"Dimensions: {list(var_data.dims)}")
                        
                        # Show statistics
                        st.write("**Statistics:**")
                        st.write(f"Min: {float(var_data.min()):.4f}")
                        st.write(f"Max: {float(var_data.max()):.4f}")
                        st.write(f"Mean: {float(var_data.mean()):.4f}")
                        st.write(f"Std: {float(var_data.std()):.4f}")
                        
                        # Create visualization
                        if "time" in var_data.dims:
                            # Time series
                            time_series = var_data.mean(dim=["lat", "lon"]).values
                            time_axis = var_data.time.values
                            
                            fig_time = px.line(
                                x=time_axis,
                                y=time_series,
                                title=f"{selected_var} Time Series (Spatial Average)"
                            )
                            st.plotly_chart(fig_time, use_container_width=True)
                        else:
                            # Spatial plot
                            if "lat" in var_data.dims and "lon" in var_data.dims:
                                fig_spatial = px.imshow(
                                    var_data.values,
                                    title=f"{selected_var} Spatial Distribution",
                                    color_continuous_scale="viridis"
                                )
                                st.plotly_chart(fig_spatial, use_container_width=True)
            except Exception as e:
                st.warning(f"Failed to load environmental data: {e}")
        else:
            st.info("No environmental NetCDF file found.")

elif page == "ℹ️ About":
    st.title("ℹ️ About the Marine Data Analytics Platform")
    
    st.markdown("""
    ## 🐟 Marine Data Analytics Platform v2.0
    
    A comprehensive dashboard for analyzing marine biological data, integrating multiple data types:
    
    ### 📊 Data Types
    - **Morphological Data**: Otolith images and measurements
    - **Molecular Data**: ASV (Amplicon Sequence Variant) counts
    - **Environmental Data**: NetCDF environmental variables
    - **Metadata**: Sample collection information, coordinates, dates
    
    ### 🚀 Features
    - **Interactive Dashboard**: Real-time data visualization and analysis
    - **Advanced Filtering**: Multi-dimensional data filtering and search
    - **Geographic Mapping**: Interactive maps with customizable visualization
    - **Statistical Analysis**: Comprehensive statistical tools and diversity metrics
    - **Data Export**: Multiple export formats (CSV, Excel, JSON)
    - **Sample Analysis**: Detailed individual sample examination
    
    ### 🔧 Technical Stack
    - **Frontend**: Streamlit with Plotly visualizations
    - **Data Processing**: Pandas, NumPy, XArray
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Data Storage**: Parquet, SQLite, NetCDF
    
    ### 📁 Data Pipeline
    1. **ETL Phase 1**: Data ingestion and validation (`etl_phase1.py`)
    2. **ETL Phase 2**: Data integration and processing (`etl_phase2_integration.py`)
    3. **Dashboard**: Interactive analysis and visualization
    
    ### 🚀 Deployment
    This application is designed for deployment on Streamlit Cloud:
    1. Push code to GitHub repository
    2. Connect repository to Streamlit Cloud
    3. Deploy with automatic dependency management
    
    ### 📞 Support
    For technical support or questions about the platform, please refer to the documentation or contact the development team.
    """)
    
    # System information
    st.markdown("### 🔧 System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**App Version**: {APP_VERSION}")
        st.markdown(f"**Python Version**: {sys.version.split()[0]}")
        st.markdown(f"**Streamlit Version**: {st.__version__}")
    
    with col2:
        st.markdown(f"**Pandas Version**: {pd.__version__}")
        st.markdown(f"**NumPy Version**: {np.__version__}")
        st.markdown(f"**Plotly Version**: {px.__version__}")
    
    # Data status
    if df is not None:
        summary = get_data_summary(df)
        st.markdown("### 📊 Current Dataset Status")
        st.json(summary)
    else:
        st.warning("No dataset currently loaded.")

# End of enhanced application
