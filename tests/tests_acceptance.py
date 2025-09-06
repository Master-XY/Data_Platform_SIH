# tests/test_acceptance.py
import os
import json
import pandas as pd
import pytest

ORIG = "data/samples.csv"
CLEANED = "data/processed/cleaned_samples.csv"
UNIFIED = "data/processed/unified.parquet"
TAXON_MAP = "data/processed/taxon_map.json"
NC_PATH = "data/env_data.nc"

def load_dataframe():
    if os.path.exists(CLEANED):
        return pd.read_csv(CLEANED, dtype=str)
    if os.path.exists(UNIFIED):
        return pd.read_parquet(UNIFIED)
    pytest.skip("No cleaned or unified dataset found; run ETL first")

def test_valid_fraction():
    orig = pd.read_csv(ORIG, dtype=str)
    df = load_dataframe()
    frac = len(df) / max(1, len(orig))
    assert frac >= 0.95, f"Valid fraction too low: {frac:.3f}"

def test_species_validation():
    # Skip if no taxon_map (optional feature)
    if not os.path.exists(TAXON_MAP):
        pytest.skip("No taxon_map.json — species validation not enabled")
    with open(TAXON_MAP) as fh:
        mapping = json.load(fh)
    df = load_dataframe()
    names = df.get("species", pd.Series()).dropna().unique().tolist()
    if not names:
        pytest.skip("No species names present")
    matched = sum(1 for n in names if n in mapping and mapping.get(n))
    frac = matched / max(1, len(names))
    assert frac >= 0.90, f"Species auto-match fraction too low: {frac:.2f}"

def test_image_availability():
    df = load_dataframe()
    if "image_available" not in df.columns:
        pytest.skip("image_available column not present; skip image checks")
    avail = df["image_available"].astype(bool).sum()
    frac = avail / max(1, len(df))
    # threshold: at least 20% of samples should have images in MVP; adjust if needed
    assert frac >= 0.20, f"Too few images available: {frac:.2f}"
    # ensure flagged thumbnails exist
    missing = []
    for p, flag in zip(df.get("thumbnail_path", []), df["image_available"]):
        if flag:
            if not (isinstance(p, str) and os.path.exists(p)):
                missing.append(p)
    assert not missing, f"thumbnail_path flagged but files missing: {missing[:5]}"

def test_netcdf_loading():
    try:
        import xarray as xr
    except Exception:
        pytest.skip("xarray not installed; skip NetCDF tests")
    assert os.path.exists(NC_PATH), "env_data.nc not found"
    ds = xr.open_dataset(NC_PATH)
    assert len(ds.data_vars) > 0, "NetCDF contains no data variables"
