#!/usr/bin/env python3
"""
ETL Phase 1 - CSV ingestion, validation, and cleaned output

Place this script in the root of your VS Code project and run it from there.

Primary features:
 - Reads samples CSV and optional ASV table, images folder and NetCDF path
 - Validates required columns and basic rules (lat/lon ranges, date parsing, non-negative depth)
 - Normalizes missing values ("NA", "NULL", "-999", etc.)
 - Produces:
    * cleaned_samples.csv (valid rows)
    * invalid_samples.csv (rows flagged for review, with reasons)
    * provenance.json (processing metadata + summary)
    * optional: writes cleaned data into a SQLite DB table `samples`
 - Optional: basic GBIF name reconciliation (controlled by --taxon-validate). This requires `pygbif` or internet access and is OFF by default.

Usage example:
 python etl_phase1.py --input data/samples.csv --asv data/asv_table.tsv --images data/images --nc data/env_data.nc --output data/processed --sqlite data/processed/mvp.db --enforce-acceptance

"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime , timezone

import numpy as np
import pandas as pd

# ---------- Configuration / defaults ----------
REQUIRED_COLS = [
    "sample_id",
    "collection_date",
    "latitude",
    "longitude",
    "depth_m",
    "species",
    "image_ref",
    "asv_table_ref",
]

MISSING_VALUE_MARKERS = ["", "NA", "N/A", "NULL", "-999", "-9999"]
ACCEPTANCE_DEFAULT = 0.95

# ---------- Helpers ----------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv(path):
    logging.info(f"Reading CSV: {path}")
    df = pd.read_csv(path, dtype=str)
    return df


def coerce_types_and_normalize(df):
    # Normalize column names
    df = df.rename(columns=lambda c: c.strip())

    # Replace common missing markers with NaN
    df = df.replace(MISSING_VALUE_MARKERS, np.nan)

    # Ensure expected columns exist (case-insensitive match)
    cols_lower = {c.lower(): c for c in df.columns}
    for rc in REQUIRED_COLS:
        if rc not in df.columns and rc.lower() in cols_lower:
            df = df.rename(columns={cols_lower[rc.lower()]: rc})

    # Convert numeric fields
    for col in ["latitude", "longitude", "depth_m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse dates
    if "collection_date" in df.columns:
        df["collection_date_parsed"] = pd.to_datetime(
            df["collection_date"], errors="coerce"
        )

    return df


def validate_dataframe(df, images_dir=None):
    """Return (valid_df, invalid_df, diagnostics)
    invalid_df includes a column `invalid_reasons` listing issues for each row.
    """
    reasons = []
    df = df.copy()

    # Track reasons per row in a list, then join
    reason_lists = [[] for _ in range(len(df))]

    # Required column check
    for col in REQUIRED_COLS:
        if col not in df.columns:
            logging.error(f"Missing required column: {col}")
            raise RuntimeError(f"Missing required column: {col}")

    # sample_id presence
    missing_id_mask = df["sample_id"].isna() | (df["sample_id"].str.strip() == "")
    for i in df.index[missing_id_mask]:
        reason_lists[i].append("missing_sample_id")

    # coordinates
    lat_ok = df["latitude"].between(-90, 90)
    lon_ok = df["longitude"].between(-180, 180)
    for i in df.index[~lat_ok.fillna(False)]:
        reason_lists[i].append("invalid_latitude")
    for i in df.index[~lon_ok.fillna(False)]:
        reason_lists[i].append("invalid_longitude")

    # date parse
    date_ok = ~df["collection_date_parsed"].isna()
    for i in df.index[~date_ok]:
        reason_lists[i].append("invalid_collection_date")

    # depth
    depth_invalid = df["depth_m"].notna() & (df["depth_m"] < 0)
    for i in df.index[depth_invalid]:
        reason_lists[i].append("invalid_depth")

    # image availability (non-fatal): mark missing images but do not drop
    if images_dir is not None:
        image_avail = []
        for i, row in df.iterrows():
            img_ref = row.get("image_ref")
            if pd.isna(img_ref):
                image_avail.append(False)
                reason_lists[i].append("image_missing")
            else:
                # Build path relative to images_dir
                img_path = os.path.join(images_dir, os.path.basename(img_ref))
                if os.path.exists(img_path):
                    image_avail.append(True)
                else:
                    image_avail.append(False)
                    reason_lists[i].append("image_not_found")
        df["image_available"] = image_avail

    # duplicate sample_id check (warning)
    dup_mask = df["sample_id"].duplicated(keep=False)
    for i in df.index[dup_mask]:
        reason_lists[i].append("duplicate_sample_id")

    # Combine reasons
    df["invalid_reasons"] = [";".join(r) if r else "" for r in reason_lists]

    # Rows with any of the severe reasons are invalid; define severe reasons
    severe_reasons = set([
        "missing_sample_id",
        "invalid_latitude",
        "invalid_longitude",
        "invalid_collection_date",
        "invalid_depth",
    ])

    invalid_mask = df["invalid_reasons"].apply(lambda s: any(r in severe_reasons for r in s.split(";") if r))

    valid_df = df[~invalid_mask].copy()
    invalid_df = df[invalid_mask].copy()

    # Diagnostics
    diagnostics = {
        "total_rows": len(df),
        "valid_rows": len(valid_df),
        "invalid_rows": len(invalid_df),
    }
    # count reasons
    reason_counts = {}
    for rstr in df["invalid_reasons"]:
        if rstr:
            for r in rstr.split(";"):
                reason_counts[r] = reason_counts.get(r, 0) + 1
    diagnostics["reason_counts"] = reason_counts

    return valid_df, invalid_df, diagnostics


def save_outputs(valid_df, invalid_df, diagnostics, output_dir, sqlite_path=None):
    ensure_dir(output_dir)
    cleaned_csv = os.path.join(output_dir, "cleaned_samples.csv")
    invalid_csv = os.path.join(output_dir, "invalid_samples.csv")
    prov_json = os.path.join(output_dir, "provenance.json")

    logging.info(f"Writing cleaned rows: {cleaned_csv}")
    valid_df.to_csv(cleaned_csv, index=False)

    logging.info(f"Writing invalid rows: {invalid_csv}")
    invalid_df.to_csv(invalid_csv, index=False)

    # Optionally write to sqlite
    if sqlite_path:
        ensure_dir(os.path.dirname(sqlite_path) or "./")
        logging.info(f"Writing cleaned data to sqlite db: {sqlite_path}")
        conn = sqlite3.connect(sqlite_path)
        try:
            valid_df.to_sql("samples", conn, if_exists="replace", index=False)
        finally:
            conn.close()

    # provenance
    prov = {
        "script": os.path.basename(__file__),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.splitlines()[0],
        "pandas_version": pd.__version__,
        "diagnostics": diagnostics,
    }
    with open(prov_json, "w") as fh:
        json.dump(prov, fh, indent=2)

    logging.info("Saved outputs and provenance.")
    return {
        "cleaned_csv": cleaned_csv,
        "invalid_csv": invalid_csv,
        "provenance": prov_json,
        "sqlite": sqlite_path,
    }


def read_asv_table(path):
    if not path:
        return None
    if not os.path.exists(path):
        logging.warning(f"ASV table not found at {path}")
        return None
    logging.info(f"Reading ASV table: {path}")
    try:
        asv = pd.read_csv(path, sep="\t", dtype=str)
        return asv
    except Exception:
        logging.exception("Failed to read ASV table")
        return None


# Optional GBIF reconciliation (very lightweight; requires `pygbif`)
# The function tries to import pygbif and if it fails it will return an empty mapping.

def reconcile_with_gbif(names):
    mapping = {}
    try:
        from pygbif import species

        for name in names:
            if pd.isna(name) or name.strip() == "":
                mapping[name] = {"match": None}
                continue
            try:
                res = species.name_backbone(name=name)
                mapping[name] = res
            except Exception as e:
                mapping[name] = {"error": str(e)}
    except Exception:
        logging.warning(
            "pygbif not available or failed. Install pygbif to enable GBIF reconciliation. Skipping."
        )
    return mapping


# ---------- CLI / main ----------

def parse_args():
    p = argparse.ArgumentParser(description="ETL Phase 1 - ingest & validate samples.csv")
    p.add_argument("--input", default="data/samples.csv", help="Path to samples CSV")
    p.add_argument("--asv", default="data/asv_table.tsv", help="Path to ASV table (tsv)")
    p.add_argument("--images", default="data/images", help="Images directory")
    p.add_argument("--nc", default=None, help="NetCDF path (optional)")
    p.add_argument("--output", default="data/processed", help="Output directory for cleaned files")
    p.add_argument("--sqlite", default=None, help="Optional sqlite db path to write cleaned table")
    p.add_argument(
        "--enforce-acceptance",
        action="store_true",
        help=f"If set, exit non-zero when valid row fraction < {ACCEPTANCE_DEFAULT}",
    )
    p.add_argument(
        "--acceptance-threshold",
        type=float,
        default=ACCEPTANCE_DEFAULT,
        help="Fraction of rows that must pass validation (0-1)",
    )
    p.add_argument(
        "--taxon-validate",
        action="store_true",
        help="If set, attempt GBIF name reconciliation (requires pygbif)",
    )
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    ensure_dir(args.output)

    # Read CSV
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    df = read_csv(args.input)

    # Coerce types, normalize
    df = coerce_types_and_normalize(df)

    # Basic validation
    valid_df, invalid_df, diagnostics = validate_dataframe(df, images_dir=args.images)

    logging.info(
        f"Validation summary: {diagnostics['valid_rows']} valid / {diagnostics['total_rows']} total"
    )

    # Read ASV table (just for presence check and summary)
    asv = read_asv_table(args.asv)
    if asv is not None:
        # quick check: how many sample_ids match
        if "sample_id" in asv.columns:
            shared = set(valid_df["sample_id"]).intersection(set(asv["sample_id"]))
            logging.info(f"ASV table contains counts for {len(shared)} samples (within cleaned set)")
        else:
            logging.warning("ASV table does not contain a `sample_id` column; skipping join checks.")

    # Optional: taxon reconciliation
    taxon_map_path = None
    if args.taxon_validate:
        names = valid_df["species"].dropna().unique().tolist()
        logging.info(f"Reconciling {len(names)} unique species names via GBIF (may be slow)")
        mapping = reconcile_with_gbif(names)
        # save mapping to file
        taxon_map_path = os.path.join(args.output, "taxon_map.json")
        with open(taxon_map_path, "w") as fh:
            json.dump(mapping, fh, indent=2)
        logging.info(f"Saved taxon_map to {taxon_map_path}")

    # Save outputs
    outputs = save_outputs(valid_df, invalid_df, diagnostics, args.output, sqlite_path=args.sqlite)

    # Acceptance criteria enforcement
    valid_frac = diagnostics["valid_rows"] / max(1, diagnostics["total_rows"])
    logging.info(f"Valid fraction: {valid_frac:.3f}")

    if valid_frac < args.acceptance_threshold:
        logging.warning(
            f"Valid fraction {valid_frac:.3f} is below acceptance threshold {args.acceptance_threshold:.3f}"
        )
        if args.enforce_acceptance:
            logging.error("Enforcement requested; exiting with error due to acceptance failure.")
            sys.exit(2)

    logging.info("ETL completed successfully.")
    logging.info(f"Outputs: {outputs}")


if __name__ == "__main__":
    main()
