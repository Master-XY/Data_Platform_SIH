# etl_phase2_integration.py
"""
Phase 2 integration skeleton.
Run:
python etl_phase2_integration.py --samples data/processed/cleaned_samples.csv --asv data/asv_table.tsv --images data/images --nc data/env_data.nc --output data/processed --sqlite data/processed/mvp.db
"""
import argparse, os, json
import pandas as pd
from pathlib import Path
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/processed/cleaned_samples.csv")
    p.add_argument("--asv", default="data/asv_table.tsv")
    p.add_argument("--images", default="data/images")
    p.add_argument("--nc", default="data/env_data.nc")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--sqlite", default=None)
    return p.parse_args()

def load_asv(asv_path):
    df = pd.read_csv(asv_path, sep="\t", dtype=str)
    # handle taxonomy-first-row if present
    first = df.iloc[0].tolist()
    if any(not str(x).replace('.','',1).isdigit() for x in first[1:]):
        taxa = df.iloc[0].to_dict()
        df = df.iloc[1:].copy()
    else:
        taxa = None
    df.loc[:, df.columns.difference(["sample_id"])] = df.loc[:, df.columns.difference(["sample_id"])].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    return df, taxa

def make_thumbnails(images_dir, thumb_dir, size=(400,400)):
    os.makedirs(thumb_dir, exist_ok=True)
    for img in Path(images_dir).glob("*"):
        out = Path(thumb_dir) / img.name
        if not out.exists():
            try:
                im = Image.open(img)
                im.thumbnail(size)
                im.save(out)
            except Exception as e:
                print("thumb fail", img, e)

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    samples = pd.read_csv(args.samples, dtype=str)
    asv_df, taxa = load_asv(args.asv)
    # merge (wide)
    merged = samples.merge(asv_df, on="sample_id", how="left")
    # image checks + thumbnails
    thumb_dir = os.path.join(args.output, "thumbnails")
    make_thumbnails(args.images, thumb_dir)
    merged["thumbnail_path"] = merged["image_ref"].apply(lambda p: os.path.join(thumb_dir, os.path.basename(p)) if pd.notna(p) else None)
    merged["image_available"] = merged["thumbnail_path"].apply(lambda p: os.path.exists(p) if p else False)
    # write unified parquet
    unified_path = os.path.join(args.output, "unified.parquet")
    merged.to_parquet(unified_path, index=False)
    # write simple report
    report = {
        "rows": len(merged),
        "images_available": int(merged["image_available"].sum())
    }
    with open(os.path.join(args.output, "integration_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("Wrote", unified_path, "report saved.")
if __name__ == "__main__":
    main()
