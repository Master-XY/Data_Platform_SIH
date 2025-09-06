#!/usr/bin/env python3
"""
Generate acceptance testing report for Phase 4.
"""

import json
from pathlib import Path
import pandas as pd

# Paths
processed_dir = Path("data/processed")
samples_file = processed_dir / "cleaned_samples.csv"
invalid_file = processed_dir / "invalid_samples.csv"
unified_file = processed_dir / "unified.parquet"
report_file = processed_dir / "acceptance_report.json"

def main():
    report = {}

    # 1. Check sample load success
    if samples_file.exists():
        df = pd.read_csv(samples_file)
        report["total_samples"] = len(df)
    else:
        report["total_samples"] = 0

    if invalid_file.exists():
        df_invalid = pd.read_csv(invalid_file)
        report["invalid_samples"] = len(df_invalid)
    else:
        report["invalid_samples"] = 0

    if report["total_samples"] > 0:
        valid_fraction = (report["total_samples"] /
                          (report["total_samples"] + report["invalid_samples"]))
        report["valid_fraction"] = round(valid_fraction, 3)
        report["criterion_valid_fraction"] = report["valid_fraction"] >= 0.95
    else:
        report["valid_fraction"] = 0
        report["criterion_valid_fraction"] = False

    # 2. Check unified dataset
    report["unified_exists"] = unified_file.exists()

    # 3. Missing image handling
    if unified_file.exists():
        df_unified = pd.read_parquet(unified_file)
        if "image_available" in df_unified.columns:
            missing = (~df_unified["image_available"]).sum()
            report["missing_images"] = int(missing)
        else:
            report["missing_images"] = None
    else:
        report["missing_images"] = None

    # Save report
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Acceptance report written to {report_file}")

if __name__ == "__main__":
    main()
