#!/usr/bin/env python3
"""Aggregate completed subject-disjoint personalization folds."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from personalization_benchmark import (
    aggregate_results,
    aggregate_subject_results,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.cv_root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: List[Dict[str, object]] = []
    fold_configs = []
    test_subjects = []
    for fold in range(args.folds):
        fold_dir = args.cv_root / f"fold{fold}-personalization"
        results_path = fold_dir / "results.csv"
        summary_path = fold_dir / "summary.json"
        if not results_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Incomplete personalization fold: {fold_dir}")
        with results_path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                converted: Dict[str, object] = dict(row)
                converted["fold"] = fold
                for key in (
                    "calibration_size", "repeat", "selection_seed", "evaluation_clips",
                    "calibration_fit_clips", "calibration_validation_clips",
                ):
                    converted[key] = int(converted[key])
                for key in (
                    "adapter_scale", "calibration_validation_gain_deg", "base_mean_deg",
                    "base_median_deg", "base_p90_deg", "personalized_mean_deg",
                    "personalized_median_deg", "personalized_p90_deg", "improvement_mean_deg",
                ):
                    converted[key] = float(converted[key])
                rows.append(converted)
        with summary_path.open("r") as handle:
            payload = json.load(handle)
        fold_configs.append(payload["config"])
        test_subjects.extend(payload["config"]["held_out_test_sids"])

    if len(test_subjects) != len(set(test_subjects)):
        raise AssertionError("A test subject appears in more than one fold")
    if len(test_subjects) != 56:
        raise AssertionError(f"Expected 56 unique test subjects, found {len(test_subjects)}")

    summary_rows = aggregate_results(rows, expected_subjects=len(test_subjects))
    subject_rows = aggregate_subject_results(rows)
    fold_rows = []
    for fold in range(args.folds):
        selected = [row for row in rows if row["fold"] == fold]
        fold_subjects = len(fold_configs[fold]["held_out_test_sids"])
        for row in aggregate_results(selected, expected_subjects=fold_subjects):
            fold_rows.append({"fold": fold, **row})

    write_csv(output_dir / "results_all_folds.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "subject_summary.csv", subject_rows)
    write_csv(output_dir / "fold_summary.csv", fold_rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump({
            "subjects": len(test_subjects),
            "folds": args.folds,
            "rows": len(rows),
            "fold_configs": fold_configs,
            "summary": summary_rows,
        }, handle, indent=2)
    print(json.dumps({
        "output_dir": str(output_dir),
        "subjects": len(test_subjects),
        "rows": len(rows),
        "best": sorted(
            summary_rows,
            key=lambda row: row["mean_improvement_deg"],
            reverse=True,
        )[:10],
    }, indent=2))


if __name__ == "__main__":
    main()
