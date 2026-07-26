#!/usr/bin/env python3
"""Compare mechanism diagnostics and attribute personalization failures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_vector(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def classify_transport(
    cosine: float,
    magnitude_ratio: float,
    attenuation_threshold: float,
    amplification_threshold: float,
) -> str:
    if cosine < 0.0:
        return "direction_reversal"
    if magnitude_ratio < attenuation_threshold:
        return "magnitude_attenuation"
    if magnitude_ratio > amplification_threshold:
        return "magnitude_amplification"
    return "approximately_preserved"


def load_suite(
    suite: str,
    path: Path,
    attenuation_threshold: float,
    amplification_threshold: float,
) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for row in source:
        if row["method"] != "bias":
            continue
        calibration_bias = parse_vector(row["calibration_bias_deg_json"])
        evaluation_bias = parse_vector(row["evaluation_oracle_bias_deg_json"])
        calibration_magnitude = float(np.linalg.norm(calibration_bias))
        evaluation_magnitude = float(np.linalg.norm(evaluation_bias))
        cosine = float(
            calibration_bias @ evaluation_bias
            / max(calibration_magnitude * evaluation_magnitude, 1e-12)
        )
        magnitude_ratio = evaluation_magnitude / max(calibration_magnitude, 1e-12)
        rows.append(
            {
                "suite": suite,
                "fold": int(row["fold"]),
                "protocol": row["protocol"],
                "sid": row["sid"],
                "calibration_bias_magnitude_deg": calibration_magnitude,
                "evaluation_bias_magnitude_deg": evaluation_magnitude,
                "evaluation_to_calibration_magnitude_ratio": magnitude_ratio,
                "bias_direction_cosine": cosine,
                "bias_drift_deg": float(row["bias_drift_deg"]),
                "prediction_centroid_shift_deg":
                    float(row["prediction_centroid_shift_deg"]),
                "evaluation_outside_calibration_box_rate":
                    float(row["evaluation_outside_calibration_box_rate"]),
                "calibration_explained_fraction":
                    float(row["calibration_explained_fraction"]),
                "evaluation_gain_deg": float(row["evaluation_gain_deg"]),
                "transport_class": classify_transport(
                    cosine,
                    magnitude_ratio,
                    attenuation_threshold,
                    amplification_threshold,
                ),
            }
        )
    if not rows:
        raise ValueError(f"{suite}: no bias diagnostics in {path}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for suite in sorted({row["suite"] for row in rows}):
        for protocol in sorted({row["protocol"] for row in rows if row["suite"] == suite}):
            selected = [
                row for row in rows
                if row["suite"] == suite and row["protocol"] == protocol
            ]
            gains = np.asarray([row["evaluation_gain_deg"] for row in selected])
            ratios = np.asarray([
                row["evaluation_to_calibration_magnitude_ratio"] for row in selected
            ])
            classes = {
                label: float(np.mean([row["transport_class"] == label for row in selected]))
                for label in (
                    "direction_reversal",
                    "magnitude_attenuation",
                    "magnitude_amplification",
                    "approximately_preserved",
                )
            }
            summaries.append(
                {
                    "suite": suite,
                    "protocol": protocol,
                    "subjects": len(selected),
                    "mean_calibration_bias_magnitude_deg": float(np.mean([
                        row["calibration_bias_magnitude_deg"] for row in selected
                    ])),
                    "mean_evaluation_bias_magnitude_deg": float(np.mean([
                        row["evaluation_bias_magnitude_deg"] for row in selected
                    ])),
                    "mean_magnitude_ratio": float(ratios.mean()),
                    "median_magnitude_ratio": float(np.median(ratios)),
                    "mean_bias_direction_cosine": float(np.mean([
                        row["bias_direction_cosine"] for row in selected
                    ])),
                    "mean_bias_drift_deg": float(np.mean([
                        row["bias_drift_deg"] for row in selected
                    ])),
                    "mean_evaluation_gain_deg": float(gains.mean()),
                    "gain_to_magnitude_ratio_correlation": float(
                        np.corrcoef(gains, ratios)[0, 1]
                    ) if len(selected) > 1 else float("nan"),
                    **{f"{label}_rate": rate for label, rate in classes.items()},
                }
            )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_suite(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected NAME=/path/to/mechanism_subjects.csv") from error
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", type=parse_suite, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attenuation-threshold", type=float, default=0.75)
    parser.add_argument("--amplification-threshold", type=float, default=1.25)
    args = parser.parse_args()
    if not 0 < args.attenuation_threshold < 1:
        parser.error("--attenuation-threshold must be between 0 and 1")
    if args.amplification_threshold <= 1:
        parser.error("--amplification-threshold must exceed 1")
    rows = [
        row
        for name, path in args.suite
        for row in load_suite(
            name, path, args.attenuation_threshold, args.amplification_threshold
        )
    ]
    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "failure_attribution_subjects.csv", rows)
    write_csv(args.output_dir / "failure_attribution_summary.csv", summaries)
    (args.output_dir / "failure_attribution.json").write_text(
        json.dumps(
            {
                "thresholds": {
                    "attenuation": args.attenuation_threshold,
                    "amplification": args.amplification_threshold,
                    "classification_is_descriptive_not_a_runtime_gate": True,
                },
                "summary": summaries,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
