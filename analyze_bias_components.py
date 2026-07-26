#!/usr/bin/env python3
"""Decompose a fitted two-parameter gaze bias on the untouched evaluation split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from personalization_mechanism_analysis import angular_errors_deg, apply_bias, record_key
from research_suite_config import load_config
from validate_personalization_promotion import bootstrap_lower_bound


COMPONENTS = {
    "zero": (0, 0),
    "yaw_only": (1, 0),
    "pitch_only": (0, 1),
    "full": (1, 1),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def component_gains(
    predictions: np.ndarray,
    targets: np.ndarray,
    bias_deg: Iterable[float],
) -> dict[str, dict[str, float]]:
    bias = np.deg2rad(np.asarray(list(bias_deg), dtype=np.float64))
    if bias.shape != (2,):
        raise ValueError(f"Expected [yaw, pitch] bias, got shape {bias.shape}")
    base = float(angular_errors_deg(predictions, targets).mean())
    output = {}
    for name, mask in COMPONENTS.items():
        selected = bias * np.asarray(mask)
        error = base
        if name != "zero":
            corrected = apply_bias({"bias": selected}, predictions)
            error = float(angular_errors_deg(corrected, targets).mean())
        output[name] = {"error_deg": error, "gain_deg": base - error}
    return output


def load_evaluation(result_dir: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    summary = json.loads((result_dir / "summary.json").read_text())
    manifest = json.loads((result_dir / "split_manifest.json").read_text())
    cache_dir = Path(summary["config"]["cache_dir"])
    output = {}
    for protocol, subjects in manifest.items():
        for sid, split in subjects.items():
            if split.get("status") == "skipped":
                continue
            payload = torch.load(
                cache_dir / f"{sid}.pt", map_location="cpu", weights_only=False
            )
            index = {record_key(record): i for i, record in enumerate(payload["records"])}
            indices = [index[record_key(record)] for record in split["evaluation"]]
            output[(protocol, sid)] = (
                payload["predictions"][indices].float().numpy(),
                payload["targets"][indices].float().numpy(),
            )
    return output


def analyze_result_dir(
    suite: str, result_dir: Path, fold: int, tolerance_deg: float
) -> list[dict[str, Any]]:
    evaluations = load_evaluation(result_dir)
    output = []
    for row in read_csv(result_dir / "results.csv"):
        if row["method"] != "bias":
            continue
        key = (row["protocol"], row["sid"])
        parameters = json.loads(row["adapter_parameters_json"])
        bias_deg = parameters["bias_deg"]
        gains = component_gains(*evaluations[key], bias_deg)
        reported_full_gain = float(row["improvement_mean_deg"])
        consistency_error = abs(gains["full"]["gain_deg"] - reported_full_gain)
        if consistency_error > tolerance_deg:
            raise ValueError(
                f"{result_dir} {key} K={row['calibration_size']} repeat={row['repeat']}: "
                f"full gain mismatch {consistency_error:.8f} deg"
            )
        for component, metrics in gains.items():
            output.append(
                {
                    "suite": suite,
                    "result_dir": str(result_dir),
                    "fold": fold,
                    "protocol": row["protocol"],
                    "sid": row["sid"],
                    "calibration_size": int(row["calibration_size"]),
                    "repeat": int(row["repeat"]),
                    "selection_seed": int(row["selection_seed"]),
                    "component": component,
                    "yaw_bias_deg": float(bias_deg[0]),
                    "pitch_bias_deg": float(bias_deg[1]),
                    "base_error_deg": gains["zero"]["error_deg"],
                    "component_error_deg": metrics["error_deg"],
                    "gain_deg": metrics["gain_deg"],
                    "reported_full_gain_deg": reported_full_gain,
                    "full_consistency_error_deg": consistency_error,
                }
            )
    return output


def subject_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["suite"],
            row["protocol"],
            row["calibration_size"],
            row["sid"],
            row["component"],
        )
        groups[key].append(row["gain_deg"])
    return [
        {
            "suite": key[0],
            "protocol": key[1],
            "calibration_size": key[2],
            "sid": key[3],
            "component": key[4],
            "repeats": len(values),
            "mean_gain_deg": float(np.mean(values)),
        }
        for key, values in sorted(groups.items())
    ]


def aggregate_subjects(
    rows: list[dict[str, Any]], confidence_level: float, bootstrap_repeats: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[(row["suite"], row["protocol"], row["calibration_size"], row["sid"])][
            row["component"]
        ] = row["mean_gain_deg"]

    summaries, increments = [], []
    configurations = sorted({key[:3] for key in grouped})
    for suite, protocol, calibration_size in configurations:
        subjects = [
            values
            for key, values in grouped.items()
            if key[:3] == (suite, protocol, calibration_size)
        ]
        for component in COMPONENTS:
            values = np.asarray([item[component] for item in subjects])
            seed_key = f"{suite}/{protocol}/{calibration_size}/{component}"
            summaries.append(
                {
                    "suite": suite,
                    "protocol": protocol,
                    "calibration_size": calibration_size,
                    "component": component,
                    "subjects": len(values),
                    "mean_gain_deg": float(values.mean()),
                    "median_gain_deg": float(np.median(values)),
                    "positive_subject_rate": float(np.mean(values > 0.0)),
                    "gain_confidence_lower_bound_deg": bootstrap_lower_bound(
                        values, confidence_level, bootstrap_repeats, seed_key
                    ),
                }
            )
        comparisons = (
            ("full_vs_yaw_only", "full", "yaw_only"),
            ("full_vs_pitch_only", "full", "pitch_only"),
        )
        for name, stronger, reduced in comparisons:
            values = np.asarray(
                [item[stronger] - item[reduced] for item in subjects]
            )
            increments.append(
                {
                    "suite": suite,
                    "protocol": protocol,
                    "calibration_size": calibration_size,
                    "comparison": name,
                    "subjects": len(values),
                    "mean_increment_deg": float(values.mean()),
                    "positive_subject_rate": float(np.mean(values > 0.0)),
                    "increment_confidence_lower_bound_deg": bootstrap_lower_bound(
                        values,
                        confidence_level,
                        bootstrap_repeats,
                        f"{suite}/{protocol}/{calibration_size}/{name}",
                    ),
                }
            )
    return summaries, increments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--consistency-tolerance-deg", type=float, default=3e-4)
    args = parser.parse_args()

    rows = []
    for suite, _, result_dirs in load_config(args.config):
        for fold, result_dir in enumerate(result_dirs):
            rows.extend(
                analyze_result_dir(
                    suite, result_dir, fold, args.consistency_tolerance_deg
                )
            )
    subjects = subject_summary(rows)
    summaries, increments = aggregate_subjects(
        subjects, args.confidence_level, args.bootstrap_repeats
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "bias_component_rows.csv", rows)
    write_csv(args.output_dir / "bias_component_subject_summary.csv", subjects)
    write_csv(args.output_dir / "bias_component_summary.csv", summaries)
    write_csv(args.output_dir / "bias_component_increment_summary.csv", increments)
    payload = {
        "config": str(args.config),
        "confidence_level": args.confidence_level,
        "bootstrap_repeats": args.bootstrap_repeats,
        "rows": len(rows),
        "subjects": len({(r["suite"], r["protocol"], r["sid"]) for r in rows}),
        "max_full_consistency_error_deg": max(
            row["full_consistency_error_deg"] for row in rows
        ),
        "summary": summaries,
        "increment_summary": increments,
    }
    (args.output_dir / "bias_component_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "rows": payload["rows"],
                "subjects": payload["subjects"],
                "max_full_consistency_error_deg": payload[
                    "max_full_consistency_error_deg"
                ],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
