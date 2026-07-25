#!/usr/bin/env python3
"""Measure whether a subject bias is stable enough to personalize."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from personalization_benchmark import stratified_sample_indices
from personalization_mechanism_analysis import (
    angular_errors_deg,
    apply_bias,
    fit_bias,
    record_key,
    vectors_to_angles,
    wrap_angle,
)


def estimate_bias(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return fit_bias(predictions, targets)["bias"]


def split_half_stability(
    predictions: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, float]:
    full_bias = estimate_bias(predictions, targets)
    first = np.arange(0, len(predictions), 2)
    second = np.arange(1, len(predictions), 2)
    if len(first) < 2 or len(second) < 2:
        raise ValueError("At least four calibration clips are required")
    first_bias = estimate_bias(predictions[first], targets[first])
    second_bias = estimate_bias(predictions[second], targets[second])
    difference = first_bias - second_bias
    difference[0] = wrap_angle(np.asarray(difference[0]))
    return full_bias, float(np.linalg.norm(np.rad2deg(difference)))


def stability_gate(
    bias: np.ndarray,
    split_half_difference_deg: float,
    max_split_half_difference_deg: float,
    signal_to_instability_ratio: float,
) -> tuple[bool, float]:
    magnitude_deg = float(np.linalg.norm(np.rad2deg(bias)))
    threshold = signal_to_instability_ratio * split_half_difference_deg
    active = (
        split_half_difference_deg <= max_split_half_difference_deg
        and magnitude_deg >= threshold
    )
    return active, magnitude_deg


def reliability_scale(bias: np.ndarray, split_half_difference_deg: float) -> float:
    """Estimate signal reliability from two independent half-sample estimates."""
    magnitude_sq = float(np.linalg.norm(np.rad2deg(bias))) ** 2
    if magnitude_sq <= 1e-12:
        return 0.0
    # Difference variance is twice a half-estimate variance; averaging both
    # halves divides that estimate variance by two, hence the factor 1/4.
    noise_sq = split_half_difference_deg ** 2 / 4.0
    return float(np.clip(1.0 - noise_sq / magnitude_sq, 0.0, 1.0))


def analyze_subject(
    predictions: np.ndarray,
    targets: np.ndarray,
    calibration_indices: list[int],
    evaluation_indices: list[int],
    calibration_sizes: list[int],
    repeats: int,
    seed_components: tuple[int, int, int],
    max_split_half_difference_deg: float,
    signal_to_instability_ratio: float,
) -> list[dict[str, Any]]:
    calibration_predictions = predictions[calibration_indices]
    calibration_targets = targets[calibration_indices]
    evaluation_predictions = predictions[evaluation_indices]
    evaluation_targets = targets[evaluation_indices]
    base_error = float(angular_errors_deg(evaluation_predictions, evaluation_targets).mean())
    oracle_bias = estimate_bias(evaluation_predictions, evaluation_targets)
    rows = []
    for calibration_size in calibration_sizes:
        if calibration_size > len(calibration_indices):
            continue
        for repeat in range(repeats):
            selection_seed = int(np.random.SeedSequence(
                [*seed_components, calibration_size, repeat]
            ).generate_state(1)[0])
            positions = stratified_sample_indices(
                len(calibration_indices), calibration_size, selection_seed
            )
            selected_predictions = calibration_predictions[positions]
            selected_targets = calibration_targets[positions]
            bias, instability = split_half_stability(
                selected_predictions, selected_targets
            )
            active, magnitude = stability_gate(
                bias,
                instability,
                max_split_half_difference_deg,
                signal_to_instability_ratio,
            )
            shrinkage_scale = reliability_scale(bias, instability)
            personalized = apply_bias({"bias": bias}, evaluation_predictions)
            shrunk = apply_bias(
                {"bias": bias * shrinkage_scale}, evaluation_predictions
            )
            personalized_error = float(
                angular_errors_deg(personalized, evaluation_targets).mean()
            )
            shrunk_error = float(angular_errors_deg(shrunk, evaluation_targets).mean())
            drift = bias - oracle_bias
            drift[0] = wrap_angle(np.asarray(drift[0]))
            rows.append(
                {
                    "calibration_size": calibration_size,
                    "repeat": repeat,
                    "selection_seed": selection_seed,
                    "yaw_bias_deg": float(np.rad2deg(bias[0])),
                    "pitch_bias_deg": float(np.rad2deg(bias[1])),
                    "bias_magnitude_deg": magnitude,
                    "split_half_difference_deg": instability,
                    "oracle_bias_drift_deg": float(np.linalg.norm(np.rad2deg(drift))),
                    "gate_active": int(active),
                    "reliability_scale": shrinkage_scale,
                    "base_error_deg": base_error,
                    "ungated_error_deg": personalized_error,
                    "ungated_gain_deg": base_error - personalized_error,
                    "gated_error_deg": personalized_error if active else base_error,
                    "gated_gain_deg": base_error - (personalized_error if active else base_error),
                    "shrunk_error_deg": shrunk_error,
                    "shrunk_gain_deg": base_error - shrunk_error,
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    keys = sorted({(row["protocol"], row["calibration_size"]) for row in rows})
    for protocol, calibration_size in keys:
        selected = [
            row for row in rows
            if row["protocol"] == protocol and row["calibration_size"] == calibration_size
        ]
        subject_ids = sorted({row["sid"] for row in selected})
        subject_metrics = []
        for sid in subject_ids:
            subject_rows = [row for row in selected if row["sid"] == sid]
            subject_metrics.append(
                {
                    "ungated": np.mean([row["ungated_gain_deg"] for row in subject_rows]),
                    "gated": np.mean([row["gated_gain_deg"] for row in subject_rows]),
                    "activation": np.mean([row["gate_active"] for row in subject_rows]),
                    "scale": np.mean([row["reliability_scale"] for row in subject_rows]),
                    "shrunk": np.mean([row["shrunk_gain_deg"] for row in subject_rows]),
                    "instability": np.mean(
                        [row["split_half_difference_deg"] for row in subject_rows]
                    ),
                    "parameter_std": float(np.sqrt(
                        np.var([row["yaw_bias_deg"] for row in subject_rows])
                        + np.var([row["pitch_bias_deg"] for row in subject_rows])
                    )),
                }
            )
        summaries.append(
            {
                "protocol": protocol,
                "calibration_size": calibration_size,
                "subjects": len(subject_ids),
                "mean_parameter_std_deg": float(np.mean(
                    [item["parameter_std"] for item in subject_metrics]
                )),
                "mean_split_half_difference_deg": float(np.mean(
                    [item["instability"] for item in subject_metrics]
                )),
                "activation_rate": float(np.mean(
                    [item["activation"] for item in subject_metrics]
                )),
                "mean_reliability_scale": float(np.mean(
                    [item["scale"] for item in subject_metrics]
                )),
                "ungated_macro_gain_deg": float(np.mean(
                    [item["ungated"] for item in subject_metrics]
                )),
                "gated_macro_gain_deg": float(np.mean(
                    [item["gated"] for item in subject_metrics]
                )),
                "shrunk_macro_gain_deg": float(np.mean(
                    [item["shrunk"] for item in subject_metrics]
                )),
                "ungated_subject_win_rate": float(np.mean(
                    [item["ungated"] > 0.001 for item in subject_metrics]
                )),
                "gated_subject_win_rate": float(np.mean(
                    [item["gated"] > 0.001 for item in subject_metrics]
                )),
                "shrunk_subject_win_rate": float(np.mean(
                    [item["shrunk"] > 0.001 for item in subject_metrics]
                )),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-sizes", default="10,20,50")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-split-half-difference-deg", type=float, default=0.5)
    parser.add_argument("--signal-to-instability-ratio", type=float, default=2.0)
    args = parser.parse_args()
    calibration_sizes = sorted({int(value) for value in args.calibration_sizes.split(",")})
    rows: list[dict[str, Any]] = []
    for fold_dir in sorted(args.cv_dir.glob("fold*-personalization")):
        fold_id = int(fold_dir.name.split("-")[0].replace("fold", ""))
        summary = json.loads((fold_dir / "summary.json").read_text())
        cache_dir = Path(summary["config"]["cache_dir"])
        manifest = json.loads((fold_dir / "split_manifest.json").read_text())
        for protocol_index, (protocol, subjects) in enumerate(manifest.items()):
            for sid_index, (sid, split) in enumerate(subjects.items()):
                if split.get("status") == "skipped":
                    continue
                payload = torch.load(
                    cache_dir / f"{sid}.pt", map_location="cpu", weights_only=False
                )
                index = {record_key(record): i for i, record in enumerate(payload["records"])}
                calibration_indices = [
                    index[record_key(record)] for record in split["calibration_pool"]
                ]
                evaluation_indices = [
                    index[record_key(record)] for record in split["evaluation"]
                ]
                subject_rows = analyze_subject(
                    payload["predictions"].float().numpy(),
                    payload["targets"].float().numpy(),
                    calibration_indices,
                    evaluation_indices,
                    calibration_sizes,
                    args.repeats,
                    (args.seed, fold_id, protocol_index * 100 + sid_index),
                    args.max_split_half_difference_deg,
                    args.signal_to_instability_ratio,
                )
                rows.extend(
                    {"fold": fold_id, "protocol": protocol, "sid": sid, **row}
                    for row in subject_rows
                )
    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "bias_stability_repeats.csv", rows)
    write_csv(args.output_dir / "bias_stability_summary.csv", summaries)
    (args.output_dir / "bias_stability_summary.json").write_text(
        json.dumps(
            {
                "gate": {
                    "max_split_half_difference_deg":
                        args.max_split_half_difference_deg,
                    "signal_to_instability_ratio": args.signal_to_instability_ratio,
                    "uses_evaluation_labels": False,
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
