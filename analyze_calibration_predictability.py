#!/usr/bin/env python3
"""Test whether calibration-only signals predict cross-checkpoint transfer."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research_suite_config import load_config


FEATURE_NAMES = (
    "log_calibration_size",
    "is_interleaved",
    "parameter_magnitude_deg",
    "parameter_instability_deg",
    "reliability_scale",
)


def load_observations_from_suites(
    suites: list[tuple[str, Path, list[Path]]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for suite, _, result_dirs in suites:
        for result_dir in result_dirs:
            with (result_dir / "results.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                for row in csv.DictReader(handle):
                    if row["method"] != "bias" or row.get("gate_strategy") != "reliability":
                        continue
                    key = (
                        suite,
                        row["protocol"],
                        int(row["calibration_size"]),
                        row["sid"],
                    )
                    grouped[key].append(row)
    observations = []
    for (suite, protocol, calibration_size, sid), rows in sorted(grouped.items()):
        observations.append(
            {
                "suite": suite,
                "protocol": protocol,
                "calibration_size": calibration_size,
                "sid": sid,
                "repeats": len(rows),
                "parameter_magnitude_deg": float(np.mean([
                    float(row["parameter_magnitude_deg"]) for row in rows
                ])),
                "parameter_instability_deg": float(np.mean([
                    float(row["parameter_instability_deg"]) for row in rows
                ])),
                "reliability_scale": float(np.mean([
                    float(row["adapter_scale"]) for row in rows
                ])),
                "evaluation_gain_deg": float(np.mean([
                    float(row["improvement_mean_deg"]) for row in rows
                ])),
            }
        )
    if not observations:
        raise ValueError("No reliability-gated bias observations found")
    return observations


def load_observations(config_path: Path) -> list[dict[str, Any]]:
    return load_observations_from_suites(load_config(config_path))


def feature_vector(row: dict[str, Any]) -> list[float]:
    return [
        math.log(float(row["calibration_size"])),
        float(row["protocol"] == "interleaved"),
        float(row["parameter_magnitude_deg"]),
        float(row["parameter_instability_deg"]),
        float(row["reliability_scale"]),
    ]


def fit_ridge(
    features: np.ndarray, targets: np.ndarray, regularization: float = 1e-3
) -> dict[str, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack((np.ones(len(features)), standardized))
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict_ridge(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["mean"]) / model["scale"]
    design = np.column_stack((np.ones(len(features)), standardized))
    return design @ model["coefficients"]


def evaluate_leave_one_suite_out(
    observations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    suites = sorted({row["suite"] for row in observations})
    predictions = []
    fold_summaries = []
    for held_suite in suites:
        train = [row for row in observations if row["suite"] != held_suite]
        test = [row for row in observations if row["suite"] == held_suite]
        train_x = np.asarray([feature_vector(row) for row in train])
        train_y = np.asarray([row["evaluation_gain_deg"] for row in train])
        test_x = np.asarray([feature_vector(row) for row in test])
        test_y = np.asarray([row["evaluation_gain_deg"] for row in test])
        model = fit_ridge(train_x, train_y)
        predicted = predict_ridge(model, test_x)
        baseline = np.full(len(test), train_y.mean())
        for row, estimate, baseline_estimate in zip(test, predicted, baseline):
            predictions.append(
                {
                    **row,
                    "predicted_gain_deg": float(estimate),
                    "baseline_predicted_gain_deg": float(baseline_estimate),
                    "prediction_error_deg": float(estimate - row["evaluation_gain_deg"]),
                    "actual_positive": int(row["evaluation_gain_deg"] > 0.001),
                    "predicted_positive": int(estimate > 0.001),
                }
            )
        model_rmse = float(np.sqrt(np.mean((predicted - test_y) ** 2)))
        baseline_rmse = float(np.sqrt(np.mean((baseline - test_y) ** 2)))
        fold_summaries.append(
            {
                "held_out_suite": held_suite,
                "observations": len(test),
                "model_rmse_deg": model_rmse,
                "baseline_rmse_deg": baseline_rmse,
                "rmse_improvement_deg": baseline_rmse - model_rmse,
                "sign_accuracy": float(np.mean((predicted > 0.001) == (test_y > 0.001))),
                "actual_mean_gain_deg": float(test_y.mean()),
                "predicted_mean_gain_deg": float(predicted.mean()),
            }
        )
    return predictions, fold_summaries


def overall_summary(
    predictions: list[dict[str, Any]], folds: list[dict[str, Any]]
) -> dict[str, Any]:
    actual = np.asarray([row["evaluation_gain_deg"] for row in predictions])
    predicted = np.asarray([row["predicted_gain_deg"] for row in predictions])
    baseline = np.asarray([
        row["baseline_predicted_gain_deg"] for row in predictions
    ])
    return {
        "features": list(FEATURE_NAMES),
        "validation": "leave-one-checkpoint-suite-out",
        "observations": len(predictions),
        "suites": len(folds),
        "model_rmse_deg": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "baseline_rmse_deg": float(np.sqrt(np.mean((baseline - actual) ** 2))),
        "sign_accuracy": float(np.mean((predicted > 0.001) == (actual > 0.001))),
        "gain_correlation": float(np.corrcoef(predicted, actual)[0, 1]),
        "suites_with_better_rmse_than_baseline": int(np.sum([
            fold["rmse_improvement_deg"] > 0 for fold in folds
        ])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    observations = load_observations(args.config)
    predictions, folds = evaluate_leave_one_suite_out(observations)
    summary = overall_summary(predictions, folds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "calibration_observations.csv", observations)
    write_csv(args.output_dir / "leave_one_suite_out_predictions.csv", predictions)
    write_csv(args.output_dir / "leave_one_suite_out_summary.csv", folds)
    (args.output_dir / "predictability_summary.json").write_text(
        json.dumps({"summary": summary, "folds": folds}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
