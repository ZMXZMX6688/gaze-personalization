#!/usr/bin/env python3
"""Identify subject-specific gaze bias before comparing compensation models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from personalize_from_universal import vector_to_angles


def wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def angular_errors_deg(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    predictions = predictions / np.linalg.norm(predictions, axis=1, keepdims=True)
    targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)
    cosine = np.clip(np.sum(predictions * targets, axis=1), -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def vectors_to_angles(vectors: np.ndarray) -> np.ndarray:
    return vector_to_angles(torch.from_numpy(vectors).float()).numpy()


def angles_to_vectors(angles: np.ndarray) -> np.ndarray:
    yaw, pitch = angles[:, 0], angles[:, 1]
    cosine_pitch = np.cos(pitch)
    return np.stack(
        (np.sin(yaw) * cosine_pitch, np.sin(pitch), np.cos(yaw) * cosine_pitch),
        axis=1,
    )


def fit_bias(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    pred_angles, target_angles = vectors_to_angles(predictions), vectors_to_angles(targets)
    residual = target_angles - pred_angles
    residual[:, 0] = wrap_angle(residual[:, 0])
    bias = residual.mean(axis=0)
    return {"bias": bias}


def apply_bias(model: dict[str, Any], predictions: np.ndarray) -> np.ndarray:
    angles = vectors_to_angles(predictions) + model["bias"]
    angles[:, 0] = wrap_angle(angles[:, 0])
    return angles_to_vectors(angles)


def fit_rotation(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    cross_covariance = targets.T @ predictions
    u, _, vt = np.linalg.svd(cross_covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    matrix = u @ correction @ vt
    return {"matrix": matrix}


def apply_rotation(model: dict[str, Any], predictions: np.ndarray) -> np.ndarray:
    return predictions @ model["matrix"].T


def fit_affine(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    pred_angles, target_angles = vectors_to_angles(predictions), vectors_to_angles(targets)
    target_angles[:, 0] = pred_angles[:, 0] + wrap_angle(
        target_angles[:, 0] - pred_angles[:, 0]
    )
    design = np.column_stack((pred_angles, np.ones(len(pred_angles))))
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, target_angles, rcond=None
    )
    return {
        "matrix": coefficients[:2].T,
        "bias": coefficients[2],
        "design_rank": int(rank),
        "design_condition": float(singular_values[0] / singular_values[-1]),
    }


def apply_affine(model: dict[str, Any], predictions: np.ndarray) -> np.ndarray:
    angles = vectors_to_angles(predictions) @ model["matrix"].T + model["bias"]
    angles[:, 0] = wrap_angle(angles[:, 0])
    return angles_to_vectors(angles)


MODELS = {
    "bias": (2, fit_bias, apply_bias),
    "rotation": (3, fit_rotation, apply_rotation),
    "affine": (6, fit_affine, apply_affine),
}


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return record["sid"], record["segment_id"], record["start"], record["target_frame"]


def rotation_angle_deg(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def bias_context_diagnostics(
    calibration_predictions: np.ndarray,
    calibration_targets: np.ndarray,
    evaluation_predictions: np.ndarray,
    evaluation_targets: np.ndarray,
) -> dict[str, Any]:
    cal_pred = vectors_to_angles(calibration_predictions)
    eval_pred = vectors_to_angles(evaluation_predictions)
    cal_target = vectors_to_angles(calibration_targets)
    eval_target = vectors_to_angles(evaluation_targets)
    cal_residual = cal_target - cal_pred
    eval_residual = eval_target - eval_pred
    cal_residual[:, 0] = wrap_angle(cal_residual[:, 0])
    eval_residual[:, 0] = wrap_angle(eval_residual[:, 0])
    cal_bias = cal_residual.mean(axis=0)
    eval_bias = eval_residual.mean(axis=0)
    lower, upper = cal_pred.min(axis=0), cal_pred.max(axis=0)
    outside = np.any((eval_pred < lower) | (eval_pred > upper), axis=1)
    return {
        "calibration_bias_deg_json": json.dumps(
            np.rad2deg(cal_bias).tolist(), separators=(",", ":")
        ),
        "evaluation_oracle_bias_deg_json": json.dumps(
            np.rad2deg(eval_bias).tolist(), separators=(",", ":")
        ),
        "bias_drift_deg": float(np.linalg.norm(np.rad2deg(eval_bias - cal_bias))),
        "prediction_centroid_shift_deg": float(
            np.linalg.norm(np.rad2deg(eval_pred.mean(axis=0) - cal_pred.mean(axis=0)))
        ),
        "evaluation_outside_calibration_box_rate": float(outside.mean()),
    }


def diagnose_model(
    method: str,
    calibration_predictions: np.ndarray,
    calibration_targets: np.ndarray,
    evaluation_predictions: np.ndarray,
    evaluation_targets: np.ndarray,
) -> dict[str, Any]:
    parameter_count, fit, apply = MODELS[method]
    model = fit(calibration_predictions, calibration_targets)
    calibration_base = angular_errors_deg(calibration_predictions, calibration_targets)
    calibration_after = angular_errors_deg(
        apply(model, calibration_predictions), calibration_targets
    )
    evaluation_base = angular_errors_deg(evaluation_predictions, evaluation_targets)
    evaluation_after = angular_errors_deg(apply(model, evaluation_predictions), evaluation_targets)
    row: dict[str, Any] = {
        "method": method,
        "parameter_count": parameter_count,
        "calibration_clips": len(calibration_predictions),
        "evaluation_clips": len(evaluation_predictions),
        "calibration_base_deg": float(calibration_base.mean()),
        "calibration_compensated_deg": float(calibration_after.mean()),
        "calibration_explained_fraction": float(
            1.0 - calibration_after.mean() / calibration_base.mean()
        ),
        "evaluation_base_deg": float(evaluation_base.mean()),
        "evaluation_compensated_deg": float(evaluation_after.mean()),
        "evaluation_gain_deg": float(evaluation_base.mean() - evaluation_after.mean()),
        "parameters_json": "",
        "orthogonality_error": "",
        "determinant": "",
        "condition_number": "",
        **bias_context_diagnostics(
            calibration_predictions,
            calibration_targets,
            evaluation_predictions,
            evaluation_targets,
        ),
    }
    if method == "bias":
        row["parameters_json"] = json.dumps(
            {"yaw_bias_deg": math.degrees(model["bias"][0]),
             "pitch_bias_deg": math.degrees(model["bias"][1])},
            separators=(",", ":"),
        )
    elif method == "rotation":
        matrix = model["matrix"]
        row.update(
            {
                "parameters_json": json.dumps(
                    {"rotation_angle_deg": rotation_angle_deg(matrix),
                     "matrix": matrix.tolist()},
                    separators=(",", ":"),
                ),
                "orthogonality_error": float(np.linalg.norm(matrix.T @ matrix - np.eye(3))),
                "determinant": float(np.linalg.det(matrix)),
                "condition_number": float(np.linalg.cond(matrix)),
            }
        )
    else:
        matrix = model["matrix"]
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        row.update(
            {
                "parameters_json": json.dumps(
                    {"matrix": matrix.tolist(),
                     "bias_deg": np.rad2deg(model["bias"]).tolist(),
                     "design_rank": model["design_rank"],
                     "design_condition": model["design_condition"],
                     "singular_values": singular_values.tolist()},
                    separators=(",", ":"),
                ),
                "orthogonality_error": float(np.linalg.norm(matrix.T @ matrix - np.eye(2))),
                "determinant": float(np.linalg.det(matrix)),
                "condition_number": float(np.linalg.cond(matrix)),
            }
        )
    return row


def analyze_result_dir(
    fold_dir: Path, protocols: set[str], fold_id: int
) -> list[dict[str, Any]]:
    summary = json.loads((fold_dir / "summary.json").read_text())
    cache_dir = Path(summary["config"]["cache_dir"])
    manifest = json.loads((fold_dir / "split_manifest.json").read_text())
    rows: list[dict[str, Any]] = []
    for protocol, subjects in manifest.items():
        if protocol not in protocols:
            continue
        for sid, split in subjects.items():
            if split.get("status") == "skipped":
                continue
            payload = torch.load(cache_dir / f"{sid}.pt", map_location="cpu", weights_only=False)
            index = {record_key(record): i for i, record in enumerate(payload["records"])}
            calibration_indices = [index[record_key(record)] for record in split["calibration_pool"]]
            evaluation_indices = [index[record_key(record)] for record in split["evaluation"]]
            predictions = payload["predictions"].float().numpy()
            targets = payload["targets"].float().numpy()
            for method in MODELS:
                row = diagnose_model(
                    method,
                    predictions[calibration_indices],
                    targets[calibration_indices],
                    predictions[evaluation_indices],
                    targets[evaluation_indices],
                )
                row = {"fold": fold_id, "protocol": protocol, "sid": sid, **row}
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for protocol in sorted({row["protocol"] for row in rows}):
        for method in MODELS:
            selected = [row for row in rows if row["protocol"] == protocol and row["method"] == method]
            gains = np.asarray([row["evaluation_gain_deg"] for row in selected])
            output.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "parameter_count": MODELS[method][0],
                    "subjects": len(selected),
                    "mean_calibration_explained_fraction": float(np.mean(
                        [row["calibration_explained_fraction"] for row in selected]
                    )),
                    "mean_evaluation_gain_deg": float(gains.mean()),
                    "median_evaluation_gain_deg": float(np.median(gains)),
                    "positive_subject_rate": float(np.mean(gains > 0.001)),
                    "gain_p05_deg": float(np.percentile(gains, 5)),
                    "gain_p95_deg": float(np.percentile(gains, 95)),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocols", default="chronological,interleaved")
    parser.add_argument(
        "--result-dirs",
        type=lambda value: [Path(item) for item in value.split(",") if item],
        default=None,
        help="Explicit comma-separated benchmark result directories",
    )
    args = parser.parse_args()
    protocols = {item.strip() for item in args.protocols.split(",") if item.strip()}
    fold_dirs = args.result_dirs or sorted(args.cv_dir.glob("fold*-personalization"))
    if not fold_dirs:
        parser.error(f"No fold personalization directories in {args.cv_dir}")
    rows = [
        row
        for fold_id, fold_dir in enumerate(fold_dirs)
        for row in analyze_result_dir(fold_dir, protocols, fold_id)
    ]
    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mechanism_subjects.csv", rows)
    write_csv(args.output_dir / "mechanism_summary.csv", summaries)
    (args.output_dir / "mechanism_summary.json").write_text(
        json.dumps({"models": {
            "bias": {"parameters": 2, "geometry": "yaw/pitch translation"},
            "rotation": {"parameters": 3, "geometry": "SO(3), orthogonal, det=+1"},
            "affine": {"parameters": 6, "geometry": "2D tangent affine, non-orthogonal"},
        }, "summary": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
