#!/usr/bin/env python3
"""Explain pitch-only success through calibration-to-evaluation bias transport."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from personalization_mechanism_analysis import record_key, vectors_to_angles
from research_suite_config import load_config


def pitch_residual_deg(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.rad2deg(
        vectors_to_angles(targets)[:, 1] - vectors_to_angles(predictions)[:, 1]
    )


def transport_metrics(learned_pitch_deg: float, oracle_residuals_deg: np.ndarray) -> dict:
    oracle = float(np.mean(oracle_residuals_deg))
    drift = learned_pitch_deg - oracle
    sign_match = (
        float(np.sign(learned_pitch_deg) == np.sign(oracle))
        if learned_pitch_deg != 0 and oracle != 0 else 0.0
    )
    return {
        "evaluation_oracle_pitch_deg": oracle,
        "pitch_transport_drift_deg": abs(drift),
        "signed_pitch_transport_error_deg": drift,
        "pitch_sign_match": sign_match,
        "pitch_overshoot": float(
            abs(learned_pitch_deg) > abs(oracle) and sign_match == 1.0
        ),
        "evaluation_pitch_residual_std_deg": float(np.std(oracle_residuals_deg)),
    }


def evaluation_residuals(result_dir: Path) -> dict:
    summary = json.loads((result_dir / "summary.json").read_text())
    manifest = json.loads((result_dir / "split_manifest.json").read_text())
    cache_dir = Path(summary["config"]["cache_dir"])
    output = {}
    for protocol, subjects in manifest.items():
        for sid, split in subjects.items():
            if split.get("status") == "skipped":
                continue
            payload = torch.load(cache_dir / f"{sid}.pt", map_location="cpu", weights_only=False)
            index = {record_key(r): i for i, r in enumerate(payload["records"])}
            positions = [index[record_key(r)] for r in split["evaluation"]]
            output[(protocol, sid)] = pitch_residual_deg(
                payload["predictions"][positions].float().numpy(),
                payload["targets"][positions].float().numpy(),
            )
    return output


def correlation(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pitch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    detailed = []
    for suite, _, original_dirs in load_config(args.config):
        for fold, original_dir in enumerate(original_dirs):
            residuals = evaluation_residuals(original_dir)
            path = args.pitch_root / suite / f"fold{fold}" / "results.csv"
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    parameter = json.loads(row["adapter_parameters_json"])["pitch_bias_deg"]
                    learned = float(parameter)
                    metrics = transport_metrics(learned, residuals[(row["protocol"], row["sid"])])
                    detailed.append({
                        "suite": suite, "fold": fold, "protocol": row["protocol"],
                        "sid": row["sid"], "calibration_size": int(row["calibration_size"]),
                        "repeat": int(row["repeat"]), "learned_pitch_deg": learned,
                        "gain_deg": float(row["improvement_mean_deg"]), **metrics,
                    })
    groups = defaultdict(list)
    for row in detailed:
        groups[(row["suite"], row["protocol"], row["calibration_size"], row["sid"])].append(row)
    subjects = []
    for key, values in sorted(groups.items()):
        subjects.append({
            "suite": key[0], "protocol": key[1], "calibration_size": key[2], "sid": key[3],
            **{name: float(np.mean([r[name] for r in values])) for name in (
                "learned_pitch_deg", "evaluation_oracle_pitch_deg",
                "pitch_transport_drift_deg", "pitch_sign_match", "pitch_overshoot",
                "evaluation_pitch_residual_std_deg", "gain_deg")},
        })
    summary = {
        "subject_configurations": len(subjects),
        "gain_vs_transport_drift_correlation": correlation(
            [r["pitch_transport_drift_deg"] for r in subjects],
            [r["gain_deg"] for r in subjects]),
        "gain_vs_sign_match_correlation": correlation(
            [r["pitch_sign_match"] for r in subjects], [r["gain_deg"] for r in subjects]),
        "mean_gain_sign_match_deg": float(np.mean(
            [r["gain_deg"] for r in subjects if r["pitch_sign_match"] >= .5])),
        "mean_gain_sign_mismatch_deg": float(np.mean(
            [r["gain_deg"] for r in subjects if r["pitch_sign_match"] < .5])),
        "sign_match_rate": float(np.mean([r["pitch_sign_match"] for r in subjects])),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("pitch_transport_rows.csv", detailed),
                          ("pitch_transport_subject_summary.csv", subjects)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    (args.output_dir / "pitch_transport_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
