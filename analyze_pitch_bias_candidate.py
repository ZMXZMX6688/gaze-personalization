#!/usr/bin/env python3
"""Pair pitch-only replay rows with the original two-parameter bias rows."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research_suite_config import load_config
from validate_personalization_promotion import bootstrap_lower_bound


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pitch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paired = defaultdict(lambda: defaultdict(list))
    for suite, _, sources in load_config(args.config):
        for fold, source in enumerate(sources):
            pitch = args.pitch_root / suite / f"fold{fold}" / "results.csv"
            original = {
                (r["protocol"], r["sid"], r["calibration_size"], r["repeat"]): r
                for r in rows(source / "results.csv") if r["method"] == "bias"
            }
            for row in rows(pitch):
                key = (row["protocol"], row["sid"], row["calibration_size"], row["repeat"])
                old = original[key]
                group = (suite, row["protocol"], int(row["calibration_size"]), row["sid"])
                paired[group]["pitch"].append(float(row["improvement_mean_deg"]))
                paired[group]["bias"].append(float(old["improvement_mean_deg"]))
    subjects = []
    for key, value in sorted(paired.items()):
        pitch, bias = np.mean(value["pitch"]), np.mean(value["bias"])
        subjects.append((*key, pitch, bias, pitch - bias))
    output = []
    for config in sorted({row[:3] for row in subjects}):
        selected = [row for row in subjects if row[:3] == config]
        gain = np.asarray([row[4] for row in selected])
        delta = np.asarray([row[6] for row in selected])
        output.append({
            "suite": config[0], "protocol": config[1], "calibration_size": config[2],
            "subjects": len(selected), "pitch_mean_gain_deg": float(gain.mean()),
            "pitch_gain_lower_bound_deg": bootstrap_lower_bound(
                gain, .95, 10000, f"{config}/pitch"),
            "pitch_vs_bias_mean_increment_deg": float(delta.mean()),
            "pitch_vs_bias_increment_lower_bound_deg": bootstrap_lower_bound(
                delta, .95, 10000, f"{config}/delta"),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(output[0])
    with (args.output / "pitch_candidate_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(output)
    payload = {
        "configurations": len(output),
        "positive_mean": sum(r["pitch_mean_gain_deg"] > 0 for r in output),
        "positive_lower_bound": sum(r["pitch_gain_lower_bound_deg"] > 0 for r in output),
        "better_than_bias_mean": sum(r["pitch_vs_bias_mean_increment_deg"] > 0 for r in output),
        "better_than_bias_lower_bound": sum(
            r["pitch_vs_bias_increment_lower_bound_deg"] > 0 for r in output),
        "rows": output,
    }
    (args.output / "pitch_candidate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
