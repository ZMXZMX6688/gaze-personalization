#!/usr/bin/env python3
"""Visualize checkpoint × protocol × K personalization transfer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_gain_matrix(
    rows: list[dict],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    suites = list(dict.fromkeys(row["suite"] for row in rows))
    columns = sorted({
        (row["protocol"], int(row["calibration_size"])) for row in rows
    })
    protocol_labels = {"chronological": "Chron.", "interleaved": "Inter."}
    labels = [
        f"{protocol_labels.get(protocol, protocol)}\nK={size}"
        for protocol, size in columns
    ]
    lookup = {
        (row["suite"], row["protocol"], int(row["calibration_size"])):
            float(row["mean_gain_deg"])
        for row in rows
    }
    status_lookup = {
        (row["suite"], row["protocol"], int(row["calibration_size"])):
            row["status"] == "PASS"
        for row in rows
    }
    matrix = np.asarray([
        [lookup[(suite, protocol, size)] for protocol, size in columns]
        for suite in suites
    ])
    passed = np.asarray([
        [status_lookup[(suite, protocol, size)] for protocol, size in columns]
        for suite in suites
    ])
    return suites, labels, matrix, passed


def plot_matrix(
    suites: list[str],
    labels: list[str],
    matrix: np.ndarray,
    passed: np.ndarray,
    output_dir: Path,
) -> None:
    limit = max(float(np.abs(matrix).max()), 0.01)
    figure, axis = plt.subplots(
        figsize=(max(8.0, len(labels) * 1.45), max(4.0, len(suites) * 0.72))
    )
    image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(labels)), labels=labels)
    axis.set_yticks(np.arange(len(suites)), labels=suites)
    axis.set_xlabel("Protocol and calibration size")
    axis.set_ylabel("Validation suite")
    axis.set_title("Frozen reliability-bias transfer gain (degrees)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if abs(value) > limit * 0.55 else "black"
            axis.text(
                column, row,
                f"{value:+.3f}\n{'PASS' if passed[row, column] else 'FAIL'}",
                ha="center", va="center", color=color, fontsize=8,
                fontweight="bold" if passed[row, column] else "normal",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Baseline − personalized error (°)")
    figure.tight_layout()
    figure.savefig(output_dir / "checkpoint_validation_matrix.svg")
    figure.savefig(
        output_dir / "checkpoint_validation_matrix.png", dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.promotion_report.read_text())
    suites, labels, matrix, passed = build_gain_matrix(report["rows"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_matrix(suites, labels, matrix, passed, args.output_dir)
    with (args.output_dir / "checkpoint_validation_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["suite", *labels])
        writer.writerows(
            [suite, *[f"{value:.9f}" for value in values]]
            for suite, values in zip(suites, matrix)
        )
    with (args.output_dir / "checkpoint_validation_status.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["suite", *labels])
        writer.writerows(
            [suite, *["PASS" if value else "FAIL" for value in values]]
            for suite, values in zip(suites, passed)
        )
    print(
        f"Wrote {len(suites)} × {len(labels)} validation matrix to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
