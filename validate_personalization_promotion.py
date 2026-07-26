#!/usr/bin/env python3
"""Block promotion when personalization fails an independent validation suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_summary(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return payload["summary"]


def load_subject_gains(
    summary_path: Path, method: str, protocol: str, calibration_size: int
) -> np.ndarray:
    path = summary_path.parent / "subject_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing paired subject results: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        gains = [
            float(row["mean_improvement_deg"])
            for row in csv.DictReader(handle)
            if row["method"] == method
            and row["protocol"] == protocol
            and int(row["calibration_size"]) == calibration_size
        ]
    if not gains:
        raise ValueError(
            f"No subject gains for {method}/{protocol}/K={calibration_size} in {path}"
        )
    return np.asarray(gains)


def bootstrap_lower_bound(
    values: np.ndarray,
    confidence_level: float,
    repeats: int,
    seed_key: str,
) -> float:
    seed = int.from_bytes(
        hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    return float(np.quantile(values[indices].mean(axis=1), 1.0 - confidence_level))


def validate_suites(
    suites: list[tuple[str, Path]],
    method: str,
    protocols: set[str],
    calibration_sizes: set[int],
    minimum_gain_deg: float,
    confidence_level: float = 0.95,
    bootstrap_repeats: int = 10000,
) -> dict[str, Any]:
    rows = []
    for suite_name, path in suites:
        matching = [
            row for row in load_summary(path)
            if row["method"] == method
            and row["protocol"] in protocols
            and int(row["calibration_size"]) in calibration_sizes
        ]
        expected = len(protocols) * len(calibration_sizes)
        if len(matching) != expected:
            raise ValueError(
                f"{suite_name}: expected {expected} configurations, found {len(matching)}"
            )
        for row in matching:
            gain = float(row["mean_improvement_deg"])
            protocol = row["protocol"]
            calibration_size = int(row["calibration_size"])
            subject_gains = load_subject_gains(
                path, method, protocol, calibration_size
            )
            lower_bound = bootstrap_lower_bound(
                subject_gains,
                confidence_level,
                bootstrap_repeats,
                f"{suite_name}/{method}/{protocol}/{calibration_size}",
            )
            passed = gain >= minimum_gain_deg and lower_bound >= minimum_gain_deg
            rows.append(
                {
                    "suite": suite_name,
                    "summary_path": str(path),
                    "protocol": protocol,
                    "method": method,
                    "calibration_size": calibration_size,
                    "subjects": int(row["subjects"]),
                    "mean_gain_deg": gain,
                    "gain_confidence_lower_bound_deg": lower_bound,
                    "confidence_level": confidence_level,
                    "bootstrap_repeats": bootstrap_repeats,
                    "minimum_gain_deg": minimum_gain_deg,
                    "status": "PASS" if passed else "FAIL",
                }
            )
    failed = [row for row in rows if row["status"] == "FAIL"]
    return {
        "status": "PASS" if not failed else "FAIL",
        "policy": {
            "method": method,
            "protocols": sorted(protocols),
            "calibration_sizes": sorted(calibration_sizes),
            "minimum_gain_deg": minimum_gain_deg,
            "confidence_level": confidence_level,
            "bootstrap_repeats": bootstrap_repeats,
            "requires_subject_bootstrap_lower_bound": True,
            "all_suites_must_pass": True,
        },
        "rows": rows,
        "failed_configurations": len(failed),
    }


def parse_suite(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected NAME=/path/to/summary.json") from error
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", type=parse_suite, required=True)
    parser.add_argument("--method", default="bias")
    parser.add_argument("--protocols", default="chronological,interleaved")
    parser.add_argument("--calibration-sizes", default="10,20,50")
    parser.add_argument("--minimum-gain-deg", type=float, default=0.0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_suites(
        args.suite,
        args.method,
        {value for value in args.protocols.split(",") if value},
        {int(value) for value in args.calibration_sizes.split(",") if value},
        args.minimum_gain_deg,
        args.confidence_level,
        args.bootstrap_repeats,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "promotion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "promotion_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    print(
        f"{report['status']}: {report['failed_configurations']} configurations failed"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
