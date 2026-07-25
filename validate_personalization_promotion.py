#!/usr/bin/env python3
"""Block promotion when personalization fails an independent validation suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return payload["summary"]


def validate_suites(
    suites: list[tuple[str, Path]],
    method: str,
    protocols: set[str],
    calibration_sizes: set[int],
    minimum_gain_deg: float,
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
            rows.append(
                {
                    "suite": suite_name,
                    "summary_path": str(path),
                    "protocol": row["protocol"],
                    "method": method,
                    "calibration_size": int(row["calibration_size"]),
                    "subjects": int(row["subjects"]),
                    "mean_gain_deg": gain,
                    "minimum_gain_deg": minimum_gain_deg,
                    "status": "PASS" if gain >= minimum_gain_deg else "FAIL",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_suites(
        args.suite,
        args.method,
        {value for value in args.protocols.split(",") if value},
        {int(value) for value in args.calibration_sizes.split(",") if value},
        args.minimum_gain_deg,
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
