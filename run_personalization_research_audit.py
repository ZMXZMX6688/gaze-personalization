#!/usr/bin/env python3
"""Run mechanism, attribution, and promotion audits on completed benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from attribute_personalization_failure import (
    load_suite as load_attribution_suite,
    summarize as summarize_attribution,
)
from personalization_mechanism_analysis import (
    analyze_result_dir,
    summarize as summarize_mechanisms,
)
from plot_checkpoint_validation_matrix import build_gain_matrix, plot_matrix
from validate_personalization_promotion import validate_suites


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_suite(value: str) -> tuple[str, Path, list[Path]]:
    try:
        name, payload = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected NAME=/benchmark/dir or NAME=/summary.json::/result/dir1,/result/dir2"
        ) from error
    if "::" in payload:
        summary, result_dirs = payload.split("::", 1)
        directories = [Path(item) for item in result_dirs.split(",") if item]
        if not directories:
            raise argparse.ArgumentTypeError("Suite must contain at least one result directory")
        return name, Path(summary), directories
    directory = Path(payload)
    return name, directory / "summary.json", [directory]


def load_config(path: Path) -> list[tuple[str, Path, list[Path]]]:
    payload = json.loads(path.read_text())
    return [
        (
            suite["name"],
            Path(suite["summary_path"]),
            [Path(value) for value in suite["result_dirs"]],
        )
        for suite in payload["suites"]
    ]


def git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", type=parse_suite)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="bias")
    parser.add_argument("--protocols", default="chronological,interleaved")
    parser.add_argument("--calibration-sizes", default="10,20,50")
    parser.add_argument("--minimum-gain-deg", type=float, default=0.0)
    args = parser.parse_args()
    if bool(args.suite) == bool(args.config):
        parser.error("Provide exactly one of --suite or --config")
    suites = load_config(args.config) if args.config else args.suite
    protocols = {value for value in args.protocols.split(",") if value}
    calibration_sizes = {
        int(value) for value in args.calibration_sizes.split(",") if value
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)

    mechanism_rows = []
    inputs = []
    for suite_index, (name, summary_path, result_dirs) in enumerate(suites):
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        result_inputs = []
        for result_dir in result_dirs:
            manifest_path = result_dir / "split_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            result_inputs.append(
                {
                    "result_dir": str(result_dir),
                    "split_manifest_sha256": sha256(manifest_path),
                }
            )
        inputs.append(
            {
                "suite": name,
                "summary_path": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "results": result_inputs,
            }
        )
        for result_index, result_dir in enumerate(result_dirs):
            suite_rows = analyze_result_dir(
                result_dir, protocols, suite_index * 100 + result_index
            )
            mechanism_rows.extend({"suite": name, **row} for row in suite_rows)

    mechanism_dir = args.output_dir / "mechanism"
    mechanism_dir.mkdir()
    write_csv(mechanism_dir / "mechanism_subjects.csv", mechanism_rows)
    mechanism_summary = []
    for name, _, _ in suites:
        selected = [
            {key: value for key, value in row.items() if key != "suite"}
            for row in mechanism_rows if row["suite"] == name
        ]
        mechanism_summary.extend(
            {"suite": name, **row} for row in summarize_mechanisms(selected)
        )
    write_csv(mechanism_dir / "mechanism_summary.csv", mechanism_summary)

    attribution_rows = []
    # Reuse the just-written combined mechanism table by selecting suite rows.
    for name, _, _ in suites:
        suite_path = mechanism_dir / f"{name}_mechanism_subjects.csv"
        write_csv(
            suite_path,
            [
                {key: value for key, value in row.items() if key != "suite"}
                for row in mechanism_rows if row["suite"] == name
            ],
        )
        attribution_rows.extend(
            load_attribution_suite(name, suite_path, 0.75, 1.25)
        )
    attribution_summary = summarize_attribution(attribution_rows)
    attribution_dir = args.output_dir / "attribution"
    attribution_dir.mkdir()
    write_csv(attribution_dir / "failure_attribution_subjects.csv", attribution_rows)
    write_csv(attribution_dir / "failure_attribution_summary.csv", attribution_summary)

    promotion = validate_suites(
        [(name, summary_path) for name, summary_path, _ in suites],
        args.method,
        protocols,
        calibration_sizes,
        args.minimum_gain_deg,
    )
    promotion_dir = args.output_dir / "promotion"
    promotion_dir.mkdir()
    (promotion_dir / "promotion_report.json").write_text(
        json.dumps(promotion, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(promotion_dir / "promotion_report.csv", promotion["rows"])
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir()
    suites, labels, matrix = build_gain_matrix(promotion["rows"])
    plot_matrix(suites, labels, matrix, figure_dir)
    with (figure_dir / "checkpoint_validation_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["suite", *labels])
        writer.writerows(
            [suite, *[f"{value:.9f}" for value in values]]
            for suite, values in zip(suites, matrix)
        )

    project_root = Path(__file__).resolve().parent
    manifest = {
        "status": promotion["status"],
        "git_commit": git_commit(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "inputs": inputs,
        "outputs": {
            "mechanism_subjects_sha256":
                sha256(mechanism_dir / "mechanism_subjects.csv"),
            "failure_attribution_sha256":
                sha256(attribution_dir / "failure_attribution_summary.csv"),
            "promotion_report_sha256":
                sha256(promotion_dir / "promotion_report.json"),
            "validation_matrix_csv_sha256":
                sha256(figure_dir / "checkpoint_validation_matrix.csv"),
            "validation_matrix_svg_sha256":
                sha256(figure_dir / "checkpoint_validation_matrix.svg"),
        },
    }
    (args.output_dir / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{promotion['status']}: audited {len(suites)} suites; "
        f"{promotion['failed_configurations']} promotion failures"
    )
    return 0 if promotion["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
