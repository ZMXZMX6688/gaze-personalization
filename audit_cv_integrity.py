#!/usr/bin/env python3
"""Audit subject-disjoint CV splits and personalization artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _clip_key(clip: dict[str, Any]) -> tuple[Any, ...]:
    return clip["sid"], clip["segment_id"], clip["start"], clip["target_frame"]


def audit_cv(split_path: Path, cv_dir: Path) -> dict[str, Any]:
    split = _load_json(split_path)
    errors: list[str] = []
    warnings: list[str] = []
    fold_rows: list[dict[str, Any]] = []
    test_counts: Counter[str] = Counter()
    declared_subjects = int(split["subjects"])
    folds = split["folds"]

    if len(folds) != int(split["n_folds"]):
        errors.append("split fold count does not match n_folds")

    universe: set[str] = set()
    for fold in folds:
        fold_id = int(fold["fold"])
        train, val, test = (set(fold[f"{name}_sids"]) for name in ("train", "val", "test"))
        universe.update(train | val | test)
        test_counts.update(test)
        if train & val or train & test or val & test:
            errors.append(f"fold {fold_id}: train/val/test subjects overlap")
        if len(train | val | test) != declared_subjects:
            errors.append(f"fold {fold_id}: partition covers {len(train | val | test)} subjects")

        artifact_dir = cv_dir / f"fold{fold_id}-personalization"
        manifest_path = artifact_dir / "split_manifest.json"
        results_path = artifact_dir / "results.csv"
        if not manifest_path.exists() or not results_path.exists():
            errors.append(f"fold {fold_id}: missing personalization artifacts")
            continue

        manifest = _load_json(manifest_path)
        protocol_rows = 0
        evaluable_sids_by_protocol: dict[str, set[str]] = {}
        skipped_rows = 0
        for protocol, subjects in manifest.items():
            manifest_sids = set(subjects)
            if manifest_sids != test:
                errors.append(
                    f"fold {fold_id}/{protocol}: manifest subjects differ from test_sids"
                )
            for sid, entry in subjects.items():
                if entry.get("status") == "skipped":
                    skipped_rows += 1
                    if entry.get("feasible_calibration_sizes"):
                        errors.append(
                            f"fold {fold_id}/{protocol}/{sid}: skipped with feasible sizes"
                        )
                    if not entry.get("reason"):
                        errors.append(
                            f"fold {fold_id}/{protocol}/{sid}: skipped without reason"
                        )
                    continue
                evaluable_sids_by_protocol.setdefault(protocol, set()).add(sid)
                calibration = entry["calibration_pool"]
                evaluation = entry["evaluation"]
                calibration_segments = set(entry["calibration_segment_ids"])
                evaluation_segments = set(entry["evaluation_segment_ids"])
                if calibration_segments & evaluation_segments:
                    errors.append(f"fold {fold_id}/{protocol}/{sid}: segment leakage")
                calibration_keys = {_clip_key(clip) for clip in calibration}
                evaluation_keys = {_clip_key(clip) for clip in evaluation}
                if calibration_keys & evaluation_keys:
                    errors.append(f"fold {fold_id}/{protocol}/{sid}: clip leakage")
                if any(clip["sid"] != sid for clip in calibration + evaluation):
                    errors.append(f"fold {fold_id}/{protocol}/{sid}: clip SID mismatch")
                if {clip["segment_id"] for clip in calibration} != calibration_segments:
                    errors.append(
                        f"fold {fold_id}/{protocol}/{sid}: calibration segment index mismatch"
                    )
                if {clip["segment_id"] for clip in evaluation} != evaluation_segments:
                    errors.append(
                        f"fold {fold_id}/{protocol}/{sid}: evaluation segment index mismatch"
                    )
                protocol_rows += 1

        with results_path.open(newline="", encoding="utf-8") as handle:
            results = list(csv.DictReader(handle))
        for protocol in manifest:
            result_sids = {row["sid"] for row in results if row["protocol"] == protocol}
            expected_sids = evaluable_sids_by_protocol.get(protocol, set())
            if result_sids != expected_sids:
                errors.append(
                    f"fold {fold_id}/{protocol}: result subjects differ from evaluable manifest subjects"
                )
        result_protocols = {row["protocol"] for row in results}
        expected_result_protocols = {
            protocol for protocol, sids in evaluable_sids_by_protocol.items() if sids
        }
        if result_protocols != expected_result_protocols:
            errors.append(f"fold {fold_id}: result protocols differ from evaluable manifest")
        duplicate_keys = Counter(
            (
                row["protocol"],
                row["method"],
                row["sid"],
                row["calibration_size"],
                row["repeat"],
            )
            for row in results
        )
        duplicate_count = sum(count - 1 for count in duplicate_keys.values() if count > 1)
        if duplicate_count:
            errors.append(f"fold {fold_id}: {duplicate_count} duplicate result rows")

        fold_rows.append(
            {
                "fold": fold_id,
                "train_subjects": len(train),
                "validation_subjects": len(val),
                "test_subjects": len(test),
                "manifest_protocol_subject_rows": protocol_rows,
                "skipped_protocol_subject_rows": skipped_rows,
                "result_rows": len(results),
                "status": "PASS" if not any(e.startswith(f"fold {fold_id}") for e in errors) else "FAIL",
            }
        )

    if len(universe) != declared_subjects:
        errors.append(f"global subject universe has {len(universe)}, expected {declared_subjects}")
    missing = sorted(sid for sid in universe if test_counts[sid] == 0)
    repeated = sorted(sid for sid, count in test_counts.items() if count != 1)
    if missing:
        errors.append(f"{len(missing)} subjects never appear in a test fold")
    if repeated:
        errors.append(f"{len(repeated)} subjects do not appear exactly once in test folds")

    return {
        "status": "PASS" if not errors else "FAIL",
        "split": str(split_path),
        "cv_dir": str(cv_dir),
        "declared_subjects": declared_subjects,
        "observed_subjects": len(universe),
        "test_subjects_once": sum(count == 1 for count in test_counts.values()),
        "folds": fold_rows,
        "errors": errors,
        "warnings": warnings,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fields = [
        "fold",
        "train_subjects",
        "validation_subjects",
        "test_subjects",
        "manifest_protocol_subject_rows",
        "skipped_protocol_subject_rows",
        "result_rows",
        "status",
    ]
    with (output_dir / "integrity_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["folds"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_cv(args.split_json, args.cv_dir)
    write_report(report, args.output_dir)
    print(
        f"{report['status']}: {report['test_subjects_once']}/"
        f"{report['declared_subjects']} subjects assigned to a test fold exactly once; "
        f"{len(report['errors'])} errors"
    )
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
