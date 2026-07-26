import csv
import json

from audit_cv_integrity import audit_cv


def _write_fixture(tmp_path, leak=False):
    split = {
        "subjects": 2,
        "n_folds": 2,
        "folds": [
            {"fold": 0, "train_sids": ["B"], "val_sids": [], "test_sids": ["A"]},
            {"fold": 1, "train_sids": ["A"], "val_sids": [], "test_sids": ["B"]},
        ],
    }
    split_path = tmp_path / "folds.json"
    split_path.write_text(json.dumps(split))
    for fold, sid in enumerate(("A", "B")):
        artifact = tmp_path / f"fold{fold}-personalization"
        artifact.mkdir()
        calibration_segment = 1
        evaluation_segment = 1 if leak and fold == 0 else 3
        clip = lambda segment: {
            "sid": sid,
            "segment_id": segment,
            "start": segment * 10,
            "target_frame": segment * 10 + 5,
        }
        manifest = {
            "chronological": {
                sid: {
                    "calibration_pool": [clip(calibration_segment)],
                    "evaluation": [clip(evaluation_segment)],
                    "calibration_segment_ids": [calibration_segment],
                    "evaluation_segment_ids": [evaluation_segment],
                }
            }
        }
        (artifact / "split_manifest.json").write_text(json.dumps(manifest))
        with (artifact / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["protocol", "method", "sid", "calibration_size", "repeat"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "protocol": "chronological",
                    "method": "bias",
                    "sid": sid,
                    "calibration_size": 1,
                    "repeat": 0,
                }
            )
    return split_path


def test_audit_passes_complete_isolated_cv(tmp_path):
    split_path = _write_fixture(tmp_path)
    report = audit_cv(split_path, tmp_path)
    assert report["status"] == "PASS"
    assert report["test_subjects_once"] == 2


def test_audit_detects_segment_leakage(tmp_path):
    split_path = _write_fixture(tmp_path, leak=True)
    report = audit_cv(split_path, tmp_path)
    assert report["status"] == "FAIL"
    assert any("segment leakage" in error for error in report["errors"])


def test_audit_accepts_documented_skipped_subject(tmp_path):
    split_path = _write_fixture(tmp_path)
    artifact = tmp_path / "fold0-personalization"
    manifest = json.loads((artifact / "split_manifest.json").read_text())
    manifest["chronological"]["A"] = {
        "status": "skipped",
        "feasible_calibration_sizes": [],
        "reason": "insufficient clips",
    }
    (artifact / "split_manifest.json").write_text(json.dumps(manifest))
    with (artifact / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protocol", "method", "sid", "calibration_size", "repeat"],
        )
        writer.writeheader()
    report = audit_cv(split_path, tmp_path)
    assert report["status"] == "PASS"
    assert report["folds"][0]["skipped_protocol_subject_rows"] == 1
