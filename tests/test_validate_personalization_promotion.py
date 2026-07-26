import csv
import json

from validate_personalization_promotion import validate_suites


def _summary(path, gain):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": [{
        "protocol": "chronological",
        "method": "bias",
        "calibration_size": 20,
        "subjects": 6,
        "mean_improvement_deg": gain,
    }]}))
    with (path.parent / "subject_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "protocol", "method", "calibration_size", "sid",
            "mean_improvement_deg",
        ])
        writer.writeheader()
        for index in range(10):
            writer.writerow({
                "protocol": "chronological",
                "method": "bias",
                "calibration_size": 20,
                "sid": f"S{index}",
                "mean_improvement_deg": gain,
            })
    return path


def test_promotion_requires_every_independent_suite_to_pass(tmp_path):
    strong = _summary(tmp_path / "strong" / "summary.json", 0.05)
    shifted = _summary(tmp_path / "shifted" / "summary.json", -0.02)
    report = validate_suites(
        [("strong", strong), ("shifted", shifted)],
        "bias", {"chronological"}, {20}, 0.0,
    )
    assert report["status"] == "FAIL"
    assert report["failed_configurations"] == 1


def test_promotion_passes_when_all_suites_meet_threshold(tmp_path):
    first = _summary(tmp_path / "first" / "summary.json", 0.05)
    second = _summary(tmp_path / "second" / "summary.json", 0.01)
    report = validate_suites(
        [("first", first), ("second", second)],
        "bias", {"chronological"}, {20}, 0.0,
    )
    assert report["status"] == "PASS"


def test_positive_mean_fails_when_subject_bootstrap_lower_bound_crosses_zero(tmp_path):
    summary = _summary(tmp_path / "unstable" / "summary.json", 0.01)
    with (summary.parent / "subject_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "protocol", "method", "calibration_size", "sid",
            "mean_improvement_deg",
        ])
        writer.writeheader()
        for index, gain in enumerate([-0.2, -0.1, 0.0, 0.1, 0.25]):
            writer.writerow({
                "protocol": "chronological", "method": "bias",
                "calibration_size": 20, "sid": f"S{index}",
                "mean_improvement_deg": gain,
            })
    report = validate_suites(
        [("unstable", summary)], "bias", {"chronological"}, {20}, 0.0,
    )
    assert report["rows"][0]["mean_gain_deg"] > 0.0
    assert report["rows"][0]["gain_confidence_lower_bound_deg"] < 0.0
    assert report["status"] == "FAIL"
