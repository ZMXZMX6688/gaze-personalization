import json

from validate_personalization_promotion import validate_suites


def _summary(path, gain):
    path.write_text(json.dumps({"summary": [{
        "protocol": "chronological",
        "method": "bias",
        "calibration_size": 20,
        "subjects": 6,
        "mean_improvement_deg": gain,
    }]}))
    return path


def test_promotion_requires_every_independent_suite_to_pass(tmp_path):
    strong = _summary(tmp_path / "strong.json", 0.05)
    shifted = _summary(tmp_path / "shifted.json", -0.02)
    report = validate_suites(
        [("strong", strong), ("shifted", shifted)],
        "bias", {"chronological"}, {20}, 0.0,
    )
    assert report["status"] == "FAIL"
    assert report["failed_configurations"] == 1


def test_promotion_passes_when_all_suites_meet_threshold(tmp_path):
    first = _summary(tmp_path / "first.json", 0.05)
    second = _summary(tmp_path / "second.json", 0.01)
    report = validate_suites(
        [("first", first), ("second", second)],
        "bias", {"chronological"}, {20}, 0.0,
    )
    assert report["status"] == "PASS"
