import json

from research_suite_config import load_config, parse_suite
from run_personalization_research_audit import sha256


def test_sha256_is_stable_for_reproduction_inputs(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b'{"value":1}\n')
    assert sha256(path) == "3a37782e8974c48eebf2a0517c866ad15641c53b3d31993188796b56aeb79624"


def test_suite_can_bind_aggregate_summary_to_multiple_result_dirs():
    name, summary, result_dirs = parse_suite(
        "cv5=/aggregate/summary.json::/fold0,/fold1"
    )
    assert name == "cv5"
    assert str(summary) == "/aggregate/summary.json"
    assert [str(path) for path in result_dirs] == ["/fold0", "/fold1"]


def test_config_loads_authoritative_summary_and_result_dirs(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"suites": [{
        "name": "external",
        "summary_path": "/aggregate/summary.json",
        "result_dirs": ["/fold0", "/fold1"],
    }]}))
    assert load_config(path) == [
        ("external", path.__class__("/aggregate/summary.json"),
         [path.__class__("/fold0"), path.__class__("/fold1")])
    ]
