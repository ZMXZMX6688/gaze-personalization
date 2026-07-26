import numpy as np

from analyze_bias_components import aggregate_subjects, component_gains
from personalization_mechanism_analysis import angles_to_vectors


def test_component_gains_isolates_yaw_bias():
    predictions = angles_to_vectors(np.zeros((8, 2)))
    targets = angles_to_vectors(np.tile(np.deg2rad([2.0, 0.0]), (8, 1)))
    gains = component_gains(predictions, targets, [2.0, 0.0])
    assert gains["full"]["error_deg"] < 1e-5
    assert gains["yaw_only"]["error_deg"] < 1e-5
    assert abs(gains["pitch_only"]["gain_deg"]) < 1e-5


def test_component_gains_isolates_pitch_bias():
    predictions = angles_to_vectors(np.zeros((8, 2)))
    targets = angles_to_vectors(np.tile(np.deg2rad([0.0, -3.0]), (8, 1)))
    gains = component_gains(predictions, targets, [0.0, -3.0])
    assert gains["full"]["error_deg"] < 1e-5
    assert gains["pitch_only"]["error_deg"] < 1e-5
    assert abs(gains["yaw_only"]["gain_deg"]) < 1e-5


def test_aggregate_subjects_is_macro_and_paired():
    rows = []
    values = {
        "s1": {"zero": 0.0, "yaw_only": 0.2, "pitch_only": 0.1, "full": 0.4},
        "s2": {"zero": 0.0, "yaw_only": 0.4, "pitch_only": 0.2, "full": 0.5},
    }
    for sid, components in values.items():
        for component, gain in components.items():
            rows.append(
                {
                    "suite": "suite",
                    "protocol": "chronological",
                    "calibration_size": 10,
                    "sid": sid,
                    "component": component,
                    "mean_gain_deg": gain,
                }
            )
    summaries, increments = aggregate_subjects(rows, 0.95, 100)
    full = next(row for row in summaries if row["component"] == "full")
    vs_yaw = next(
        row for row in increments if row["comparison"] == "full_vs_yaw_only"
    )
    assert full["mean_gain_deg"] == 0.45
    assert np.isclose(vs_yaw["mean_increment_deg"], 0.15)
