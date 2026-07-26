import numpy as np

from analyze_pitch_transport import transport_metrics


def test_transport_metrics_distinguish_match_mismatch_and_overshoot():
    matched = transport_metrics(1.0, np.array([0.7, 0.9]))
    mismatched = transport_metrics(-1.0, np.array([0.7, 0.9]))
    overshot = transport_metrics(1.2, np.array([0.7, 0.9]))
    assert matched["pitch_sign_match"] == 1.0
    assert mismatched["pitch_sign_match"] == 0.0
    assert overshot["pitch_overshoot"] == 1.0
    assert matched["pitch_transport_drift_deg"] < mismatched["pitch_transport_drift_deg"]
