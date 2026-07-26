import numpy as np

from plot_checkpoint_validation_matrix import build_gain_matrix


def test_gain_matrix_preserves_suite_and_protocol_k_mapping():
    rows = [
        {"suite": "a", "protocol": "chronological", "calibration_size": 20,
         "mean_gain_deg": 0.1},
        {"suite": "a", "protocol": "interleaved", "calibration_size": 20,
         "mean_gain_deg": -0.2},
        {"suite": "b", "protocol": "chronological", "calibration_size": 20,
         "mean_gain_deg": 0.3},
        {"suite": "b", "protocol": "interleaved", "calibration_size": 20,
         "mean_gain_deg": 0.4},
    ]
    for index, row in enumerate(rows):
        row["status"] = "PASS" if index % 2 == 0 else "FAIL"
    suites, labels, matrix, passed = build_gain_matrix(rows)
    assert suites == ["a", "b"]
    assert labels == ["Chron.\nK=20", "Inter.\nK=20"]
    assert np.allclose(matrix, [[0.1, -0.2], [0.3, 0.4]])
    assert passed.tolist() == [[True, False], [True, False]]
