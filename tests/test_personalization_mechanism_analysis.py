import numpy as np

from personalization_mechanism_analysis import (
    angles_to_vectors,
    diagnose_model,
    fit_rotation,
)


def _axis_rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def test_rotation_fit_is_proper_orthogonal_and_recovers_mapping():
    rng = np.random.default_rng(7)
    predictions = rng.normal(size=(100, 3))
    predictions /= np.linalg.norm(predictions, axis=1, keepdims=True)
    expected = _axis_rotation([1, 2, -1], 3.0)
    targets = predictions @ expected.T
    fitted = fit_rotation(predictions, targets)["matrix"]
    assert np.allclose(fitted, expected, atol=1e-10)
    assert np.allclose(fitted.T @ fitted, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(fitted), 1.0)


def test_diagnostics_distinguish_bias_from_unneeded_affine_complexity():
    rng = np.random.default_rng(8)
    angles = rng.uniform([-0.3, -0.2], [0.3, 0.2], size=(200, 2))
    predictions = angles_to_vectors(angles)
    targets = angles_to_vectors(angles + np.deg2rad([1.4, -0.7]))
    bias = diagnose_model("bias", predictions[:100], targets[:100], predictions[100:], targets[100:])
    affine = diagnose_model(
        "affine", predictions[:100], targets[:100], predictions[100:], targets[100:]
    )
    assert bias["parameter_count"] == 2
    assert affine["parameter_count"] == 6
    assert bias["evaluation_compensated_deg"] < 0.002
    assert affine["evaluation_compensated_deg"] < 0.002
