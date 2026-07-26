import numpy as np

from bias_stability_analysis import reliability_scale, split_half_stability, stability_gate
from personalization_mechanism_analysis import angles_to_vectors


def test_stable_bias_passes_stability_gate():
    rng = np.random.default_rng(4)
    angles = rng.uniform([-0.2, -0.1], [0.2, 0.1], size=(20, 2))
    predictions = angles_to_vectors(angles)
    targets = angles_to_vectors(angles + np.deg2rad([1.2, -0.8]))
    bias, difference = split_half_stability(predictions, targets)
    active, magnitude = stability_gate(bias, difference, 0.5, 2.0)
    assert difference < 0.01
    assert magnitude > 1.4
    assert active


def test_inconsistent_half_bias_is_rejected():
    rng = np.random.default_rng(5)
    angles = rng.uniform([-0.2, -0.1], [0.2, 0.1], size=(20, 2))
    predictions = angles_to_vectors(angles)
    offsets = np.tile(np.deg2rad([1.0, 0.0]), (20, 1))
    offsets[1::2, 0] *= -1
    targets = angles_to_vectors(angles + offsets)
    bias, difference = split_half_stability(predictions, targets)
    active, _ = stability_gate(bias, difference, 0.5, 2.0)
    assert difference > 1.9
    assert not active


def test_reliability_scale_shrinks_noise_but_preserves_stable_signal():
    stable = reliability_scale(np.deg2rad(np.array([1.0, 0.0])), 0.1)
    noisy = reliability_scale(np.deg2rad(np.array([0.2, 0.0])), 1.0)
    assert stable > 0.99
    assert noisy == 0.0
