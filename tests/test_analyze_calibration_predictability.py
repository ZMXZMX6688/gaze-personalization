import numpy as np

from analyze_calibration_predictability import fit_ridge, predict_ridge


def test_ridge_recovers_linear_calibration_signal():
    features = np.asarray([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
    ])
    targets = 0.2 + 0.5 * features[:, 0] - 0.3 * features[:, 1]
    model = fit_ridge(features, targets, regularization=1e-8)
    predicted = predict_ridge(model, features)
    assert np.allclose(predicted, targets, atol=1e-7)
