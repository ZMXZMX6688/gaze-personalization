import math

import torch

from make_subject_cv_folds import make_subject_cv_folds
from personalization_benchmark import (
    RotationAdapter,
    TangentAffineAdapter,
    fit_guarded_adapter,
    rotation_matrix_from_vector,
    stratified_sample_indices,
)
from personalize_from_universal import angles_to_vector, angular_errors_deg


def test_stratified_sampling_is_reproducible_and_unique():
    first = stratified_sample_indices(50, 10, seed=17)
    second = stratified_sample_indices(50, 10, seed=17)
    third = stratified_sample_indices(50, 10, seed=18)
    assert first == second
    assert first != third
    assert first == sorted(first)
    assert len(set(first)) == 10


def test_rotation_matrix_and_adapter_identity():
    matrix = rotation_matrix_from_vector(torch.zeros(3))
    assert torch.allclose(matrix, torch.eye(3), atol=1e-7)
    vectors = torch.nn.functional.normalize(torch.randn(20, 3), dim=-1)
    assert torch.allclose(RotationAdapter()(vectors), vectors, atol=1e-6)


def test_rotation_adapter_recovers_synthetic_rotation():
    torch.manual_seed(8)
    predictions = torch.nn.functional.normalize(torch.randn(80, 3), dim=-1)
    true_rotation = torch.deg2rad(torch.tensor([1.2, -1.8, 0.7]))
    matrix = rotation_matrix_from_vector(true_rotation)
    targets = predictions @ matrix.T
    adapter, guard = fit_guarded_adapter(
        "rotation",
        predictions,
        targets,
        steps=350,
        learning_rate=0.04,
        regularization=0.0,
        max_bias_deg=10.0,
        max_linear_delta=0.25,
        validation_fraction=0.25,
        min_validation_gain_deg=0.01,
    )
    assert guard["selected_scale"] == 1.0
    error = angular_errors_deg(adapter(predictions), targets).mean()
    assert error < 0.03


def test_affine_adapter_recovers_synthetic_user():
    torch.manual_seed(9)
    angles = torch.empty(100, 2).uniform_(-0.3, 0.3)
    angles[:, 1].mul_(0.6)
    predictions = angles_to_vector(angles)
    matrix = torch.tensor([[1.06, 0.03], [-0.02, 0.94]])
    bias = torch.deg2rad(torch.tensor([1.5, -0.8]))
    targets = angles_to_vector(angles @ matrix.T + bias)
    adapter, guard = fit_guarded_adapter(
        "affine",
        predictions,
        targets,
        steps=500,
        learning_rate=0.03,
        regularization=0.0,
        max_bias_deg=10.0,
        max_linear_delta=0.25,
        validation_fraction=0.25,
        min_validation_gain_deg=0.01,
    )
    assert guard["selected_scale"] == 1.0
    assert torch.allclose(adapter.linear_delta, matrix - torch.eye(2), atol=0.015)
    assert torch.allclose(adapter.bias, bias, atol=math.radians(0.08))


def test_guard_falls_back_for_inconsistent_rotation():
    predictions = torch.nn.functional.normalize(torch.randn(20, 3), dim=-1)
    fit_matrix = rotation_matrix_from_vector(torch.deg2rad(torch.tensor([0.0, 2.0, 0.0])))
    val_matrix = rotation_matrix_from_vector(torch.deg2rad(torch.tensor([0.0, -2.0, 0.0])))
    targets = torch.cat((
        predictions[:15] @ fit_matrix.T,
        predictions[15:] @ val_matrix.T,
    ))
    adapter, guard = fit_guarded_adapter(
        "rotation",
        predictions,
        targets,
        steps=300,
        learning_rate=0.04,
        regularization=0.0,
        max_bias_deg=10.0,
        max_linear_delta=0.25,
        validation_fraction=0.25,
        min_validation_gain_deg=0.02,
    )
    assert guard["selected_scale"] == 0.0
    assert torch.equal(adapter.rotation.detach(), torch.zeros(3))


def test_subject_cv_folds_are_disjoint_and_cover_all_subjects():
    sids = [f"S{i:02d}" for i in range(17)]
    folds = make_subject_cv_folds(sids, n_folds=5, val_size=3, seed=12)
    covered = []
    for fold in folds:
        train = set(fold["train_sids"])
        val = set(fold["val_sids"])
        test = set(fold["test_sids"])
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert train | val | test == set(sids)
        covered.extend(test)
    assert sorted(covered) == sorted(sids)
