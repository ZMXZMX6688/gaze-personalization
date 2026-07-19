import math

import torch

from make_subject_cv_folds import make_subject_cv_folds
from personalization_benchmark import (
    RotationAdapter,
    TangentAffineAdapter,
    aggregate_results,
    find_largest_feasible_split,
    fit_guarded_adapter,
    rotation_matrix_from_vector,
    stratified_sample_indices,
)
from personalize_from_universal import ClipRecord, angles_to_vector, angular_errors_deg


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


def _record(segment_id, offset):
    start = segment_id * 100 + offset
    return ClipRecord(
        sid="short",
        start=start,
        target_frame=start + 28,
        segment_id=segment_id,
        segment_start=segment_id * 100,
        segment_end=(segment_id + 1) * 100,
    )


def test_adaptive_split_uses_largest_feasible_requested_k():
    records = [
        _record(segment_id, offset)
        for segment_id in range(3)
        for offset in range(0, 75, 5)
    ]
    pool_size, calibration, evaluation, _, failures = find_largest_feasible_split(
        records,
        calibration_pool_sizes=[5, 10, 20, 50],
        gap_segments=1,
        protocol="chronological",
    )
    assert pool_size == 10
    assert len(calibration) == 10
    assert {record.segment_id for record in evaluation} == {2}
    assert set(failures) == {50, 20}


def test_adaptive_split_rejects_recording_without_embargoed_evaluation():
    records = [
        _record(segment_id, offset)
        for segment_id in range(2)
        for offset in range(0, 75, 5)
    ]
    try:
        find_largest_feasible_split(
            records,
            calibration_pool_sizes=[5, 10, 20, 50],
            gap_segments=1,
            protocol="interleaved",
        )
    except ValueError as exc:
        assert "No feasible interleaved split" in str(exc)
    else:
        raise AssertionError("Expected short recording to be rejected")


def test_aggregate_reports_subject_coverage():
    rows = []
    for repeat in range(2):
        for sid in ("S1", "S2"):
            rows.append({
                "protocol": "chronological",
                "method": "bias",
                "calibration_size": 5,
                "repeat": repeat,
                "sid": sid,
                "base_mean_deg": 1.0,
                "personalized_mean_deg": 0.9,
                "improvement_mean_deg": 0.1,
                "adapter_scale": 1.0,
            })
    summary = aggregate_results(rows, expected_subjects=4)[0]
    assert summary["subjects"] == 2
    assert summary["subject_coverage_rate"] == 0.5
