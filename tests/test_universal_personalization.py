import math

import torch

from personalize_from_universal import (
    AngularBiasAdapter,
    ClipRecord,
    angles_to_vector,
    fit_angular_bias,
    fit_guarded_angular_bias,
    split_calibration_pool,
    split_interleaved_calibration_pool,
    vector_to_angles,
)
from train_ours_two_stage import SubjectEmbedding


def test_angle_vector_roundtrip():
    angles = torch.tensor([
        [-0.20, -0.10],
        [0.00, 0.00],
        [0.25, 0.15],
    ])
    recovered = vector_to_angles(angles_to_vector(angles))
    assert torch.allclose(recovered, angles, atol=1e-6)


def test_zero_adapter_is_identity():
    vectors = torch.nn.functional.normalize(torch.randn(12, 3), dim=-1)
    vectors[:, 2].abs_().add_(0.2)
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    adapter = AngularBiasAdapter()
    assert torch.allclose(adapter(vectors), vectors, atol=1e-6)


def test_bias_calibration_recovers_synthetic_user():
    torch.manual_seed(4)
    base_angles = torch.empty(80, 2).uniform_(-0.25, 0.25)
    base_angles[:, 1].mul_(0.6)
    predictions = angles_to_vector(base_angles)
    true_bias = torch.deg2rad(torch.tensor([2.2, -1.4]))
    targets = angles_to_vector(base_angles + true_bias)
    adapter = fit_angular_bias(
        predictions,
        targets,
        steps=300,
        regularization=0.0,
    )
    assert torch.allclose(adapter.bias.detach(), true_bias, atol=math.radians(0.05))


def test_guarded_calibration_keeps_stable_bias():
    angles = torch.zeros(20, 2)
    predictions = angles_to_vector(angles)
    targets = angles_to_vector(angles + torch.deg2rad(torch.tensor([2.0, -1.0])))
    adapter, guard = fit_guarded_angular_bias(
        predictions, targets, regularization=0.0, min_validation_gain_deg=0.01
    )
    assert guard["selected_scale"] > 0.0
    assert torch.linalg.vector_norm(adapter.bias).item() > math.radians(1.0)


def test_guarded_calibration_falls_back_on_temporal_drift():
    predictions = angles_to_vector(torch.zeros(20, 2))
    fit_targets = angles_to_vector(
        torch.zeros(15, 2) + torch.deg2rad(torch.tensor([2.0, 0.0]))
    )
    validation_targets = angles_to_vector(
        torch.zeros(5, 2) + torch.deg2rad(torch.tensor([-2.0, 0.0]))
    )
    targets = torch.cat((fit_targets, validation_targets))
    adapter, guard = fit_guarded_angular_bias(
        predictions, targets, regularization=0.0, validation_fraction=0.25
    )
    assert guard["selected_scale"] == 0.0
    assert torch.equal(adapter.bias.detach(), torch.zeros(2))


def _record(segment_id, start):
    return ClipRecord(
        sid="S1",
        start=start,
        target_frame=start + 28,
        segment_id=segment_id,
        segment_start=segment_id * 100,
        segment_end=(segment_id + 1) * 100,
    )


def test_temporal_split_has_segment_and_frame_embargo():
    records = [
        _record(segment_id, segment_id * 100 + offset)
        for segment_id in range(8)
        for offset in (0, 20, 40)
    ]
    calibration, evaluation, calibration_segments = split_calibration_pool(
        records, max_calibration_clips=5, gap_segments=1
    )
    assert len(calibration) == 5
    assert set(calibration_segments).isdisjoint({r.segment_id for r in evaluation})
    assert max(r.target_frame for r in calibration) < min(r.start for r in evaluation)
    assert min(r.segment_id for r in evaluation) >= max(calibration_segments) + 2


def test_interleaved_split_has_segment_neighbor_and_frame_embargo():
    records = [
        _record(segment_id, segment_id * 100 + offset)
        for segment_id in range(12)
        for offset in (0, 20, 40)
    ]
    calibration, evaluation, calibration_segments = split_interleaved_calibration_pool(
        records, max_calibration_clips=8, gap_segments=1
    )
    calibration_ids = set(calibration_segments)
    evaluation_ids = {record.segment_id for record in evaluation}

    assert len(calibration) == 8
    assert calibration_ids == {0, 5, 11}
    assert evaluation_ids == {2, 3, 7, 8, 9}
    assert calibration_ids.isdisjoint(evaluation_ids)
    assert all(
        calibration_record.target_frame < evaluation_record.start
        or evaluation_record.target_frame < calibration_record.start
        for calibration_record in calibration
        for evaluation_record in evaluation
    )


def test_subject_embedding_starts_as_universal_identity():
    module = SubjectEmbedding(num_subjects=3, embed_dim=4, feat_dim=6, hidden=5, num_layers=2)
    subjects = torch.tensor([0, 1, 2])
    assert torch.equal(module.forward_feat_scale(subjects), torch.ones(3, 1, 6))
    assert torch.equal(
        module.forward_hidden_init(subjects, batch_size=3, device=torch.device("cpu")),
        torch.zeros(2, 3, 5),
    )


def test_hidden_init_keeps_subject_and_layer_axes():
    module = SubjectEmbedding(num_subjects=2, embed_dim=2, feat_dim=2, hidden=2, num_layers=2)
    with torch.no_grad():
        module.embedding.weight.copy_(torch.eye(2))
        module.hidden_proj.weight.copy_(torch.tensor([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]))
        module.hidden_proj.bias.zero_()
    output = module.forward_hidden_init(
        torch.tensor([0, 1]), batch_size=2, device=torch.device("cpu")
    )
    expected = torch.tensor([
        [[1.0, 2.0], [10.0, 20.0]],
        [[3.0, 4.0], [30.0, 40.0]],
    ])
    assert torch.equal(output, expected)
