import copy

import pytest

from prepare_confirmation_run import fingerprint, prepare


PROTOCOL = {
    "status": "frozen_before_external_confirmation",
    "candidate": {"method": "pitch_bias", "trainable_parameter_count": 1,
                  "parameter_name": "pitch_bias", "yaw_is_fixed_zero": True},
    "calibration": {"gate_strategy": "reliability", "uses_evaluation_labels": False},
    "evaluation": {"subject_disjoint": True, "segment_frame_isolation": True,
                   "confidence_level": .95, "bootstrap_repeats": 10000,
                   "all_prespecified_configurations_must_pass": True},
    "novelty_requirement": {"must_not_reuse_candidate_selection_subjects": True,
                            "must_include_unseen_checkpoint_or_device": True},
    "prohibited_after_data_access": ["a", "b", "c", "d", "e"],
}
OLD_HASH = "1" * 64
NEW_HASH = "2" * 64
SELECTION = {"subject_ids": ["OLD"], "checkpoint_sha256": [OLD_HASH],
             "device_families": ["old-device"]}
COHORT = {"confirmation_id": "c1", "subject_ids": ["NEW"],
          "checkpoint_sha256": NEW_HASH, "device_family": "old-device"}


def test_prepare_locks_novel_cohort_with_stable_fingerprints():
    lock = prepare(PROTOCOL, SELECTION, COHORT)
    assert lock["status"] == "LOCKED_FOR_CONFIRMATION"
    assert lock["checkpoint_is_new"] is True
    assert lock["protocol_sha256"] == fingerprint(PROTOCOL)


def test_prepare_rejects_subject_overlap():
    cohort = copy.deepcopy(COHORT); cohort["subject_ids"] = ["OLD"]
    with pytest.raises(ValueError, match="overlap"):
        prepare(PROTOCOL, SELECTION, cohort)


def test_prepare_rejects_old_checkpoint_and_device():
    cohort = copy.deepcopy(COHORT); cohort["checkpoint_sha256"] = OLD_HASH
    with pytest.raises(ValueError, match="unseen"):
        prepare(PROTOCOL, SELECTION, cohort)


def test_prepare_rejects_non_sha_checkpoint_identity():
    cohort = copy.deepcopy(COHORT); cohort["checkpoint_sha256"] = "model-final"
    with pytest.raises(ValueError, match="64 lowercase"):
        prepare(PROTOCOL, SELECTION, cohort)
