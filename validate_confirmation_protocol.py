#!/usr/bin/env python3
"""Validate invariants of the frozen pitch-only confirmation protocol."""

import argparse
import json
from pathlib import Path


def validate(payload: dict) -> None:
    candidate = payload["candidate"]
    calibration = payload["calibration"]
    evaluation = payload["evaluation"]
    novelty = payload["novelty_requirement"]
    assert payload["status"] == "frozen_before_external_confirmation"
    assert candidate["method"] == "pitch_bias"
    assert candidate["trainable_parameter_count"] == 1
    assert candidate["parameter_name"] == "pitch_bias"
    assert candidate["yaw_is_fixed_zero"] is True
    assert calibration["gate_strategy"] == "reliability"
    assert calibration["uses_evaluation_labels"] is False
    assert evaluation["subject_disjoint"] is True
    assert evaluation["segment_frame_isolation"] is True
    assert evaluation["confidence_level"] == 0.95
    assert evaluation["bootstrap_repeats"] >= 10000
    assert evaluation["all_prespecified_configurations_must_pass"] is True
    assert novelty["must_not_reuse_candidate_selection_subjects"] is True
    assert novelty["must_include_unseen_checkpoint_or_device"] is True
    assert len(payload["prohibited_after_data_access"]) >= 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    validate(json.loads(args.protocol.read_text()))
    print(f"PASS: frozen confirmation protocol is valid: {args.protocol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
