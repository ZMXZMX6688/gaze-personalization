import copy
import json
from pathlib import Path

import pytest

from validate_confirmation_protocol import validate


def protocol():
    return json.loads(
        (Path(__file__).parents[1] / "confirmation_protocol.json").read_text()
    )


def test_frozen_confirmation_protocol_is_valid():
    validate(protocol())


def test_protocol_rejects_evaluation_label_gate():
    payload = copy.deepcopy(protocol())
    payload["calibration"]["uses_evaluation_labels"] = True
    with pytest.raises(AssertionError):
        validate(payload)


def test_protocol_rejects_reused_selection_subjects():
    payload = copy.deepcopy(protocol())
    payload["novelty_requirement"]["must_not_reuse_candidate_selection_subjects"] = False
    with pytest.raises(AssertionError):
        validate(payload)
