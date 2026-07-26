from attribute_personalization_failure import classify_transport


def test_transport_classification_separates_direction_and_magnitude():
    assert classify_transport(-0.1, 1.0, 0.75, 1.25) == "direction_reversal"
    assert classify_transport(0.9, 0.5, 0.75, 1.25) == "magnitude_attenuation"
    assert classify_transport(0.9, 1.5, 0.75, 1.25) == "magnitude_amplification"
    assert classify_transport(0.9, 1.0, 0.75, 1.25) == "approximately_preserved"
