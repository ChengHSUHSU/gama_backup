import pytest


def test_get_norm_value():
    from utils import get_norm_value

    # scenario-1: val > 1
    input_value = 2
    expected_value = 1.1
    otuput_value = get_norm_value(input_value)

    assert expected_value == otuput_value

    # scenario-2: val < 0
    input_value = -2
    expected_value = 0.9
    otuput_value = get_norm_value(input_value)

    assert expected_value == otuput_value

    # scenario-1: 0 <= val <= 1
    input_value = 0.5
    expected_value = 0.5
    otuput_value = get_norm_value(input_value)

    assert expected_value == otuput_value
