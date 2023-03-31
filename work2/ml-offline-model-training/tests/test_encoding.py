import pytest
import numpy as np
from bdds_recommendation.src.preprocess.utils import encoding


def test_encoding_success():
    example_data = [2, 23, 6, 49]
    expect_data = np.array(
        [
            0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.
        ])
    np.testing.assert_array_equal(encoding(example_data, max_length=50), expect_data)

    example_data = []
    expect_data = np.array(
        [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        ])
    np.testing.assert_array_equal(encoding(example_data, max_length=50), expect_data)

    example_data = 47
    expect_data = np.array(
        [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.
        ])
    np.testing.assert_array_equal(encoding(example_data, max_length=50), expect_data)

    example_data = [47]
    expect_data = np.array(
        [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.
        ])
    np.testing.assert_array_equal(encoding(example_data, max_length=50), expect_data)
