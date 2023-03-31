import pytest
from src.preprocess.utils.tag_parser import ner_parser, flatten_ner


def test_ner_parser():
    example_data = {'PERSON-伊能靜': 1.0, 'PERSON-小哈利': 5.0, 'PERSON-庾澄慶': 1.0, 'LOCATION-台北': 1.0, 'PRODUCT-F-35戰機': 2.0}
    expect_data = {
        'person': ['伊能靜', '小哈利', '庾澄慶'],
        'location': ['台北'],
        'event': [],
        'organization': [],
        'item': ['F-35戰機'],
        'others': []
    }

    expect_empty_data = {'person': [], 'location': [], 'event': [], 'organization': [], 'item': [], 'others': []}

    assert ner_parser(example_data) == expect_data
    assert ner_parser({}) == expect_empty_data


def test_flatten_ner():

    input_data = {'person': ['伊能靜', '小哈利', '庾澄慶'], 'location': ['台北'], 'event': [], 'organization': [], 'item': [], 'others': []}
    expect_data = {'庾澄慶', '小哈利', '伊能靜', '台北'}

    assert set(flatten_ner(input_data)) == expect_data
