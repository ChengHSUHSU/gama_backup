# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from pandas._testing import assert_frame_equal
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_test_session():
    return (
        SparkSession
        .builder
        .master('local[*]')
        .appName('unit-testing')
        .getOrCreate()
    )


def test_parse_planet_category():
    from execution.prepare_planet_news_user2item_dataset import parse_planet_category
    example_text = [
        ["['放鬆星', '智多星']", "['新奇的鬆', '國際新聞']"],
        ["['放鬆星', '智多星', '電玩星']", "['新奇的鬆', '國際新聞', '科技娛樂']"],
        ["['放鬆星']", "['新奇的鬆']"]
    ]

    input_dataframe = pd.DataFrame(data=example_text, columns=['cat0', 'cat1'])
    output_dataframe = parse_planet_category(input_dataframe)[['cat0_0', 'cat0_1', 'cat1_0', 'cat1_1']]

    expect_text = [
        ['放鬆星', '智多星', '新奇的鬆', '國際新聞'],
        ['放鬆星', '智多星', '新奇的鬆', '國際新聞'],
        ['放鬆星', None, '新奇的鬆', None]
    ]
    expect_dataframe = pd.DataFrame(data=expect_text, columns=['cat0_0', 'cat0_1', 'cat1_0', 'cat1_1'])

    assert_frame_equal(expect_dataframe, output_dataframe)
