from unittest import TestCase

import pytest
import json
import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql.types import StringType, ArrayType, StructType, StructField
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


def test_join_event_with_user_profile(spark_test_session):
    from src.gcsreader import join_event_with_user_profile

    df_left = spark_test_session.createDataFrame(
        [
            ('12345678', '20220401', 'P11232489160', 'event')
        ],
        schema='userid string, date string, uuid string, event string'
    )

    df_profile = spark_test_session.createDataFrame(
        [
            ('12345678', '20220401', 'user_category', 'item_title_embedding')
        ],
        schema='userid string, date string, user_category string, item_title_embedding string'
    )

    condition = [df_left.userid == df_profile.userid, df_left.date == df_profile.date]

    output_df = join_event_with_user_profile(df_left, df_profile, cond=condition, how='left')

    expect_df = spark_test_session.createDataFrame(
        [
            ('12345678', '20220401', 'P11232489160', 'event', 'user_category', 'item_title_embedding')
        ],
        schema='userid string, date string, uuid string, event string, user_category string, item_title_embedding string'
    )

    assert expect_df.columns == output_df.columns
    assert expect_df.collect() == output_df.collect()


def test_info_json_map_transform():
    from src.gcsreader.udf import info_json_map_transform

    # scenario 1: no filter_info
    df = pd.DataFrame(columns=['info', 'id'])
    df = df.append({'info': json.dumps({'content_id': '100', 'cat0': ['時事星']}), 'id': '100'}, ignore_index=True)
    df = df.append({'info': json.dumps({'content_id': '101', 'cat0': ['時事星', '放鬆星']}), 'id': '101'}, ignore_index=True)
    df = df.append({'info': json.dumps({'content_id': '102', 'cat0': ['電玩星']}), 'id': '102'}, ignore_index=True)
    df = df.append({'info': json.dumps({'content_id': '103', 'cat0': ['文學星']}), 'id': '103'}, ignore_index=True)
    df = df.append({'info': json.dumps({'content_id': '104'}), 'id': '104'}, ignore_index=True)

    df['parsed_id'] = df['info'].apply(lambda x: info_json_map_transform(x, None, 'content_id', []))
    for index, row in df.iterrows():
        assert row['parsed_id'] == row['id']

    # scenario 2: info == filter_info
    df_filter = pd.DataFrame(columns=['info', 'id'])
    df_filter = df_filter.append({'info': json.dumps({'content_id': '100', 'cat0': ['時事星']}), 'id': '100'}, ignore_index=True)
    df_filter = df_filter.append({'info': json.dumps({'content_id': '101', 'cat0': ['時事星', '放鬆星']}), 'id': '101'}, ignore_index=True)
    df_filter = df_filter.append({'info': json.dumps({'content_id': '102', 'cat0': ['電玩星']}), 'id': '102'}, ignore_index=True)
    df_filter = df_filter.append({'info': json.dumps({'content_id': '103', 'cat0': ['文學星']}), 'id': ''}, ignore_index=True)
    df_filter = df_filter.append({'info': json.dumps({'content_id': '104'}), 'id': ''}, ignore_index=True)
    df_filter['parsed_id'] = df_filter['info'].apply(lambda x: info_json_map_transform(x, x, 'content_id', 'cat0', ['時事星', '放鬆星', '電玩星']))
    for index, row in df_filter.iterrows():
        assert row['parsed_id'] == row['id']

    # scenario 3: info != click_info
    df_filter = pd.DataFrame(columns=['info', 'id'])
    df_filter = df_filter.append({'info': json.dumps({'content_id': '100'}), 'id': '100', 'click_info': json.dumps({'sec': 'pop'})}, ignore_index=True)
    df_filter = df_filter.append({'info': json.dumps({'content_id': '101'}), 'id': '', 'click_info': json.dumps({'sec': 'hello'})}, ignore_index=True)

    df_filter['parsed_id'] = df_filter.apply(lambda x: info_json_map_transform(x['info'], x['click_info'], 'content_id', 'sec', ['pop']), axis=1)
    for index, row in df_filter.iterrows():
        assert row['parsed_id'] == row['id']


def test_info_array_of_json_map_transform():
    from src.gcsreader.udf import info_array_of_json_map_transform

    # scenario 1: no filter info
    df = pd.DataFrame(columns=['info', 'id'])
    df = df.append({'info': json.dumps([{'content_id': '100'}, {'content_id': '101'}]), 'id': ['100', '101']}, ignore_index=True)
    df = df.append({'info': json.dumps([{'content_id': '101', 'sec': 'news'}, {'content_id': '102', 'sec': 'video'}]), 'id': ['101', '102']}, ignore_index=True)
    df = df.append({'info': json.dumps([{'content_id': '100', 'sec': 'news'}, {'content_id': '101'}]), 'id': ['100', '101']}, ignore_index=True)
    df['parsed_id'] = df['info'].apply(lambda x: info_array_of_json_map_transform(x, None, 'content_id', None, []))

    for index, row in df.iterrows():
        assert row['parsed_id'] == row['id']

    # scenario 2: info == filter_info
    df = pd.DataFrame(columns=['info', 'id'])
    df = df.append({'info': json.dumps([{'content_id': '100'}, {'content_id': '101'}]), 'id': []}, ignore_index=True)
    df = df.append({'info': json.dumps([{'content_id': '101', 'sec': 'news'}, {'content_id': '102', 'sec': 'video'}]), 'id': ['101']}, ignore_index=True)
    df = df.append({'info': json.dumps([{'content_id': '102', 'sec': 'news'}, {'content_id': '103', 'sec': 'news'}, {'content_id': '104'}]), 'id': ['102', '103']}, ignore_index=True)
    df['parsed_id'] = df['info'].apply(lambda x: info_array_of_json_map_transform(x, x, 'content_id', 'sec', ['news']))

    for index, row in df.iterrows():
        assert row['parsed_id'] == row['id']

    # scenario 3: info != filter_info
    df = pd.DataFrame(columns=['info', 'id'])
    df = df.append({'info': json.dumps([{'content_id': '100'}, {'content_id': '101'}]),
                    'click_info': json.dumps({'sec': 'novel'}),
                    'id': []}, ignore_index=True)
    df = df.append({'info': json.dumps([{'content_id': '101'}, {'content_id': '102'}]),
                    'click_info': json.dumps({'sec': 'news'}),
                    'id': ['101', '102']}, ignore_index=True)
    df = df.append({'info': json.dumps([{'content_id': '102'},
                                        {'content_id': '103'},
                                        {'content_id': '104'}]),
                    'click_info': json.dumps({'sec': 'news'}),
                    'id': ['102', '103', '104']}, ignore_index=True)

    df['parsed_id'] = df.apply(lambda x: info_array_of_json_map_transform(x['info'], x['click_info'], 'content_id', 'sec', ['news']), axis=1)

    for index, row in df.iterrows():
        assert row['parsed_id'] == row['id']


def test_udf_get_candidate_arms(spark_test_session):
    from src.gcsreader.udf import udf_get_candidate_arms

    schema = StructType([StructField('uuid', StringType()),
                         StructField('uuids', ArrayType(StringType()))])

    df_input = spark_test_session.createDataFrame([('D09222492347', ['D09222492347', 'P10250867561', 'P07251263957']),
                                                   ('P04252053878', ['P04252053878', 'P08251589907', 'P04243384739', 'P05240570426']),
                                                   ('P04251754294', ['P04251754294', 'P04252053878', 'P08251589907'])],
                                                   schema=schema)

    df_output = df_input.withColumn('uuids', udf_get_candidate_arms(3, neg_candidate_col='uuids', pos_id_col='uuid')(f.struct('uuid', 'uuids')))

    df_expected = spark_test_session.createDataFrame([('D09222492347', ['D09222492347', 'P10250867561', 'P07251263957']),
                                                      ('P04252053878', ['P04252053878', 'P08251589907', 'P04243384739', 'P05240570426']),
                                                      ('P04251754294', ['P04251754294', 'P04252053878', 'P08251589907'])],
                                                     schema=schema)

    df_output_collect = sorted(df_output.select('uuids').collect())
    df_expected_collect = sorted(df_expected.select('uuids').collect())

    for i, row in enumerate(df_output_collect):
        assert sorted(row['uuids']) == sorted(df_expected_collect[i]['uuids'])


def test_extend_similar_score_field(spark_test_session):
    from src.gcsreader.utils import extend_similar_score_field
    df_expected = spark_test_session.createDataFrame(
        [
            ('100', [1,2,1],[1,2,1], 6.0)
        ],
        schema='userid string, user_title_embedding string, item_title_embedding string, similarity_score float'
    )

    df_input = spark_test_session.createDataFrame(
        [
            ('100', [1,2,1],[1,2,1])
        ],
        schema='userid string, user_title_embedding string, item_title_embedding string'
    )

    df_output = extend_similar_score_field(df_input, 'similarity_score', ['user_title_embedding', 'item_title_embedding'])

    assert df_expected.columns == df_output.columns
    assert df_expected.collect() == df_output.collect()


def test_filter_candidate(spark_test_session):
    from src.gcsreader.utils import filter_candidate
    df_expected = spark_test_session.createDataFrame(
        [
            ('101', 9)
        ],
        schema='userid string, similarity_score int'
    )

    df_input = spark_test_session.createDataFrame(
        [
            ('100', [1,2,1],[1,2,1], 6),
            ('101', [2,2,1],[2,2,1], 9)
        ],
        schema='userid string, user_title_embedding string, item_title_embedding string, similarity_score int'
    )
    df_output = filter_candidate(df_input, 1, sort_col='similarity_score', dedup_unique_col=['userid'], select_col=['userid','similarity_score'])

    assert df_expected.columns == df_output.columns
    assert df_expected.collect() == df_output.collect()


def test_get_similarity_samples_pool(spark_test_session):
    from src.gcsreader.samples.samples_pool import get_similarity_samples_pool
    df_expected = spark_test_session.createDataFrame(
        [
            ('101', 2, '22220102')
        ],
        schema='userid string, content_id string, publish_time string'
    )

    df_input = spark_test_session.createDataFrame(
        [
            ('100', '1', '22220101', [1,2,1], [1,2,1]),
            ('101', '2', '22220102', [2,2,1], [2,2,1])
        ],
        schema='userid string, content_id string, publish_time string, user_title_embedding string, item_title_embedding string'
    )

    df_output = get_similarity_samples_pool(df_input, 1)

    assert df_expected.columns == df_output.columns
    assert df_expected.collect() == df_output.collect()
