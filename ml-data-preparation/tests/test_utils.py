import pytest
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


def test_process_daily_positive(spark_test_session):
    from utils import sample_positive_data
    df = spark_test_session.createDataFrame(
        [
            ('12345678', '20220401', 'P11232489160'),
            ('12345678', '20220402', 'P11232489160'),
            ('12345678', '20220403', 'P11232489160'),
            ('12345678', '20220404', 'P11232489160'),
            ('12345678', '20220405', 'P11232489160')
        ],
        schema='openid string, date string, uuid string'
    )

    output_df = sample_positive_data(df, 1, 1)

    expect_df = spark_test_session.createDataFrame(
        [
            ('12345678', '20220401', 'P11232489160'),
            ('12345678', '20220402', 'P11232489160'),
            ('12345678', '20220403', 'P11232489160'),
            ('12345678', '20220404', 'P11232489160'),
            ('12345678', '20220405', 'P11232489160')
        ],
        schema='openid string, date string, uuid string'
    )

    assert expect_df.columns == output_df.columns
    assert sorted(expect_df.collect()) == sorted(output_df.collect())


def test_aggregate_view_also_view_udf(spark_test_session):
    import pyspark.sql.functions as f
    from utils import parse_view_also_view_score_udf
    from pyspark.sql.types import StringType, MapType, FloatType, StructType, StructField

    uuid_col = 'uuid'
    click_uuid_col = 'click_uuid'
    view_also_view_col = 'view_also_view_json'
    date_col = 'date'

    input_schema = StructType([StructField(uuid_col, StringType()),
                               StructField(click_uuid_col, StringType()),
                               StructField(view_also_view_col, MapType(StringType(), MapType(StringType(), MapType(StringType(), FloatType())))),
                               StructField(date_col, StringType())])

    output_schema = StructType([StructField(uuid_col, StringType()),
                                StructField(click_uuid_col, StringType()),
                                StructField(view_also_view_col, MapType(StringType(), MapType(StringType(), MapType(StringType(), FloatType())))),
                                StructField(date_col, StringType()),
                                StructField('view_also_view_score', FloatType())])

    df_test = spark_test_session.createDataFrame([('uuid_1', 'click_uuid_1', {'uuid_1': {'20220501': {'click_uuid_1': 0.1, 'click_uuid_2': 0.2}, '20220420': {'click_uuid_1': 0.2}}}, '20220429'),
                                                  ('uuid_2', 'click_uuid_2', {'uuid_2': {'20220501': {'click_uuid_1': 0.1, 'click_uuid_2': 0.2}, '20220420': {'click_uuid_1': 0.2}}}, '20220505'),
                                                  ('uuid_2', 'click_uuid_2', {'uuid_2': {'20220501': {'click_uuid_1': 0.1, 'click_uuid_2': 0.2}, '20220420': {'click_uuid_1': 0.2}}}, '20220301'),
                                                  ('uuid_1', 'click_uuid_3', {'uuid_1': {'20220501': {'click_uuid_1': 0.1, 'click_uuid_2': 0.2}, '20220420': {'click_uuid_1': 0.2}}}, '20220502')],
                                                 schema=input_schema)

    df_output = df_test.withColumn('view_also_view_score', parse_view_also_view_score_udf(uuid_col, click_uuid_col, view_also_view_col, date_col)(f.struct(f.col(uuid_col),
                                                                                                                                                           f.col(click_uuid_col),
                                                                                                                                                           f.col(view_also_view_col),
                                                                                                                                                           f.col(date_col))))

    df_expected = spark_test_session.createDataFrame([('uuid_1', 'click_uuid_1', {'uuid_1': {'20220501': {'click_uuid_1': 0.1, 'click_uuid_2': 0.2}, '20220420': {'click_uuid_1': 0.2}}}, '20220429', 0.2),
                                                      ('uuid_2', 'click_uuid_2', {'uuid_2': {'20220501': {'click_uuid_1': 0.1, 'click_uuid_2': 0.2}, '20220420': {'click_uuid_1': 0.2}}}, '20220505', 0.2),
                                                      ('uuid_2', 'click_uuid_2', {'uuid_2': {'20220501': {'click_uuid_1': 0.1, 'click_uuid_2': 0.2}, '20220420': {'click_uuid_1': 0.2}}}, '20220301', 0.0),
                                                      ('uuid_1', 'click_uuid_3', {'uuid_1': {'20220501': {'click_uuid_1': 0.1, 'click_uuid_2': 0.2}, '20220420': {'click_uuid_1': 0.2}}}, '20220502', 0.0)],
                                                     schema=output_schema)

    assert df_expected.columns == df_output.columns
    assert sorted(df_expected.collect()) == sorted(df_output.collect())


def test_filter_by_interaction_count(spark_test_session):
    from utils import filter_by_interaction_count

    df_input = spark_test_session.createDataFrame(
        [
            ('userid_0', 'content_id_1'),
            ('userid_0', 'content_id_1'),
            ('userid_1', 'content_id_5'),
            ('userid_1', 'content_id_5'),
            ('userid_1', 'content_id_2'),
            ('userid_2', 'content_id_1'),
            ('userid_3', 'content_id_1'),
            ('userid_3', 'content_id_3')
        ],
        schema='userid string, content_id string'
    )

    df_output = filter_by_interaction_count(df_input, primary_col='userid', requisite_sequence_length=2)

    df_expect = spark_test_session.createDataFrame(
        [
            ('userid_1', 'content_id_5', 3),
            ('userid_1', 'content_id_5', 3),
            ('userid_1', 'content_id_2', 3)
        ],
        schema='userid string, content_id string, count int'
    )

    assert df_expect.columns == df_output.columns
    assert df_expect.collect() == df_output.collect()


def test_prune_preference_user_profile():
    from utils import prune_preference_user_profile

    input_user_tag_profile = """
    {
        "editor": {
            "event": {
                "e-新台幣": {
                    "cnt": 2,
                    "pref": 0.2
                },
                "e-經濟": {
                    "cnt": 1,
                    "pref": 0.1
                },
                "e-美元": {
                    "cnt": 1,
                    "pref": 0.1
                },
                "e-通膨": {
                    "cnt": 3,
                    "pref": 0.3
                },
                "e-本土疫情": {
                    "cnt": 2,
                    "pref": 0.2
                },
                "e-台積電": {
                    "cnt": 1,
                    "pref": 0.1
                }
            }
        },
        "ner": {
            "event": {
                "n-ABF": {
                    "cnt": 9,
                    "pref": 0.9
                },
                "n-購物橘子": {
                    "cnt": 1,
                    "pref": 0.1
                }
            }
        }
    }
    """

    condition = ('pref', 0.2)
    output_data = prune_preference_user_profile(input_user_tag_profile, tag_type=['editor', 'ner'], condition=condition)
    expect_data = '{"editor": {"event": {"e-新台幣": {"cnt": 2, "pref": 0.2}, "e-通膨": {"cnt": 3, "pref": 0.3}, "e-本土疫情": {"cnt": 2, "pref": 0.2}}}, "ner": {"event": {"n-ABF": {"cnt": 9, "pref": 0.9}}}}'
    assert output_data == expect_data


def test_calculate_similarity_score():
    from utils import calculate_similarity_score

    assert calculate_similarity_score('[1, 2, 1]', '[1, 2, 1]') == calculate_similarity_score([1, 2, 1], [1, 2, 1]) == 6
    assert calculate_similarity_score(None, '[1, 2, 1]') == calculate_similarity_score(None, None) == 0
