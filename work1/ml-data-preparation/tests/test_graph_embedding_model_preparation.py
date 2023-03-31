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


def test_merge_popularity_to_content(spark_test_session):
    from execution.prepare_graph_embedding_model_dataset import merge_popularity_to_content

    df_popularity = spark_test_session.createDataFrame(
        [
            ('12345678', 0.6, '20220824', '16')
        ],
        schema='uuid string, popularity_score float, date string, hour string'
    )

    df_content = spark_test_session.createDataFrame(
        [
            ('12345678', 'example_title')
        ],
        schema='content_id string, title string'
    )

    df_expect = spark_test_session.createDataFrame(
        [
            ('12345678', 'example_title', 0.6)
        ],
        schema='content_id string, title string, popularity_score float'
    )

    df_ouput = merge_popularity_to_content(df_content=df_content, df_popularity=df_popularity)

    assert df_expect.columns == df_ouput.columns
    assert df_expect.collect() == df_ouput.collect()
