# -*- coding: utf-8 -*-
import pytest
import pyspark.sql.functions as f
from src.gcsreader import GcsReader
from src.negative_sampler import _negative_sampler
from pyspark.sql import SparkSession
import pandas as pd


@pytest.fixture(scope="session")
def spark_test_session():
    return (
        SparkSession
        .builder
        .master('local[*]')
        .appName('unit-testing')
        .getOrCreate()
    )


def test_negative_sampler():

    positive_samples_list = {
        'openid': ['openid_001', 'openid_002', 'openid_003'],
        'uuid': ['uuid_001', 'uuid_002', 'uuid_003'],
        'publish_time': ['1699999999999', '1611111111111', '1655555555555']
    }
    positive_samples = pd.DataFrame(data=positive_samples_list)
    negative_samples = _negative_sampler(positive_samples, sample_size=2)

    assert negative_samples
    assert negative_samples['openid_002'] == []
    assert 'uuid_001' not in negative_samples['openid_001']
    assert 'uuid_001' not in negative_samples['openid_003']


def test_generate_negative_sampler(spark_test_session):
    from src.spark_negative_sampler import generate_negative_samples

    df_content = spark_test_session.createDataFrame(
        [
            ('P06252058477', '1526475205000'),
            ('P06252058479', '1526475205000'),
            ('P06252058476', '1800000000000'),
            ('P0123456786', '1600000000000'),
            ('P0000000000', '1526475205000'),
            ('P0000000001', '1526475205000'),
            ('P06252058472', '1526475205000'),
        ], schema='uuid string, publish_time string'
    )

    df_event = spark_test_session.createDataFrame(
        [
            ('123456', 'P0123456789', '1526475205000'),
            ('123456', 'P0123456786', '1600000000000'),
            ('123456', 'P0123456783', '1700000000000')
        ], schema='openid string, uuid string, publish_time string'
    )

    max_publish_time = df_event.select('publish_time').toPandas().sort_values(by='publish_time').values[-1][0]

    df_output = generate_negative_samples(df_event, df_content, sample_size=5)

    assert df_event.join(df_output, how='inner', on=['uuid']).count() == 0
    assert df_output.join(df_content, how='inner', on=['uuid']).filter(f.col('publish_time') > max_publish_time).count() == 0
