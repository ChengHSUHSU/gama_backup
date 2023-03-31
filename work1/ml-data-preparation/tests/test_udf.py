import pytest
from unittest import TestCase
from pyspark.sql.types import StringType


@pytest.fixture(scope='function', autouse=True)
def mock_udf_annotation(monkeypatch):
    def dummy_udf(f):
        return f

    def mock_udf(f=None, returnType=StringType()):
        return f if f else dummy_udf

    monkeypatch.setattr('pyspark.sql.functions.udf', mock_udf)


class TestUDFs(TestCase):

    def test_udf_extract_from_path(self):
        from src.gcsreader.udf import udf_extract_from_path

        date_len = len('yyyymmdd')
        hour_len = len('hh')
        hour_offset = date_len + 1

        prefix = 'category/'
        path = 'gs://pipeline-bf-data-prod-001/user_profiles_cassandra/beanfun/planet/category/20221031/news/*.parquet'
        assert udf_extract_from_path(prefix, date_len)(path) == '20221031'

        prefix = 'news/'
        path = 'gs://pipeline-bf-data-prod-001/metrics/popularity/news/20221031/01/popularity_news.csv'
        assert udf_extract_from_path(prefix, hour_len, hour_offset)(path) == '01'

    def test_udf_set_user_profile_join_date(self):
        from src.gcsreader.udf import udf_set_user_profile_join_date

        date = '20221220'
        assert udf_set_user_profile_join_date(day_ranges=3)(date) == '20221223'

    def test_udf_get_date_from_timestamp(self):
        from src.gcsreader.udf import udf_get_date_from_timestamp

        offset = 1

        timestamp = 1677296912
        date_time_str = udf_get_date_from_timestamp(offset=offset)(timestamp)
        expected_date_time_str = '20230224'
        assert date_time_str == expected_date_time_str

        timestamp = '1677383533'
        date_time_str = udf_get_date_from_timestamp(offset=offset)(timestamp)
        expected_date_time_str = '20230225'
        assert date_time_str == expected_date_time_str
