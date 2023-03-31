import json
import pytest
from unittest import TestCase
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, ArrayType, StructType, StructField



@pytest.fixture(scope='function', autouse=True)
def mock_udf_annotation(monkeypatch):
    def dummy_udf(f):
        return f

    def mock_udf(f=None, returnType=StringType()):
        return f if f else dummy_udf

    monkeypatch.setattr('pyspark.sql.functions.udf', mock_udf)


class TestUDFs(TestCase):

    def test_udf_extract_from_path(self):
        from src.gcsreader.utils import udf_extract_from_path

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

def test_info_array_of_json_map_transaction():
    from src.gcsreader.udf import info_array_of_json_map_transform
    # impression_info_data
    impression_info_data = json.dumps([{'cat1': ['a'], 'name': ['b'], 'tag': ['c', 'd'], 'uuid': '1623583360515313664'}, 
                            {'cat1': ['e'], 'name': ['f'], 'tag': ['g', 'h'], 'uuid': '6456456546546'}])
    # page_info_data
    page_info_data = json.dumps({"cat": "\u63a8\u85a6", "page": "planet", "tab": "\u6642\u4e8b\u661f"})

    content_ids = info_array_of_json_map_transform(impression_info_data, page_info_data, 'uuid', None, [])
    # epxect result
    content_ids_expect = ['1623583360515313664', '6456456546546']
    assert content_ids == content_ids_expect
