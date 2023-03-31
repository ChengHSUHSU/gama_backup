import pytest
import pyspark.sql.types as t
from unittest import TestCase


@pytest.fixture(scope='function', autouse=True)
def mock_udf_annotation(monkeypatch):
    def dummy_udf(f):
        return f

    def mock_udf(f=None, returnType=t.StringType()):
        return f if f else dummy_udf

    monkeypatch.setattr('pyspark.sql.functions.udf', mock_udf)


class TestUDFs(TestCase):

    def test_shift_datetime(self):
        from src.gcsreader.planet_comics_book_user2item import shift_datetime_udf

        input_data = ['20220201', '20220630']
        expect = [['20220202', '20220131'],
                  ['20220702', '20220627']]

        assert shift_datetime_udf(input_data[0], 1) == expect[0][0]
        assert shift_datetime_udf(input_data[0], -1) == expect[0][1]
        assert shift_datetime_udf(input_data[1], 2) == expect[1][0]
        assert shift_datetime_udf(input_data[1], -3) == expect[1][1]
