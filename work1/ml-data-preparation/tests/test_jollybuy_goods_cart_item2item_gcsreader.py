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

    def test_parse_jb_cart_page_view(self):
        from src.gcsreader.jollybuy_goods_multi_item2item import parse_jb_cart_page_view_udf

        input_data = ['[{"amount":1350,"detail":[{"amount":1350,"cat0":["電玩遊戲"],"cat1":["任天堂遊戲片"],"content_type":"others","name":"【就是要玩】NS Switch 超級瑪利歐兄弟 U 豪華版 中文版 瑪利兄弟U 馬力歐U 瑪莉歐 瑪利歐U 瑪U","uuid":"P09242956617"}],"name":"就是要玩 SWITCH好物專賣店","type":"enterprise","uuid":"S190204000321712"}]',
                      '[{"amount":47580,"detail":[{"amount":44590,"cat0":["家電影音"],"cat1":["音響、喇叭"],"content_type":"others","name":"SAMSUNG 三星 Q950A 11.1.4 聲道  Soundbar 極致聲霸 送原廠腳架","uuid":"P15243159005"},{"amount":2990,"cat0":["居家生活"],"cat1":["其他"],"content_type":"others","name":"三星 Samsung  原廠後置喇叭架 腳架","uuid":"P06251914429"}],"name":"小艾生活百貨","type":"personal","uuid":"S211027150819177"}]']

        expect = [['P09242956617'], ['P15243159005', 'P06251914429']]

        assert parse_jb_cart_page_view_udf(input_data[0]) == expect[0]
        assert parse_jb_cart_page_view_udf(input_data[1]) == expect[1]
