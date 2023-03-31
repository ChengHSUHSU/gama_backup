import pytest
import unittest
from bdds_recommendation.src.preprocess.utils.preference import Preference
from tests.config.example_data import USER_CATEGORY_EXAMPLE
import json


class TestPreference(unittest.TestCase):

    def test_get_pref_scores_mapping(self):

        output_data = Preference(config=None)._get_pref_scores_mapping(json.dumps(USER_CATEGORY_EXAMPLE), level=['click', 'cat0'])
        expect_data = {'美食': 0.6063, '活動': 0.2446, '遊戲': 0.0531, '寵物': 0.0319, '盲盒': 0.0212, '動漫': 0.0212, '小說': 0.0106, '閒聊': 0.0106}

        assert output_data == expect_data

    def test_caculate_user_pref_score(self):

        user_data = {'美食': 0.6063, '活動': 0.2446, '遊戲': 0.0531, '寵物': 0.0319, '盲盒': 0.0212, '動漫': 0.0212, '小說': 0.0106, '閒聊': 0.0106}

        # Test1: multi-hot pref score caculation
        item_data = ['美食', '閒聊', '美妝', '化妝品']
        output_data = Preference(config=None)._caculate_user_pref_score(user_data, item_data)
        expect_data = 0.6169
        assert output_data == expect_data

        # Test2: multi-hot pref score caculation - empty item data
        item_data = []
        output_data = Preference(config=None)._caculate_user_pref_score(user_data, item_data)
        expect_data = 0.0
        assert output_data == expect_data

        # Test3: one-hot pref score caculation
        item_data = '美食'
        output_data = Preference(config=None)._caculate_user_pref_score(user_data, item_data)
        expect_data = 0.6063
        assert output_data == expect_data

    def test_parse_tags_from_string(self):

        input_data = "'p-伊能靜', 'p-小哈利', 'l-台北', '美妝' , 'i-化妝品', '美食' ,'閒聊'"
        output_data = Preference(config=None)._parse_tags_from_string(input_data)
        expect_data = ['伊能靜', '小哈利', '台北', '美妝', '化妝品', '美食', '閒聊']

        assert output_data == expect_data

    def test_remove_entity_prefix(self):
        assert '伊能靜' == Preference(config=None)._remove_entity_prefix('p-伊能靜')
