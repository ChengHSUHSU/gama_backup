import pytest
import unittest
import json
import numpy as np
import pandas as pd

from bdds_recommendation.src.preprocess.utils.din import Behavior
from bdds_recommendation.src.preprocess.utils.din import padding, get_topk_preference_data, get_behavior_feature

PADDING_TOKEN = 0


class TestBehavior(unittest.TestCase):

    def test_padding(self):

        input_data = [3, 45, 6, 7, 22]

        # Test1: sequence size > length of behavior sequence
        output_data = Behavior(sequence_size=10, padding_token=PADDING_TOKEN).padding(input_data)
        expect_data = np.array([3, 45, 6, 7, 22, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(output_data, expect_data)

        # Test2: sequence size = length of behavior sequence
        output_data = Behavior(sequence_size=5, padding_token=PADDING_TOKEN).padding(input_data)
        expect_data = np.array([3, 45, 6, 7, 22])
        np.testing.assert_array_equal(output_data, expect_data)

        # Test3: sequence size < length of behavior sequence
        output_data = Behavior(sequence_size=3, padding_token=PADDING_TOKEN).padding(input_data)
        expect_data = np.array([6, 7, 22])
        np.testing.assert_array_equal(output_data, expect_data)

    def test_get_behavior_sequence_len(self):

        input_data = [3, 45, 6, 7, 22, 0, 0, 0, 0, 0]
        output_data = Behavior(sequence_size=10, padding_token=PADDING_TOKEN).get_behavior_sequence_len(input_data)
        expect_data = 5

        assert output_data == expect_data

        input_data = [3, 45, 6]
        output_data = Behavior(sequence_size=3, padding_token=PADDING_TOKEN).get_behavior_sequence_len(input_data)
        expect_data = 3

        assert output_data == expect_data

    def test_process_hist(self):

        # Test1: remove behavior user had viewed
        input_data = {
            'behavior': 45,
            'behavior_sequence': [3, 45, 6, 7, 22, 0, 0, 0, 0, 0]
        }
        output_data = Behavior(sequence_size=10, padding_token=PADDING_TOKEN) \
            .process_hist(input_data['behavior'], hist_list=input_data['behavior_sequence'])
        expect_data = [3, 6, 7, 22, 0, 0, 0, 0, 0]

        assert output_data == expect_data

        # Test2: dont need to remove
        input_data = {
            'behavior': 5,
            'behavior_sequence': [3, 45, 6, 7, 22, 0, 0, 0, 0, 0]
        }
        output_data = Behavior(sequence_size=10, padding_token=PADDING_TOKEN) \
            .process_hist(input_data['behavior'], hist_list=input_data['behavior_sequence'])
        expect_data = [3, 45, 6, 7, 22, 0, 0, 0, 0, 0]

        assert output_data == expect_data

        # Test3: enable recursive_pop_out
        input_data = {
            'behavior': 3,
            'behavior_sequence': [3, 45, 3, 7, 22, 0, 0, 0, 0, 0]
        }
        output_data = Behavior(sequence_size=10, padding_token=PADDING_TOKEN) \
            .process_hist(input_data['behavior'], hist_list=input_data['behavior_sequence'], recursive_pop_out=True)
        expect_data = [45, 7, 22, 0, 0, 0, 0, 0]

        assert output_data == expect_data


def test_padding():

    input_data = [1, 2, 3]
    output_data = padding(data=input_data, max_sequence_length=5, pad_token=0)
    expect_data = np.array([1, 2, 3, 0, 0])
    np.testing.assert_array_equal(output_data, expect_data)

    input_data = [1, 2, 3, 4, 5, 6, 7]
    output_data = padding(data=input_data, max_sequence_length=5, pad_token=0)
    expect_data = np.array([1, 2, 3, 4, 5])
    np.testing.assert_array_equal(output_data, expect_data)


def test_get_topk_preference_data():

    example_user_category_profile = {
        "click": {
            "cat1": {
                "國際新聞": {"cnt": 3, "pref": 0.12},
                "地方消息": {"cnt": 2, "pref": 0.08},
                "運動藝文": {"cnt": 10, "pref": 0.4},
                "遊戲情報": {"cnt": 2, "pref": 0.08},
                "生活要事": {"cnt": 7, "pref": 0.28},
                "娛樂劇星": {"cnt": 1, "pref": 0.04}
            },
            "cat0": {
                "時事星": {"cnt": 22, "pref": 0.88},
                "電玩星": {"cnt": 2, "pref": 0.08},
                "放鬆星": {"cnt": 1, "pref": 0.04}}}
    }

    input_data = json.dumps(example_user_category_profile)
    output_data = get_topk_preference_data(input_data, max_sequence_length=3, profile_key='click', sequence_key='cat1')
    expect_data = ['運動藝文', '生活要事', '國際新聞']

    assert output_data == expect_data


def test_get_behavior_feature():

    encoder = {'PAD': 0, 'UNKNOWN': 1, '國際新聞': 2, '地方消息': 3, '運動藝文': 4, '遊戲情報': 5, '生活要事': 6, '娛樂劇星': 7}

    example_user_category_profile = {
        "click": {
            "cat1": {
                "國際新聞": {"cnt": 3, "pref": 0.12},
                "地方消息": {"cnt": 2, "pref": 0.08},
                "運動藝文": {"cnt": 10, "pref": 0.4},
                "遊戲情報": {"cnt": 2, "pref": 0.08},
                "生活要事": {"cnt": 7, "pref": 0.28},
                "娛樂劇星": {"cnt": 1, "pref": 0.04}
            },
            "cat0": {
                "時事星": {"cnt": 22, "pref": 0.88},
                "電玩星": {"cnt": 2, "pref": 0.08},
                "放鬆星": {"cnt": 1, "pref": 0.04}}}
    }

    example_user_category_profile = json.dumps(example_user_category_profile)
    input_data = {'user_category': [example_user_category_profile]}

    # case 1: test training mode
    df_expect_hist = [4, 6, 2, 3, 5, 7, 0, 0, 0, 0]
    df_expect_seq_length = 6
    df_input = pd.DataFrame(data=input_data)
    df_output = get_behavior_feature(dataset=df_input,
                                     encoder=encoder,
                                     encoder_key='cat1',
                                     behavior_col='user_category',
                                     seq_length_col='seq_length',
                                     prefix='hist_',
                                     profile_key='click',
                                     sequence_key='cat1',
                                     max_sequence_length=10,
                                     mode='training')

    df_output_hist = df_output['hist_cat1'][0].tolist()
    df_output_seq_length = df_output['seq_length'][0]

    assert df_output_hist == df_expect_hist
    assert df_output_seq_length == df_expect_seq_length

    # case 2: test serving mode
    df_expect = [4, 6, 2, 3, 5, 7]
    df_input = example_user_category_profile
    df_output = get_behavior_feature(dataset=df_input,
                                     encoder=encoder,
                                     encoder_key='cat1',
                                     profile_key='click',
                                     sequence_key='cat1',
                                     max_sequence_length=10,
                                     mode='serving')

    assert df_output == df_expect
