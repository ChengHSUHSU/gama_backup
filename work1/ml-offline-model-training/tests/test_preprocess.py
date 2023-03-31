import unittest
import pytest
import numpy as np
import pandas as pd
import math

from bdds_recommendation.src.preprocess.utils.encoder import DataEncoder
from bdds_recommendation.src.preprocess.utils.encoder import CategoricalEncoder
from bdds_recommendation.src.preprocess.utils import convert_type, convert_unit, encoding, checknull, normalize_data, parse_content_ner
from bdds_recommendation.src.preprocess.utils import cal_features_sim, cal_ner_matching, cal_features_matching
from bdds_recommendation.src.preprocess.utils import get_embedding_interaction, get_publish_time_to_now
from bdds_recommendation.src.preprocess.utils import process_embedding, process_age, process_gender, process_tag
from bdds_recommendation.src.preprocess.utils import merge_data_with_source


class TestDataEncoder(unittest.TestCase):

    def test_add_data(self):
        test_data = ['購車趣', '購車趣', 'NOW保庇', 'NOW保庇', 'NOW保庇', 'NOW健康', '蒼冥灰月', '設計家']

        # Case 1: enable_padding=False, enable_unknown=False
        encoder = DataEncoder(enable_padding=False, enable_unknown=False)
        encoder.add_data(test_data)

        assert encoder.data2index == {'購車趣': 0, 'NOW保庇': 1, 'NOW健康': 2, '蒼冥灰月': 3, '設計家': 4}
        assert encoder.data2count == {'購車趣': 2, 'NOW保庇': 3, 'NOW健康': 1, '蒼冥灰月': 1, '設計家': 1}
        assert encoder.index2data == {0: '購車趣', 1: 'NOW保庇', 2: 'NOW健康', 3: '蒼冥灰月', 4: '設計家'}
        assert encoder.num_of_data == 5

        # Case 2: enable_padding=True, enable_unknown=False
        encoder = DataEncoder(enable_padding=True, enable_unknown=False)
        encoder.add_data(test_data)

        assert encoder.data2index == {'PAD': 0, '購車趣': 1, 'NOW保庇': 2, 'NOW健康': 3, '蒼冥灰月': 4, '設計家': 5}
        assert encoder.data2count == {'PAD': 1, '購車趣': 2, 'NOW保庇': 3, 'NOW健康': 1, '蒼冥灰月': 1, '設計家': 1}
        assert encoder.index2data == {0: 'PAD', 1: '購車趣', 2: 'NOW保庇', 3: 'NOW健康', 4: '蒼冥灰月', 5: '設計家'}
        assert encoder.num_of_data == 6

        # Case 3: enable_padding=False, enable_unknown=True
        encoder = DataEncoder(enable_padding=False, enable_unknown=True)
        encoder.add_data(test_data)

        assert encoder.data2index == {'UNKNOWN': 0, '購車趣': 1, 'NOW保庇': 2, 'NOW健康': 3, '蒼冥灰月': 4, '設計家': 5}
        assert encoder.data2count == {'UNKNOWN': 1, '購車趣': 2, 'NOW保庇': 3, 'NOW健康': 1, '蒼冥灰月': 1, '設計家': 1}
        assert encoder.index2data == {0: 'UNKNOWN', 1: '購車趣', 2: 'NOW保庇', 3: 'NOW健康', 4: '蒼冥灰月', 5: '設計家'}
        assert encoder.num_of_data == 6

        # Case 4: enable_padding=True, enable_unknown=True
        encoder = DataEncoder(enable_padding=True, enable_unknown=True)
        encoder.add_data(test_data)

        assert encoder.data2index == {'PAD': 0, 'UNKNOWN': 1, '購車趣': 2, 'NOW保庇': 3, 'NOW健康': 4, '蒼冥灰月': 5, '設計家': 6}
        assert encoder.data2count == {'PAD': 1, 'UNKNOWN': 1, '購車趣': 2, 'NOW保庇': 3, 'NOW健康': 1, '蒼冥灰月': 1, '設計家': 1}
        assert encoder.index2data == {0: 'PAD', 1: 'UNKNOWN', 2: '購車趣', 3: 'NOW保庇', 4: 'NOW健康', 5: '蒼冥灰月', 6: '設計家'}
        assert encoder.num_of_data == 7


class TestCategoricalEncoder(unittest.TestCase):

    def test_transform(self):
        col = 'cat0'
        all_cats = ['電玩星', '時事星', '放鬆星']
        input_data = pd.Series([['時事星'], ['放鬆星', '電玩星'], ['電玩星'], ['電玩星', '時事星', '放鬆星'], ['測試']])

        # Case 1: enable_padding=False, enable_unknown=False, mode='LabelEncoding'
        categorical_encoder = CategoricalEncoder(col2label2idx={})
        output_data = categorical_encoder.encode_transform(list_data=input_data, col=col, all_cats=all_cats,
                                                           enable_padding=False, enable_unknown=False, mode='LabelEncoding')

        assert output_data == [[1], [2, 0], [0], [0, 1, 2], [0]]
        assert categorical_encoder.col2label2idx == {'cat0': {'電玩星': 0, '時事星': 1, '放鬆星': 2}}

        # Case 2: enable_padding=True, enable_unknown=False, mode='LabelEncoding'
        categorical_encoder = CategoricalEncoder(col2label2idx={})
        output_data = categorical_encoder.encode_transform(list_data=input_data, col=col, all_cats=all_cats,
                                                           enable_padding=True, enable_unknown=False, mode='LabelEncoding')

        assert output_data == [[2], [3, 1], [1], [1, 2, 3], [0]]
        assert categorical_encoder.col2label2idx == {'cat0': {'PAD': 0, '電玩星': 1, '時事星': 2, '放鬆星': 3}}

        # Case 3: enable_padding=False, enable_unknown=True, mode='LabelEncoding'
        categorical_encoder = CategoricalEncoder(col2label2idx={})
        output_data = categorical_encoder.encode_transform(list_data=input_data, col=col, all_cats=all_cats,
                                                           enable_padding=False, enable_unknown=True, mode='LabelEncoding')

        assert output_data == [[2], [3, 1], [1], [1, 2, 3], [0]]
        assert categorical_encoder.col2label2idx == {'cat0': {'UNKNOWN': 0, '電玩星': 1, '時事星': 2, '放鬆星': 3}}

        # Case 4: enable_padding=True, enable_unknown=True, mode='LabelEncoding'
        categorical_encoder = CategoricalEncoder(col2label2idx={})
        output_data = categorical_encoder.encode_transform(list_data=input_data, col=col, all_cats=all_cats,
                                                           enable_padding=True, enable_unknown=True, mode='LabelEncoding')

        assert output_data == [[3], [4, 2], [2], [2, 3, 4], [1]]
        assert categorical_encoder.col2label2idx == {'cat0': {'PAD': 0, 'UNKNOWN': 1, '電玩星': 2, '時事星': 3, '放鬆星': 4}}

        # Case 5: enable_padding=False, enable_unknown=False, mode='VectorEncoding'
        categorical_encoder = CategoricalEncoder(col2label2idx={})
        output_data = categorical_encoder.encode_transform(list_data=input_data, col=col, all_cats=all_cats,
                                                           enable_padding=False, enable_unknown=False, mode='VectorEncoding')

        assert output_data == [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0],
                               [1.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                               [1.0, 0.0, 0.0]]
        assert categorical_encoder.col2label2idx == {'cat0': {'電玩星': 0, '時事星': 1, '放鬆星': 2}}

        # Case 6: enable_padding=True, enable_unknown=False, mode='VectorEncoding'
        categorical_encoder = CategoricalEncoder(col2label2idx={})
        output_data = categorical_encoder.encode_transform(list_data=input_data, col=col, all_cats=all_cats,
                                                           enable_padding=True, enable_unknown=False, mode='VectorEncoding')

        assert output_data == [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                               [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0],
                               [1.0, 0.0, 0.0, 0.0]]
        assert categorical_encoder.col2label2idx == {'cat0': {'PAD': 0, '電玩星': 1, '時事星': 2, '放鬆星': 3}}

        # Case 7: enable_padding=False, enable_unknown=True, mode='VectorEncoding'
        categorical_encoder = CategoricalEncoder(col2label2idx={})
        output_data = categorical_encoder.encode_transform(list_data=input_data, col=col, all_cats=all_cats,
                                                           enable_padding=False, enable_unknown=True, mode='VectorEncoding')

        assert output_data == [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                               [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0],
                               [1.0, 0.0, 0.0, 0.0]]
        assert categorical_encoder.col2label2idx == {'cat0': {'UNKNOWN': 0, '電玩星': 1, '時事星': 2, '放鬆星': 3}}

        # Case 8: enable_padding=True, enable_unknown=True, mode='VectorEncoding'
        categorical_encoder = CategoricalEncoder(col2label2idx={})
        output_data = categorical_encoder.encode_transform(list_data=input_data, col=col, all_cats=all_cats,
                                                           enable_padding=True, enable_unknown=True, mode='VectorEncoding')

        assert output_data == [[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0],
                               [0.0, 1.0, 0.0, 0.0, 0.0]]
        assert categorical_encoder.col2label2idx == {'cat0': {'PAD': 0, 'UNKNOWN': 1, '電玩星': 2, '時事星': 3, '放鬆星': 4}}


def test_encoding():

    # Case 1
    test_x = []
    test_length = 5
    expected_output = np.zeros(test_length)
    np.testing.assert_array_equal(encoding(test_x, test_length), expected_output)

    # Case 2
    test_x = [1, 2, 3]
    test_length = 5
    expected_output = np.array([0, 1, 1, 1, 0])
    np.testing.assert_array_equal(encoding(test_x, test_length), expected_output)


def test_get_embedding_interaction():
    from tests.config.example_data import USER_TITLE, ITEM_TITLE

    # Case 1: both of user_title and item_title is None
    try:
        get_embedding_interaction(None, None, option='dot_product') == 0.0

    except Exception as e:
        assert str(e) == 'Get Embedding Interaction Error'

    try:
        np.testing.assert_array_equal(get_embedding_interaction(None, None, option='elementwise_product'), np.zeros(300))
    except Exception as e:
        assert str(e) == 'Get Embedding Interaction Error'

    try:
        np.testing.assert_array_equal(get_embedding_interaction(None, None, option='sum'), np.zeros(300))
    except Exception as e:
        assert str(e) == 'Get Embedding Interaction Error'

    # Case 2: both of user_title and item_title is not None
    assert get_embedding_interaction(USER_TITLE, ITEM_TITLE, option='dot_product') == 0.3

    expected_output = ITEM_TITLE.copy()
    expected_output[1] = 0.3
    np.testing.assert_array_equal(get_embedding_interaction(USER_TITLE, ITEM_TITLE, option='elementwise_product'), expected_output)

    expected_output = USER_TITLE.copy()
    expected_output[1] += 1
    np.testing.assert_array_equal(get_embedding_interaction(USER_TITLE, ITEM_TITLE, option='sum'), expected_output)


def test_process_embedding():
    # Case 1: fillna=False
    test_data = {'title_embedding': [[1.0]*300, [0.0]*20], 'click_title_embedding': [[3.0]*300, [4.0]*300]}
    test_data = pd.DataFrame.from_dict(test_data)

    output_data = process_embedding(test_data, 'title_embedding', 'click_title_embedding', mode='prod')
    output_data = process_embedding(test_data, 'title_embedding', 'click_title_embedding', mode='dot')
    output_data = process_embedding(test_data, 'title_embedding', 'click_title_embedding', mode='sum')

    expected_output = {'title_embedding': [[1.0]*300, [0.0]*300], 'click_title_embedding': [[3.0]*300, [4.0]*300],
                       'semantics_prod': [[3.0]*300, [0.0]*300], 'semantics_dot': [900.0, 0.0], 'semantics_sum': [[4.0]*300, [4.0]*300]}
    expected_output = pd.DataFrame.from_dict(expected_output)
    pd.testing.assert_frame_equal(output_data, expected_output)


def test_convert_type():
    test_data = pd.DataFrame([{'cat0': "['放鬆星', '文學星', '電玩星']",
                               'content_ner': '{"PERSON-利善臻": 1.0, "PERSON-陳以芯": 1.0}',
                               'title_embedding': '[1,2,3,4]'}])
    test_configs = {'ast': ['cat0', 'content_ner'], 'json': ['title_embedding']}
    expected_output = pd.DataFrame([{'cat0': ['放鬆星', '文學星', '電玩星'],
                                     'content_ner': {"PERSON-利善臻": 1.0, "PERSON-陳以芯": 1.0},
                                     'title_embedding': [1, 2, 3, 4]}])

    pd.testing.assert_frame_equal(convert_type(test_data, test_configs), expected_output)


def test_convert_unit():

    test_data = pd.DataFrame([{'timestamp': '1664185876814'}])
    test_configs = {'millisecs_to_secs': ['timestamp']}
    expected_output = pd.DataFrame([{'timestamp': 1664185876.814}])

    pd.testing.assert_frame_equal(convert_unit(test_data, test_configs), expected_output)


def test_parse_content_ner():
    test_data = {'content_ner': {"GPE-南韓": 1.0, "GPE-釜山": 2.0, "NORP-韓媒": 1.0, "PERSON-楊穎軒": 1.0, "PERSON-權珉娥": 1.0, "PERSON-珉娥": 7.0}}
    test_data = pd.DataFrame(data=[test_data.values()], columns=['content_ner'])

    expected_output = {'content_ner': {"GPE-南韓": 1.0, "GPE-釜山": 2.0, "NORP-韓媒": 1.0, "PERSON-楊穎軒": 1.0, "PERSON-權珉娥": 1.0, "PERSON-珉娥": 7.0},
                       'content_ner_list': ['person', 'location', 'others']}
    expected_output = pd.DataFrame(data=[expected_output.values()], columns=['content_ner', 'content_ner_list'])

    pd.testing.assert_frame_equal(parse_content_ner(test_data, {'content_ner': 'content_ner_list'}), expected_output)


def test_cal_ner_matching():
    test_ner1 = {'PERSON-伊能靜': 1.0, 'PERSON-小哈利': 5.0, 'PERSON-庾澄慶': 1.0, 'LOCATION-台北': 1.0}
    test_ner2 = {'PERSON-伊能靜': 1.0, 'PERSON-庾澄慶': 1.0, 'LOCATION-台北': 1.0}

    assert cal_ner_matching(test_ner1, test_ner2) == 3
    assert cal_ner_matching(test_ner1, None) == 0
    assert cal_ner_matching(None, test_ner2) == 0


def test_cal_features_sim():
    test_feature1 = [0, 0, 0, 2]
    test_feature2 = [1, 1, 1, 1]

    assert cal_features_sim(mode='dot')(test_feature1, test_feature2) == 2
    assert cal_features_sim(mode='cos')(test_feature1, test_feature2) == 0.5

    test_feature1 = [0, 0, 0, 0]
    test_feature2 = [1, 1, 1, 1]

    assert cal_features_sim(mode='dot')(test_feature1, test_feature2) == 0
    assert cal_features_sim(mode='cos')(test_feature1, test_feature2) == 0


def test_cal_features_matching():
    test_feature1 = ['新奇', '生活', '頭條焦點', '即時']
    test_feature2 = ['新奇', '生活']
    assert cal_features_matching(test_feature1, test_feature2) == 2

    test_feature1 = ['新奇', '生活', '頭條焦點', '即時']
    test_feature2 = ['經濟', '政治']
    assert cal_features_matching(test_feature1, test_feature2) == 0

    test_feature1 = ['新奇', '生活', '頭條焦點', '即時']
    test_feature2 = []
    assert cal_features_matching(test_feature1, test_feature2) == 0

    test_feature1 = []
    test_feature2 = []
    assert cal_features_matching(test_feature1, test_feature2) == 0


def test_normalize_data():
    test_configs = {

        'min-max': {
            'price': None,
            'price2': 2
        },

        'z-score': {'price3': None}
    }

    input = {
        'price': [1, 2, 3],
        'price2': [1, 2, 3],
        'price3': [1, 2, 3]
    }
    data = pd.DataFrame(data=input)

    result = normalize_data(data, test_configs)

    expect = {
        'price': [0.0, 0.5, 1.0],
        'price2': [0.0, 1.0, 1.0],
        'price3': [-1.0, 0.0, 1.0]
    }
    expect_result = pd.DataFrame(data=expect)

    pd.testing.assert_frame_equal(result, expect_result)


def test_process_age():

    age_col = 'age'
    age_range = 20
    input_data = {age_col: [23, 22, '55', 80, 2, None, math.nan, '', np.nan]}
    expect_data = {age_col: ['1', '1', '2', '4', '0', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN']}

    df_input = pd.DataFrame(data=input_data)
    df_expect = pd.DataFrame(data=expect_data)

    df_output = process_age(df_input.copy(), columns=[age_col], age_range=age_range)
    pd.testing.assert_frame_equal(df_output, df_expect)

    df_output = process_age(df_input.copy(), columns=['UNKNOWN_COLUMN'], age_range=age_range)
    pd.testing.assert_frame_equal(df_input, df_output)


def test_process_gender():

    gender_col = 'gender'
    input_data = {gender_col: ['male', 'female', 'others', 'other', 'f', '123', 'm', None]}
    expect_data = {gender_col: ['male', 'female', 'other', 'other', 'female', 'other', 'male', 'UNKNOWN']}

    df_input = pd.DataFrame(data=input_data)
    df_expect = pd.DataFrame(data=expect_data)

    df_output = process_gender(df_input, columns=[gender_col])

    pd.testing.assert_frame_equal(df_output, df_expect)


def test_checknull():

    assert checknull(None)
    assert checknull(np.nan)
    assert checknull(math.nan)
    assert not checknull(23)
    assert not checknull('23')
    assert checknull('')


def test_get_publish_time_to_now():

    event_time_col = 'timestamp'
    publish_time_col = 'publish_time'
    id_col = 'content_id'
    min_to_current = 'min_to_current'
    hour_to_current = 'hour_to_current'

    input_data = {id_col: ['489059025678254080', '492339016226590720', '492339048887619584'],
                  publish_time_col: [1646118357999, 1646900368685, 1646900376472],
                  event_time_col:   [1650000000000, 1650000000000, 1650000000000]}

    expect_data = {id_col: ['489059025678254080', '492339016226590720', '492339048887619584'],
                   publish_time_col: [1646118357999, 1646900368685, 1646900376472],
                   event_time_col: [1650000000000, 1650000000000, 1650000000000],
                   min_to_current: [64694.03335, 51660.521916666665, 51660.392133333335],
                   hour_to_current: [1078.2338891666666, 861.0086986111111, 861.0065355555555]}

    df_input = pd.DataFrame(data=input_data)
    df_expect = pd.DataFrame(data=expect_data)
    df_output = get_publish_time_to_now(df_input, event_time_col='timestamp', publish_time_col='publish_time', time_type=['min', 'hour'])

    pd.testing.assert_frame_equal(df_output, df_expect)


def test_process_tag():

    tags_col = 'tags'
    input_data = {tags_col: [['p-大s', 'p-汪小菲', 'p-許雅鈞', 'p-霍建華', 'p-林心如', '大S', '汪小菲', '許雅鈞', '霍建華', '林心如', '', '-', 'p-']]}
    expect_data = {tags_col: [['大s', '汪小菲', '許雅鈞', '大S', '林心如', '霍建華']]}

    df_input = pd.DataFrame(data=input_data)
    df_expect = pd.DataFrame(data=expect_data)
    df_output = process_tag(df_input, columns=[tags_col])

    assert set(df_output[tags_col].iloc[0]) == set(df_expect[tags_col].iloc[0])


def test_merge_data_with_source():

    prefix = 'click_'

    candidate_data = {'content_id': ['content_id_0', 'content_id_1', 'content_id_2'],
                      'title': ['title_0', 'title_1', 'title_2']}
    source_data = {'content_id': 'content_id_999', 'title': 'title_999'}

    expect_data = {'click_content_id': ['content_id_0', 'content_id_1', 'content_id_2'],
                   'click_title': ['title_0', 'title_1', 'title_2'],
                   'content_id': ['content_id_999', 'content_id_999', 'content_id_999'],
                   'title': ['title_999', 'title_999', 'title_999']}

    ser_source = pd.Series(source_data)
    df_candidate = pd.DataFrame(data=candidate_data)
    df_expect = pd.DataFrame(data=expect_data)
    df_output = merge_data_with_source(dataset=df_candidate,
                                       source_data=ser_source,
                                       prefix=prefix,
                                       columns=[])

    df_expect = df_expect.reindex(sorted(df_expect.columns), axis=1)
    df_output = df_output.reindex(sorted(df_output.columns), axis=1)

    pd.testing.assert_frame_equal(df_output.reindex(sorted(df_output.columns), axis=1),
                                  df_expect.reindex(sorted(df_expect.columns), axis=1))

    expect_data = {'click_content_id': ['content_id_0', 'content_id_1', 'content_id_2'],
                   'content_id': ['content_id_999', 'content_id_999', 'content_id_999'],
                   'title': ['title_999', 'title_999', 'title_999']}
    df_expect = pd.DataFrame(data=expect_data)
    df_output = merge_data_with_source(dataset=df_candidate,
                                       source_data=ser_source,
                                       prefix=prefix,
                                       columns=['content_id'])

    pd.testing.assert_frame_equal(df_output.reindex(sorted(df_output.columns), axis=1),
                                  df_expect.reindex(sorted(df_expect.columns), axis=1))
