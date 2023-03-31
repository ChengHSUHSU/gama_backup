from bdds_recommendation.src.configs import BaseUser2itemConfig
from bdds_recommendation.src.preprocess.utils.encoder import UNKNOWN_LABEL


class NownewsNewsUser2ItemDINConfig(BaseUser2itemConfig):

    COLUMN_TO_FILLNA = {'gender': UNKNOWN_LABEL,
                        'user_category': '{}'}

    BEHAVIOR_SEQUENCE_SIZE = 10
    SEMANTICS_EMBEDDING_SIZE = 300
    FEATURE_EMBEDDING_SIZE = {'age': 8, 'gender': 2, 'category_name': 48}

    DENSE_FEATURE = ['category_pref_score', 'user_tag_editor_others']
    ONE_HOT_FEATURE = ['age', 'gender', 'category_name']

    BEHAVIOR_FEATURE = ['category_name']
    BEHAVIOR_FEATURE_SEQ_LENGTH = ['seq_length_category_name']

    # process user behavior sequence (only support profile format of {'click': {'cat1': {}}}, ex: cat/tag profile)
    # format: {col_name: [[encoder_key, event_key, cat_level], ...]}
    BEHAVIOR_SEQUENCE_FEATURES_PROCESS = {
        'user_category': [['category_name', 'click', 'cat1']]
    }

    SEMANTICS_FEATURE = ['user_title_embedding', 'item_title_embedding']
    SEMANTICS_INPUT = ['semantics_prod']

    TAG_ENTITY_LIST = ['others']    # nownews editor didn't support named entity tagging

    COLUMN_NEED_PARSE = ['tags']
    COLUMN_TO_RENAME = {}

    SEMANTICS_INTERACTION_MODE = ['prod']
    TYPE_CONVERT_MODE2COLS = {'ast': ['cat1'], 'json': ['user_title_embedding', 'item_title_embedding']}

    NORMALIZE_COLS = {'min-max': {'hour_to_current': 14 * 24}, 'z-score': {}}

    REQUISITE_COLS = [
        'gender', 'age', 'hour_to_current', 'semantics_prod',
        'category_name', 'hist_category_name', 'seq_length_category_name',
        'category_pref_score', 'user_tag_editor_others',
        'y'
    ]

    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; moode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    COL2CATS_NAMES = 'col2label.pickle'

    CATEGORY_FEATURES_PROCESS = {
        'age': [False, True, 'LabelEncoding'],
        'gender': [False, True, 'LabelEncoding'],
        'category_name': [True, True, 'LabelEncoding'],
        }
