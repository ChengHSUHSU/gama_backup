from . import BaseUser2itemConfig
from src.preprocess.utils.encoder import UNKNOWN_LABEL, PAD_LABEL


class JollybuyGoodsUser2ItemDINConfig(BaseUser2itemConfig):

    CONTENT_TYPE = 'jollybuy_goods'

    BEHAVIOR_SEQUENCE_SIZE = 10
    SEMANTICS_EMBEDDING_SIZE = 300
    FEATURE_EMBEDDING_SIZE = {'age': 8, 'gender': 2, 'cat0': 17, 'cat1': 100}

    # For DIN model
    DENSE_FEATURE = ['final_score', 'cat0_pref_score', 'cat1_pref_score', 'tag_other_pref_score']
    ONE_HOT_FEATURE = ['age', 'gender', 'cat0', 'cat1']
    SEMANTICS_INPUT = ['user_title_embedding', 'item_title_embedding']
    BEHAVIOR_FEATURE = ['cat0', 'cat1']
    BEHAVIOR_FEATURE_SEQ_LENGTH = ['seq_length_cat0', 'seq_length_cat1']

    TAG_ENTITY_LIST = ['others']

    ### Preprocesser part ###
    # we're not convert "age" type here cause it would be handled when transforming it to discrete/categorical feature
    TYPE_CONVERT_MODE2COLS = {
        'ast': ['cat0', 'cat1'],
        'json': ['user_title_embedding', 'item_title_embedding'],
        'float': ['final_score']}

    # format: {old_name: new_name}
    COLUMN_TO_RENAME = {}

    # format: {col_name: value_to_be_filled}
    COLUMN_TO_FILLNA = {
        'gender': UNKNOWN_LABEL,
        'cat0': [PAD_LABEL],
        'cat1': [PAD_LABEL],
        'final_score': 0.0,
        'user_title_embedding': [0.0] * SEMANTICS_EMBEDDING_SIZE,
        'item_title_embedding': [0.0] * SEMANTICS_EMBEDDING_SIZE,
        'user_category': '{}'
    }

    # preference columns
    USER_CAT_COL = 'user_category'
    CAT0_PREF_COL = 'cat0_pref_score'
    CAT1_PREF_COL = 'cat1_pref_score'

    USER_TAG_COL = 'user_tag'
    ITEM_TAG_COL = 'tags'
    TAG_PREF_COL = 'tag_other_pref_score'

    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; moode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    COL2CATS_NAMES = 'col2label.pickle'

    CATEGORY_FEATURES_PROCESS = {
        'age': [False, True, 'LabelEncoding'],
        'gender': [False, True, 'LabelEncoding'],
        'tags': [False, True, 'LabelEncoding'],
        'cat0': [True, True, 'LabelEncoding'],
        'cat1': [True, True, 'LabelEncoding']}

    # process user behavior sequence (only support profile format of {'click': {'cat1': {}}}, ex: cat/tag profile)
    # format: {col_name: [[cat_name, event_key, cat_level], ...]}
    BEHAVIOR_SEQUENCE_FEATURES_PROCESS = {
        'user_category': [['cat0', 'click', 'cat0'], ['cat1', 'click', 'cat1']]
    }

    REQUISITE_COLS = ['age', 'gender', 'cat0', 'cat1', 'final_score',
                      'user_title_embedding', 'item_title_embedding',
                      'cat0_pref_score', 'cat1_pref_score', 'tag_other_pref_score',
                      'hist_cat0', 'seq_length_cat0', 'hist_cat1', 'seq_length_cat1',
                      'y']

    # support min-max or z-score (Need to be optimized. Currently normalize method is too hacky)
    NORMALIZE_COLS = {
        'min-max': {},  # set val for upper bound normalize, set val None to standard min-max normalize
        'z-score': {}
    }

    NDCG_GROUPBY_KEY = 'userid'

    MONITOR_METRICS_TO_THRESHOLD_MAPPING = {'auc': 10, 'ndcg5': 10, 'ndcg10': 10, 'ndcg20': 10}
    ALERT_MAIL_RECIPIENTS = 'leoliou@gamania.com'
