from . import BaseUser2itemConfig
from src.preprocess.utils.encoder import UNKNOWN_LABEL


class PlanetNewsUser2ItemDINConfig(BaseUser2itemConfig):

    CONTENT_TYPE = 'planet_news'

    COLUMN_TO_FILLNA = {'gender': UNKNOWN_LABEL,
                        'user_category': '{}'}

    BEHAVIOR_SEQUENCE_SIZE = 10
    SEMANTICS_EMBEDDING_SIZE = 128
    FEATURE_EMBEDDING_SIZE = {'age': 8, 'gender': 2, 'site_name': 2, 'cat1': 32}

    DENSE_FEATURE = ['category_pref_score', 'user_tag_editor_others', 'user_tag_editor_person',
                     'user_tag_editor_event', 'user_tag_editor_organization', 'user_tag_editor_location']

    ONE_HOT_FEATURE = ['age', 'gender', 'site_name', 'cat1']

    BEHAVIOR_FEATURE = ['cat1']
    BEHAVIOR_FEATURE_SEQ_LENGTH = ['seq_length_cat1']

    # process user behavior sequence (only support profile format of {'click': {'cat1': {}}}, ex: cat/tag profile)
    # format: {col_name: [[cat_name, event_key, cat_level], ...]}
    BEHAVIOR_SEQUENCE_FEATURES_PROCESS = {
        'user_category': [['cat1', 'click', 'cat1']]
    }

    SEMANTICS_FEATURE = ['user_title_embedding', 'item_title_embedding']
    SEMANTICS_INPUT = ['user_title_embedding', 'item_title_embedding']   # raw semantics input

    TAG_ENTITY_LIST = ['others', 'person', 'event', 'organization', 'location']

    COLUMN_NEED_PARSE = ['tags']    # support `tags` and `content_ner`
    COLUMN_TO_RENAME = {'cat1_0': 'cat1'}

    SEMANTICS_INTERACTION_MODE = ['prod']
    TYPE_CONVERT_MODE2COLS = {'ast': [], 'json': ['user_title_embedding', 'item_title_embedding']}

    REQUISITE_COLS = ['gender', 'site_name', 'age', 'user_title_embedding',
                      'item_title_embedding', 'cat1', 'hist_cat1', 'seq_length_cat1',
                      'category_pref_score', 'user_tag_editor_others',
                      'user_tag_editor_person', 'user_tag_editor_event',
                      'user_tag_editor_organization', 'user_tag_editor_location',
                      'y']

    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; moode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    COL2CATS_NAMES = 'col2label.pickle'

    CATEGORY_FEATURES_PROCESS = {
        'age': [False, True, 'LabelEncoding'],
        'gender': [False, True, 'LabelEncoding'],
        'cat1': [True, True, 'LabelEncoding'],
        'site_name': [True, True, 'LabelEncoding']}

    NDCG_GROUPBY_KEY = 'openid'     # TODO: change to userid next iteration

    MONITOR_METRICS_TO_THRESHOLD_MAPPING = {'auc': 10, 'ndcg5': 10, 'ndcg10': 10, 'ndcg20': 10}
    ALERT_MAIL_RECIPIENTS = 'xiangchen@gamania.com'


class PlanetVideoUser2ItemLGBMConfig(BaseUser2itemConfig):

    ONE_HOT_FEATURE = ['gender', 'site_name', 'cat1']
    SEMANTICS_FEATURE = [['user_video_title_embedding', 'item_title_embedding'], ['user_news_title_embedding', 'item_title_embedding']]

    TAG_ENTITY_LIST = ['others', 'person', 'event', 'organization', 'location']

    COLUMN_NEED_PARSE = ['tags', 'content_ner']
    COLUMN_TO_RENAME = {'cat1_0': 'cat1', 'user_video_tag': 'user_tag'}

    SEMANTICS_INTERACTION_MODE = ['dot']
    TYPE_CONVERT_MODE2COLS = {'ast': ['tags'], 'json': ['user_video_title_embedding', 'user_news_title_embedding', 'item_title_embedding']}

    REQUISITE_COLS = [
        'openid', 'uuid', 'gender', 'age', 'site_name', 'cat1',
        'video_semantics_dot', 'news_semantics_dot',
        'video_category_pref_score', 'news_category_pref_score',
        'user_tag_editor_others', 'user_tag_editor_person', 'user_tag_editor_event',
        'user_tag_editor_organization', 'user_tag_editor_location',
        'user_tag_ner_others', 'user_tag_ner_person', 'user_tag_ner_event',
        'user_tag_ner_organization', 'user_tag_ner_location', 'y'
    ]
    ID_COLS = ['openid', 'uuid']
    COL2CATS_NAMES = 'col2label.pickle'
    CATEGORY_FEATURES_PROCESS = {
        'age': [False, True, 'NumericalOneHotEncoding'],
        'gender': [False, True, 'NumericalOneHotEncoding'],
        'cat1': [True, True, 'NumericalOneHotEncoding'],
        'site_name': [True, True, 'NumericalOneHotEncoding']}


class PlanetComicsBookUser2ItemLGBMConfig(BaseUser2itemConfig):

    ONE_HOT_FEATURE = ['gender', 'cat1']
    SEMANTICS_FEATURE = ['user_title_embedding', 'item_title_embedding']
    SEMANTICS_INTERACTION_MODE = ['dot']
    TAG_ENTITY_LIST = ['others', 'person', 'event', 'organization', 'location']
    COLUMN_NEED_PARSE = ['tags', 'content_ner', 'cat2']
    COLUMN_TO_RENAME = {}

    TYPE_CONVERT_MODE2COLS = {'ast': ['cat1'], 'json': ['user_title_embedding', 'item_title_embedding']}

    REQUISITE_COLS = [
        # TODO: consider following mocked feature
        # 'is_adult', 'author_penname', 'provider_nickname', 'user_tag_editor_others', 'user_tag_editor_person'
        # 'user_tag_editor_event', 'user_tag_editor_organization', 'user_tag_editor_location',
        'userid', 'uuid', 'age', 'gender', 'semantics_dot', 'category_pref_score', 'cat1',
        'read_count_repeat', 'read_count_norepeat', 'like_count', 'y'
    ]
    ID_COLS = ['userid', 'uuid']
    COL2CATS_NAMES = 'col2label.pickle'
    CATEGORY_FEATURES_PROCESS = {
        'age': [False, True, 'NumericalOneHotEncoding'],
        'gender': [False, True, 'NumericalOneHotEncoding'],
        'cat1': [True, True, 'NumericalOneHotEncoding']}
    MONITOR_METRICS_TO_THRESHOLD_MAPPING = {'auc': 10, 'ndcg5': 10, 'ndcg10': 10, 'ndcg20': 10}
    ALERT_MAIL_RECIPIENTS = 'yorkchien@gamania.com,zaidentseng@gamania.com'


class PlanetNovelBookUser2ItemDINConfig(BaseUser2itemConfig):

    COLUMN_TO_FILLNA = {'gender': UNKNOWN_LABEL,
                        'series': UNKNOWN_LABEL,
                        'user_category': '{}'}

    # preprocess configs
    TYPE_CONVERT_MODE2COLS = {
        'ast': ['cat1', 'author', 'author_penname', 'provider', 'show_platform', 'renewal'],
        'json': ['user_title_embedding', 'user_tag_embedding', 'item_title_embedding']}

    COLUMN_TO_RENAME = {}

    EMBEDDING_COL_LIST = ['user_title_embedding', 'user_tag_embedding', 'item_title_embedding']

    TAG_ENTITY_LIST = ['others', 'person', 'event', 'organization', 'location']

    ONE_HOT_FEATURE = ['age', 'gender', 'cat1', 'series', 'provider']
    COLUMN_NEED_PARSE = ['cat1', 'author_penname', 'provider', 'show_platform', 'renewal']

    BEHAVIOR_SEQUENCE_SIZE = 10

    NORMALIZE_COLS = {'min-max': {'hour_to_current': 14 * 24}, 'z-score': {}}

    # execution configs
    REQUISITE_COLS = [
        'gender', 'age', 'hour_to_current',
        'user_title_embedding', 'user_tag_embedding', 'item_title_embedding',
        'cat1', 'hist_cat1', 'seq_length_cat1', 'category_pref_score',
        'series', 'provider', 'y'
    ]
    NDCG_GROUPBY_KEY = 'userid'

    # training configs
    FEATURE_EMBEDDING_SIZE = {'age': 4,
                              'gender': 2,
                              'cat1': 8,
                              'series': 64,
                              'author_penname': 64,
                              'provider': 8,
                              'show_platform': 2,
                              'renewal': 4}

    DENSE_FEATURE = ['category_pref_score']

    SEMANTICS_EMBEDDING_SIZE = 300
    SEMANTICS_INPUT = ['user_title_embedding', 'user_tag_embedding', 'item_title_embedding']

    BEHAVIOR_FEATURE = ['cat1']
    BEHAVIOR_FEATURE_SEQ_LENGTH = ['seq_length_cat1']

    # process user behavior sequence (only support profile format of {'click': {'cat1': {}}}, ex: cat/tag profile)
    # format: {col_name: [[cat_name, event_key, cat_level], ...]}
    BEHAVIOR_SEQUENCE_FEATURES_PROCESS = {
        'user_category': [['cat1', 'click', 'cat1']]
    }

    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; moode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    COL2CATS_NAMES = 'col2label.pickle'

    CATEGORY_FEATURES_PROCESS = {
        'age': [False, True, 'LabelEncoding'],
        'gender': [False, True, 'LabelEncoding'],
        'series': [False, True, 'LabelEncoding'],
        'provider': [False, True, 'LabelEncoding'],
        'cat1': [True, True, 'LabelEncoding']}
