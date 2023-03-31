from src_v2.configs import GeneralConfigs
from src_v2.configs.models.din import BaseDINConfigs
from src_v2.configs.preprocess import BaseUser2ItemPreprocessConfigs
from src.preprocess.utils.encoder import UNKNOWN_LABEL


PlanetNewsUser2ItemGeneralConfig = GeneralConfigs.update({
    'REQUISITE_COLS': [
        'gender', 'age',
        'cat0', 'cat1', 'tags',
        'total_click_count', 'total_view_count', 'popularity_score',
        'hist_cat0', 'seq_length_cat0', 'hist_cat1', 'seq_length_cat1',
        'user_title_embedding', 'item_title_embedding',
        'cat0_pref_score', 'cat1_pref_score', 'user_tag_editor_others',
        'user_tag_editor_person', 'user_tag_editor_event',
        'user_tag_editor_organization', 'user_tag_editor_location',
        'hour_time_period',
        'y'],
    'CONTENT_TYPE': 'planet_news',
    'SERVICE_TYPE': 'user2item',
    'NDCG_GROUPBY_KEY': 'userid',
    'MONITOR_METRICS_TO_THRESHOLD_MAPPING': {'auc': 10, 'ndcg5': 10, 'ndcg10': 10, 'ndcg20': 10},
    'ALERT_MAIL_RECIPIENTS': 'xiangchen@gamania.com'
})

PlanetNewsUser2ItemDINModelConfig = BaseDINConfigs.update({
    'DENSE_FEATURE_SIZE': {
        'total_click_count': 1, 'total_view_count': 1, 'popularity_score': 1,
        'cat0_pref_score': 1, 'cat1_pref_score': 1,
        'user_tag_editor_others': 1, 'user_tag_editor_person': 1, 'user_tag_editor_event': 1,
        'user_tag_editor_organization': 1, 'user_tag_editor_location': 1,
        'user_title_embedding': 300, 'item_title_embedding': 300,
        'hour_time_period': 1
    },

    'DENSE_FEATURE': [
        'total_click_count', 'total_view_count', 'popularity_score',
        'cat0_pref_score', 'cat1_pref_score',
        'user_tag_editor_others', 'user_tag_editor_person', 'user_tag_editor_event',
        'user_tag_editor_organization', 'user_tag_editor_location',
        'user_title_embedding', 'item_title_embedding',
        'hour_time_period'
    ],

    'FEATURE_EMBEDDING_SIZE': {'age': 6, 'gender': 2, 'cat0': 32, 'cat1': 64},
    'ONE_HOT_FEATURE': ['age', 'gender', 'cat0', 'cat1'],

    'BEHAVIOR_FEATURE': ['cat0', 'cat1'],
    'BEHAVIOR_FEATURE_SEQ_LENGTH': ['seq_length_cat0', 'seq_length_cat1'],
    'BEHAVIOR_SEQUENCE_SIZE': 5
})

PlanetNewsUser2ItemDINPreprocessConfig = BaseUser2ItemPreprocessConfigs.update({

    # format: {mode: [col1, col2]}
    # only support 'ast', 'json', 'int', 'float' mode
    'TYPE_CONVERT_MODE2COLS': {
        'ast': ['cat0', 'cat1', 'tags'],
        'json': ['user_title_embedding', 'item_title_embedding'],
        'int64': ['publish_time', 'timestamp', 'total_click_count', 'total_view_count'],
        'float64': ['popularity_score']
    },

    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; mode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    'COL2CATS_NAMES': 'col2label.pickle',
    'CATEGORY_FEATURES_PROCESS': {
        'age': [False, True, 'LabelEncoding'],
        'gender': [False, True, 'LabelEncoding'],
        'cat0': [True, True, 'LabelEncoding'],
        'cat1': [True, True, 'LabelEncoding'],
    },

    # format: {old_name: new_name}
    'COLUMN_TO_RENAME': {},

    # format: {col_name: value_to_be_filled}
    'COLUMN_TO_FILLNA': {
        'gender': UNKNOWN_LABEL,
        'user_category': '{}',
        'cat0': '[]',
        'cat1': '[]',
        'tags': '[]',
        'total_click_count': 0,
        'total_view_count': 0,
        'popularity_score': 0.0,
        'user_title_embedding': [0.0] * 300,
        'item_title_embedding': [0.0] * 300,
    },


    # support min-max or z-score (Need to be optimized. Currently normalize method is too hacky)
    'NORMALIZE_COLS': {
        'min-max': {'hour_time_period': 30 * 24},
        'z-score': {}
    },

    'UNIX_TIME_COLS': ['publish_time', 'timestamp'],

    'CATEGORY_COLS': ['cat0', 'cat1'],

    # aggregate preference
    'CATEGORY_PREF_SCORE_PROCESS': [{'level': ['click', 'cat0'], 'cat_col': 'user_category', 'score_col': 'cat0_pref_score'},
                                    {'level': ['click', 'cat1'], 'cat_col': 'user_category', 'score_col': 'cat1_pref_score'}],

    'TAG_PREF_SCORE_PROCESS': [{
        'tag_entity_list': ['others', 'person', 'event', 'organization', 'location'],
        'user_tag_col': 'user_tag',
        'item_tag_col': 'tags',
        'tagging_type': 'editor',
        'score_col': ''
    }],

    # process_user_behavior_sequence: (only support profile format of {'click': {'cat1': {}}}, ex: cat/tag profile)
    # format: {col_name: [[enc_name, event_key, cat_level, hist_suffix, seq_len_suffix], ...]}
    'BEHAVIOR_SEQUENCE_FEATURES_PROCESS': {
        'user_category': [['cat0', 'click', 'cat0', 'hist_', 'seq_length_'],
                          ['cat1', 'click', 'cat1', 'hist_', 'seq_length_']]
    },

    # append_user_data: it's needed in inference mode
    # format: {'user_profile': {profile_name: [(is_append_all_profile, new_col_name, profile_col_name), ...], ...}
    #          'realtime_user_profile': {profile_name: [(is_append_all_profile, new_col_name, profile_col_name), ...], ...}}
    # if is_append_all_profile is "True", it will create a new_col_name to data and append the whole profile to it
    'APPEND_USER_DATA_PROCESS': {'user_profile': {'userMetaProfile': [(False, 'age', 'age'), (False, 'gender', 'gender')],
                                                  'planetNewsCategoryProfile': [(True, 'user_category', 'user_category')],
                                                  'planetNewsTagProfile': [(True, 'user_tag', 'user_tag')],
                                                  'planetNewsEmbeddingProfile': [(False, 'user_title_embedding', 'title')]},

                                 'realtime_user_profile': {}},

    'STATISTICS_COLS': ['last_1_day_total_view_count', 'last_1_day_total_click_count',
                        'last_3_day_total_view_count', 'last_3_day_total_click_count',
                        'last_7_day_total_view_count', 'last_7_day_total_click_count',
                        'last_30_day_total_view_count', 'last_30_day_total_click_count'],

    # chain_configs
    # holds the pipeline of preprocessor for anything you want it to.
    # CHAIN_CONFIGS format: {'func_name': {'func_param_key': 'func_param_value'}}
    'CHAIN_CONFIGS': {'append_user_data': {'disable_mode': ['train', 'validation']},
                      'append_metrics_data': {'disable_mode': ['train', 'validation']},
                      'convert_columns_name': {},
                      'process_age': {},
                      'handle_missing_data': {},
                      'convert_data_type': {},
                      'process_tag': {},
                      'process_category': {},
                      'aggregate_preference': {},
                      'encode_features': {},
                      'process_time_period': {},
                      'normalize_data': {},
                      'process_user_behavior_sequence': {}}
})

PlanetNewsUser2ItemDINConfig = PlanetNewsUser2ItemGeneralConfig.update(PlanetNewsUser2ItemDINModelConfig) \
                                                               .update(PlanetNewsUser2ItemDINPreprocessConfig)
