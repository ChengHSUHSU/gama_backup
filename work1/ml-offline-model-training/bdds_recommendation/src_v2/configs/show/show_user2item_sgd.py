from bdds_recommendation.src_v2.configs import GeneralConfigs
from bdds_recommendation.src_v2.configs.models.din import BaseDINConfigs
from bdds_recommendation.src_v2.configs.preprocess import BaseUser2ItemPreprocessConfigs
from bdds_recommendation.src.preprocess.utils.encoder import UNKNOWN_LABEL



ShowUser2ItemGeneralConfig = GeneralConfigs.update({
    'REQUISITE_COLS': [
        'cat0_pref_score',
        'hour_time_period',
        'popularity_score',
        '1_last_day_total_view_count', '1_last_day_total_click_count',
        '3_last_day_total_view_count', '3_last_day_total_click_count',
        '5_last_day_total_view_count', '5_last_day_total_click_count', 'y'],
    'CONTENT_TYPE': 'planet_news',
    'SERVICE_TYPE': 'user2item',
    'NDCG_GROUPBY_KEY': 'userid',
    'MONITOR_METRICS_TO_THRESHOLD_MAPPING': {'auc': 10, 'ndcg5': 10, 'ndcg10': 10, 'ndcg20': 10},
    'ALERT_MAIL_RECIPIENTS': 'xiangchen@gamania.com'
})

ShowUser2ItemDINModelConfig = BaseDINConfigs.update({
    'DENSE_FEATURE_SIZE': {
        'cat0_pref_score': 1,
        'hour_time_period': 1,
        '1_last_day_total_view_count':1, '1_last_day_total_click_count':1,
        '3_last_day_total_view_count':1, '3_last_day_total_click_count':1,
        '5_last_day_total_view_count':1, '5_last_day_total_click_count':1,
        'popularity_score':1, 'seq_length_cat0':1, 'hist_cat0':1
    },

    'DENSE_FEATURE': [
        'cat0_pref_score',
        'hour_time_period',
        '1_last_day_total_view_count', '1_last_day_total_click_count',
        '3_last_day_total_view_count', '3_last_day_total_click_count',
        '5_last_day_total_view_count', '5_last_day_total_click_count', 
        'popularity_score', 'seq_length_cat0', 'hist_cat0'
    ],

    'SEMANTICS_EMBEDDING_SIZE' : 300,
    'SEMANTICS_INPUT' : ['user_title_embedding', 'item_title_embedding'],   # method two : raw semantics input
    'SEMANTICS_FEATURE' : {'show': ('user_title_embedding', 'item_title_embedding')},


    'FEATURE_EMBEDDING_SIZE': {'age': 6, 'gender': 2, 'cat0': 32},
    'ONE_HOT_FEATURE': ['age', 'gender', 'cat0'],

    'BEHAVIOR_FEATURE': ['cat0'],
    'BEHAVIOR_FEATURE_SEQ_LENGTH': ['seq_length_cat0'],
    'BEHAVIOR_SEQUENCE_SIZE': 5
})

ShowUser2ItemDINPreprocessConfig = BaseUser2ItemPreprocessConfigs.update({

    # format: {mode: [col1, col2]}
    # only support 'ast', 'json', 'int', 'float' mode
    'TYPE_CONVERT_MODE2COLS': {
        'ast': ['cat0'], #'user_title_embedding', 'item_title_embedding'
        'int64': ['publish_time', 'timestamp',
                '1_last_day_total_view_count', '1_last_day_total_click_count',
        '3_last_day_total_view_count', '3_last_day_total_click_count',
        '5_last_day_total_view_count', '5_last_day_total_click_count'],
    },
    'TYPE_CONVERT_MODE2COLS_INFERENCE': {
        'ast': ['user_title_embedding', 'item_title_embedding'],
    },

    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; mode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    'COL2CATS_NAMES': 'col2label.pickle',
    'CATEGORY_FEATURES_PROCESS': {
        'age': [False, True, 'LabelEncoding'],
        'gender': [False, True, 'LabelEncoding'],
        'cat0': [True, True, 'LabelEncoding']
    },

    # format: {old_name: new_name}
    'COLUMN_TO_RENAME': {'title_embedding': 'item_title_embedding', 'final_score': 'popularity_score'},

    # format: {col_name: value_to_be_filled}
    'COLUMN_TO_FILLNA': {
        'gender': UNKNOWN_LABEL,
        'user_category': '{}',
        'cat0': '[]',
        'user_title_embedding': [0.0] * 300,
        'item_title_embedding': [0.0] * 300,
        'popularity_score': 0.0,
        '1_last_day_total_view_count': 0.0, 
        '1_last_day_total_click_count': 0.0,
        '3_last_day_total_view_count': 0.0, 
        '3_last_day_total_click_count': 0.0,
        '5_last_day_total_view_count':0.0,
        '5_last_day_total_click_count': 0.0
    },

    # support min-max or z-score (Need to be optimized. Currently normalize method is too hacky)
    'NORMALIZE_COLS': {
        'min-max': {'hour_time_period': 30 * 24},
        'z-score': {}
    },

    'UNIX_TIME_COLS': ['publish_time', 'timestamp'],
    'CATEGORY_COLS': ['cat0'],

    # aggregate preference
    'CATEGORY_PREF_SCORE_PROCESS': [{'level': ['click', 'cat0'], 'cat_col': 'user_category', 'score_col': 'cat0_pref_score'}],

    'TAG_PREF_SCORE_PROCESS': [{
        'score_col': ''
    }],

    # process_user_behavior_sequence: (only support profile format of {'click': {'cat1': {}}}, ex: cat/tag profile)
    # format: {col_name: [[enc_name, event_key, cat_level, hist_suffix, seq_len_suffix], ...]}
    'BEHAVIOR_SEQUENCE_FEATURES_PROCESS': {
        'user_category': [['cat0', 'click', 'cat0', 'hist_', 'seq_length_']]
    },

    # append_user_data: it's needed in inference mode
    # format: {'user_profile': {profile_name: [(is_append_all_profile, new_col_name, profile_col_name), ...], ...}
    #          'realtime_user_profile': {profile_name: [(is_append_all_profile, new_col_name, profile_col_name), ...], ...}}
    # if is_append_all_profile is "True", it will create a new_col_name to data and append the whole profile to it
    'APPEND_USER_DATA_PROCESS': {'user_profile': {'userMetaProfile': [(False, 'age', 'age'), (False, 'gender', 'gender')],
                                                  'planetNewsCategoryProfile': [(True, 'user_category', 'user_category')],
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
                      'append_timestamp': {'disable_mode': ['train', 'validation']},
                      'convert_columns_name': {},
                      'handle_missing_data': {},
                      'convert_data_type': {},
                      'process_category': {'disable_mode': ['inference']},
                      'aggregate_preference': {},
                      'encode_features': {},
                      'process_time_period': {},
                      'normalize_data': {},
                      'process_user_behavior_sequence': {}}

})

ShowUser2ItemSGDConfig = ShowUser2ItemGeneralConfig.update(ShowUser2ItemDINModelConfig).update(ShowUser2ItemDINPreprocessConfig)
