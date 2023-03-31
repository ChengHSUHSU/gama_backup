from bdds_recommendation.src_v2.configs import GeneralConfigs
from bdds_recommendation.src_v2.configs.models.din import BaseDINConfigs
from bdds_recommendation.src_v2.configs.preprocess import BaseUser2ItemPreprocessConfigs
from bdds_recommendation.src.preprocess.utils.encoder import UNKNOWN_LABEL


PlanetNovelUser2ItemGeneralConfig = GeneralConfigs.update({
    'REQUISITE_COLS': [
        'gender', 'age', 'hour_to_current',
        'user_title_embedding', 'user_tag_embedding', 'item_title_embedding',
         'hist_cat1', 'seq_length_cat1', 'category_pref_score', 'cat1',
        'series', 'provider', 'y'
    ],
    'CONTENT_TYPE': 'planet_novel',
    'SERVICE_TYPE': 'user2item',
    'NDCG_GROUPBY_KEY': 'userid',
    'MONITOR_METRICS_TO_THRESHOLD_MAPPING': {'auc': 10, 'ndcg5': 10, 'ndcg10': 10, 'ndcg20': 10},
    'ALERT_MAIL_RECIPIENTS': 'zaidengtseng@gamania.com,alberthsu@gamania.com'
})

PlanetNovelUser2ItemDINModelConfig = BaseDINConfigs.update({
    'COLUMN_TO_RENAME': {},
    'EMBEDDING_COL_LIST': ['user_title_embedding', 'user_tag_embedding', 'item_title_embedding'],

    'SEMANTICS_EMBEDDING_SIZE': 300,
    'SEMANTICS_INPUT': ['user_title_embedding', 'user_tag_embedding', 'item_title_embedding'],

    'DENSE_FEATURE_SIZE': {
        'category_pref_score': 1
    },
    'DENSE_FEATURE': ['category_pref_score'],

    'FEATURE_EMBEDDING_SIZE': {'age': 4,
                              'gender': 2,
                              'cat1': 8,
                              'series': 64,
                              'author_penname': 64,
                              'provider': 8,
                              'show_platform': 2,
                              'renewal': 4
                            },
    'ONE_HOT_FEATURE': ['age', 'gender', 'cat1', 'series', 'provider'],

    'BEHAVIOR_FEATURE': ['cat1'],
    'BEHAVIOR_FEATURE_SEQ_LENGTH': ['seq_length_cat1'],
    'BEHAVIOR_SEQUENCE_SIZE': 10
})

PlanetNovelUser2ItemDINPreprocessConfig = BaseUser2ItemPreprocessConfigs.update({

    # format: {mode: [col1, col2]}
    # only support 'ast', 'json', 'int', 'float' mode
    'TYPE_CONVERT_MODE2COLS': {
        'ast': ['cat1', 'author', 'author_penname', 'provider', 'show_platform', 'renewal'],
        'json': ['user_title_embedding', 'user_tag_embedding', 'item_title_embedding']},

    'TYPE_CONVERT_MODE2COLS_INFERENCE': {
        'ast': ['tags'],
        'json': ['user_title_embedding', 'item_title_embedding', 'user_tag_embedding'],
    },
    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; mode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    'COL2CATS_NAMES': 'col2label.pickle',
    'CATEGORY_FEATURES_PROCESS': {
        'age': [False, True, 'LabelEncoding'],
        'gender': [False, True, 'LabelEncoding'],
        'series': [False, True, 'LabelEncoding'],
        'provider': [False, True, 'LabelEncoding'],
        'cat1': [True, True, 'LabelEncoding']},

    # format: {old_name: new_name}
    'COLUMN_TO_RENAME': {'title_embedding': 'item_title_embedding'},

    # format: {col_name: value_to_be_filled}
    'COLUMN_TO_FILLNA': {
        'gender': UNKNOWN_LABEL,
        'series': UNKNOWN_LABEL,
        'user_category': '{}',
        'user_title_embedding': [0.0] * 300,
        'user_tag_embedding': [0.0] * 300,
        'cat1': ''
        # 'cat1': lambda x: x if len(eval(x)) > 0 else [UNKNOWN_LABEL]
    },
    # aggregate preference
    'CATEGORY_PREF_SCORE_PROCESS': [{'level': ['click', 'cat1'], 'cat_col': 'user_category', 'score_col': 'category_pref_score'}],

    # support min-max or z-score (Need to be optimized. Currently normalize method is too hacky)
    'NORMALIZE_COLS': {'min-max': {'hour_to_current': 14 * 24}, 'z-score': {}},
    'CATEGORY_COLS': ['cat1'],

    # process_user_behavior_sequence: (only support profile format of {'click': {'cat1': {}}}, ex: cat/tag profile)
    # format: {col_name: [[enc_name, event_key, cat_level, hist_suffix, seq_len_suffix], ...]}
    'BEHAVIOR_SEQUENCE_FEATURES_PROCESS': {
        'user_category': [['cat1', 'click', 'cat1', 'hist_', 'seq_length_']]
    },
    # append_user_data: it's needed in inference mode
    # format: {'user_profile': {profile_name: [(is_append_all_profile, new_col_name, profile_col_name), ...], ...}
    #          'realtime_user_profile': {profile_name: [(is_append_all_profile, new_col_name, profile_col_name), ...], ...}}
    # if is_append_all_profile is "True", it will create a new_col_name to data and append the whole profile to it
    'STATISTICS_COLS': [],
    # chain_configs
    # holds the pipeline of preprocessor for anything you want it to.
    # CHAIN_CONFIGS format: {'func_name': {'func_param_key': 'func_param_value'}}
    'CHAIN_CONFIGS': {'append_timestamp': {'disable_mode': ['train', 'validation']},
                      'convert_columns_name': {},
                      'process_age': {},
                      'handle_missing_data': {},
                      'convert_data_type': {},
                      'process_category': {'disable_mode': ['inference']},
                      'aggregate_preference': {},
                      'encode_features': {},
                      'get_publish_time_to_now': {},
                      'normalize_data': {},
                      'process_user_behavior_sequence': {}}
})

PlanetNovelUser2ItemDINConfig = PlanetNovelUser2ItemGeneralConfig.update(PlanetNovelUser2ItemDINModelConfig) \
                                                               .update(PlanetNovelUser2ItemDINPreprocessConfig)
