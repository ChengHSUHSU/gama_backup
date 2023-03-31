from src_v2.configs import Config

BaseUser2ItemPreprocessConfigs = Config({

    # aggregate_preference
    # CATEGORY_PREF_SCORE_PROCESS format: [{'level': ['click', 'cat0'], 'cat_col': 'user_category', 'score_col': 'category_pref_score}, {...}]
    # TAG_PREF_SCORE_PROCESS format: [{'tag_entity_list': ['other', 'person'], 'user_tag_col': 'user_tag, 'item_tag_col': 'tags', 'tagging_type': 'editor', 'score_col': ''}, {...}]
    # EMBEDDING_PROCESS format: [{'embed_col1': 'title_embedding', 'embed_col2': 'click_title_embedding', 'mode': 'dot'}, {...}]
    'CATEGORY_PREF_SCORE_PROCESS': [],
    'TAG_PREF_SCORE_PROCESS': [],
    'EMBEDDING_PROCESS': [],

    # convert_data_type
    # TYPE_CONVERT_MODE2COLS format: {mode: [col1, col2]}
    # only support 'ast', 'json', 'int', 'float' mode
    'TYPE_CONVERT_MODE2COLS': {},
    'TYPE_CONVERT_MODE2COLS_INFERENCE': {},

    # encode_features
    # CATEGORY_FEATURES_PROCESS format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; mode: string ('LabelEncoding' or 'VectorEncoding' or 'NumericalOneHotEncoding')
    # all categories list file
    'COL2CATS_NAMES': 'col2label.pickle',
    'CATEGORY_FEATURES_PROCESS': {},

    # handle_missing_data
    # format: {old_name: new_name}
    'COLUMN_TO_RENAME': {},

    # format: {col_name: value_to_be_filled}
    'COLUMN_TO_FILLNA': {},

    # normalize_data
    # support min-max or z-score
    # NORMALIZE_COLS format: {'min-max': {'hour_to_current': 14 * 24}, 'z-score': {}}
    'NORMALIZE_COLS': {
        'min-max': {},  # set val for upper bound normalize, set val None to standard min-max normalize
        'z-score': {}
    },

    # process_user_behavior_sequence
    # process_user_behavior_sequence: (only support profile format of {'click': {'cat1': {}}}, ex: cat/tag profile)
    # format: {col_name: [[enc_name, event_key, cat_level, hist_suffix, seq_len_suffix], ...]}
    'BEHAVIOR_SEQUENCE_FEATURES_PROCESS': {},

    # process_time_feature
    # PROCESS_TIME_FEATURE format:
    # [{'end_time_col': 'timestamp', 'start_time_col': 'publish_time', 'time_type': ['hour'], 'postfix': '_time_period}, ...]
    'PROCESS_TIME_FEATURE': {},

    # format: {'user_profile': {profile_name: [(is_append_all_profile, new_col_name, profile_col_name), ...], ...}
    #          'realtime_user_profile': {profile_name: [(is_append_all_profile, new_col_name, profile_col_name), ...], ...}}
    # if is_append_all_profile is "True", it will create a new_col_name to data and append the whole profile to it
    'APPEND_USER_DATA_PROCESS': {'user_profile': {}, 'realtime_user_profile': {}},

    # chain_configs
    # holds the pipeline of preprocessor for anything you want it to.
    # CHAIN_CONFIGS format: {'func_name': {'func_param_key': 'func_param_value'}}
    'CHAIN_CONFIGS': {'append_user_data': {'disable_mode': ['train', 'validation']},
                      'convert_data_type': {},
                      'convert_columns_name': {},
                      'handle_missing_data': {},
                      'aggregate_preference': {},
                      'encode_features': {},
                      'process_user_behavior_sequence': {},
                      'process_time_feature': {},
                      'normalize_data': {}}
})


BaseItem2ItemPreprocessConfigs = Config({

    # aggregate_preference
    # CATEGORY_PREF_SCORE_PROCESS format: [{'level': ['click', 'cat0'], 'cat_col': 'user_category', 'score_col': 'category_pref_score}, {...}]
    # TAG_PREF_SCORE_PROCESS format: [{'tag_entity_list': ['other', 'person'], 'user_tag_col': 'user_tag, 'item_tag_col': 'tags', 'tagging_type': 'editor', 'score_col': ''}, {...}]
    # EMBEDDING_PROCESS format: [{'embed_col1': 'title_embedding', 'embed_col2': 'click_title_embedding', 'mode': 'dot'}, {...}]
    'CATEGORY_PREF_SCORE_PROCESS': [],
    'TAG_PREF_SCORE_PROCESS': [],
    'EMBEDDING_PROCESS': [],

    # convert_data_type
    # TYPE_CONVERT_MODE2COLS format: {mode: [col1, col2]}
    # only support 'ast', 'json', 'int', 'float' mode
    'TYPE_CONVERT_MODE2COLS': {},
    'TYPE_CONVERT_MODE2COLS_INFERENCE': {},

    # encode_features
    # CATEGORY_FEATURES_PROCESS format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; mode: string ('LabelEncoding' or 'VectorEncoding' or 'NumericalOneHotEncoding')
    # all categories list file
    'COL2CATS_NAMES': 'col2label.pickle',
    'CATEGORY_FEATURES_PROCESS': {},

    # handle_missing_data
    # format: {old_name: new_name}
    'COLUMN_TO_RENAME': {},

    # format: {col_name: value_to_be_filled}
    'COLUMN_TO_FILLNA': {},

    # normalize_data
    # support min-max or z-score
    # NORMALIZE_COLS format: {'min-max': {'hour_to_current': 14 * 24}, 'z-score': {}}
    'NORMALIZE_COLS': {
        'min-max': {},  # set val for upper bound normalize, set val None to standard min-max normalize
        'z-score': {}
    },

    # process_time_feature
    # PROCESS_TIME_FEATURE format:
    # [{'end_time_col': 'timestamp', 'start_time_col': 'publish_time', 'time_type': ['hour'], 'postfix': '_time_period}, ...]
    'PROCESS_TIME_FEATURE': {},

    # chain_configs
    # holds the pipeline of preprocessor for anything you want it to.
    # CHAIN_CONFIGS format: {'func_name': {'func_param_key': 'func_param_value'}}
    'CHAIN_CONFIGS': {'convert_data_type': {},
                      'convert_columns_name': {},
                      'handle_missing_data': {},
                      'encode_features': {},
                      'normalize_data': {}}
})

BaseHotPreprocessConfigs = Config({
    # convert_data_type
    # TYPE_CONVERT_MODE2COLS format: {mode: [col1, col2]}
    # only support 'ast', 'json', 'int', 'float' mode
    'TYPE_CONVERT_MODE2COLS': {},

    # encode_features
    # CATEGORY_FEATURES_PROCESS format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; mode: string ('LabelEncoding' or 'VectorEncoding' or 'NumericalOneHotEncoding')
    # all categories list file
    'CATEGORY_FEATURES_PROCESS': {},

    # handle_missing_data
    # format: {old_name: new_name}
    'COLUMN_TO_RENAME': {},

    # format: {col_name: value_to_be_filled}
    'COLUMN_TO_FILLNA': {},

    # normalize_data
    # support min-max or z-score
    # NORMALIZE_COLS format: {'min-max': {'hour_to_current': 14 * 24}, 'z-score': {}}
    'NORMALIZE_COLS': {
        'min-max': {},  # set val for upper bound normalize, set val None to standard min-max normalize
        'z-score': {}
    },

    # chain_configs
    # holds the pipeline of preprocessor for anything you want it to.
    # CHAIN_CONFIGS format: {'func_name': {'func_param_key': 'func_param_value'}}
    'CHAIN_CONFIGS': {'convert_data_type': {},
                      'convert_columns_name': {},
                      'handle_missing_data': {},
                      'encode_features': {},
                      'normalize_data': {}}
})
