from bdds_recommendation.src_v2.configs import GeneralConfigs
from bdds_recommendation.src_v2.configs.models.din import BaseDINConfigs
from bdds_recommendation.src_v2.configs.preprocess import BaseUser2ItemPreprocessConfigs
from bdds_recommendation.src.preprocess.utils.encoder import UNKNOWN_LABEL, PAD_LABEL

# Jollybuy goods user2item
JollybuyGoodsUser2ItemGeneralConfig = GeneralConfigs.update({
    'REQUISITE_COLS': ['age', 'gender', 'cat0', 'cat1', 'final_score',
                       'user_title_embedding', 'item_title_embedding',
                       'cat0_pref_score', 'cat1_pref_score', 'tag_other_pref_score',
                       'hist_cat0', 'seq_length_cat0', 'hist_cat1', 'seq_length_cat1',
                       'y'],
    'CONTENT_TYPE': 'jollybuy_goods',
    'SERVICE_TYPE': 'user2item',
    'NDCG_GROUPBY_KEY': 'userid',
    'MONITOR_METRICS_TO_THRESHOLD_MAPPING': {'auc': 10, 'ndcg5': 10, 'ndcg10': 10, 'ndcg20': 10},
    'ALERT_MAIL_RECIPIENTS': 'leoliou@gamania.com'

})

JollybuyGoodsUser2ItemDINModelConfig = BaseDINConfigs.update({
    'DENSE_FEATURE_SIZE': {'final_score': 1, 'cat0_pref_score': 1,
                           'cat1_pref_score': 1, 'tag_other_pref_score': 1,
                           'user_title_embedding': 300, 'item_title_embedding': 300},
    'DENSE_FEATURE': ['final_score', 'cat0_pref_score', 'cat1_pref_score', 'tag_other_pref_score',
                      'user_title_embedding', 'item_title_embedding'],

    'FEATURE_EMBEDDING_SIZE': {'age': 8, 'gender': 2, 'cat0': 17, 'cat1': 100},
    'ONE_HOT_FEATURE': ['age', 'gender', 'cat0', 'cat1'],

    'BEHAVIOR_FEATURE': ['cat0', 'cat1'],
    'BEHAVIOR_FEATURE_SEQ_LENGTH': ['seq_length_cat0', 'seq_length_cat1'],
    'BEHAVIOR_SEQUENCE_SIZE': 10
})

JollybuyGoodsUser2ItemDINPreprocessConfig = BaseUser2ItemPreprocessConfigs.update({

    # format: {mode: [col1, col2]}
    # only support 'ast', 'json', 'int', 'float' mode
    'TYPE_CONVERT_MODE2COLS': {
        'ast': ['cat0', 'cat1'],
        'json': ['user_title_embedding', 'item_title_embedding'],
        'float': ['final_score']
    },

    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; mode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    'COL2CATS_NAMES': 'col2label.pickle',
    'CATEGORY_FEATURES_PROCESS': {
        'age': [False, True, 'LabelEncoding'],
        'gender': [False, True, 'LabelEncoding'],
        'tags': [False, True, 'LabelEncoding'],
        'cat0': [True, True, 'LabelEncoding'],
        'cat1': [True, True, 'LabelEncoding']
    },

    # format: {old_name: new_name}
    'COLUMN_TO_RENAME': {'title_embedding': 'item_title_embedding'},

    # format: {col_name: value_to_be_filled}
    'COLUMN_TO_FILLNA': {
        'gender': UNKNOWN_LABEL,
        'cat0': [PAD_LABEL],
        'cat1': [PAD_LABEL],
        'final_score': 0.0,
        'user_title_embedding': [0.0] * 300,
        'item_title_embedding': [0.0] * 300,
        'user_category': '{}'
    },

    # support min-max or z-score (Need to be optimized. Currently normalize method is too hacky)
    'NORMALIZE_COLS': {
        'min-max': {},  # set val for upper bound normalize, set val None to standard min-max normalize
        'z-score': {}
    },

    # aggregate preference
    'CATEGORY_PREF_SCORE_PROCESS': [{'level': ['click', 'cat0'], 'cat_col': 'user_category', 'score_col': 'cat0_pref_score'},
                                    {'level': ['click', 'cat1'], 'cat_col': 'user_category', 'score_col': 'cat1_pref_score'}],
    'TAG_PREF_SCORE_PROCESS': [{'tag_entity_list': ['others'], 'user_tag_col': 'user_tag', 'item_tag_col': 'tags', 'tagging_type': 'editor', 'score_col': 'tag_other_pref_score'}],

    # process_user_behavior_sequence: (only support profile format of {'click': {'cat1': {}}}, ex: cat/tag profile)
    # format: {col_name: [[enc_name, event_key, cat_level, hist_suffix, seq_len_suffix], ...]}
    'BEHAVIOR_SEQUENCE_FEATURES_PROCESS': {
        'user_category': [['cat0', 'click', 'cat0', 'hist_', 'seq_length_'],
                          ['cat1', 'click', 'cat1', 'hist_', 'seq_length_']]
    },

    # format: {'user_profile': {profile_name: [(is_append_all_profile, new_col_name, profile_col_name), ...], ...}
    #          'realtime_user_profile': {profile_name: [(is_append_all_profile, new_col_name, profile_col_name), ...], ...}}
    # if is_append_all_profile is "True", it will create a new_col_name to data and append the whole profile to it
    'APPEND_USER_DATA_PROCESS': {'user_profile': {'userMetaProfile': [(False, 'age', 'age'), (False, 'gender', 'gender')],
                                                  'jollybuyGoodsCategoryProfile': [(True, 'user_category', 'user_category')],
                                                  'jollybuyGoodsEmbeddingProfile': [(False, 'user_title_embedding', 'title')],
                                                  'jollybuyGoodsTagProfile': [(True, 'user_tag', 'user_tag')]},
                                 'realtime_user_profile': {}},

    # chain_configs
    # holds the pipeline of preprocessor for anything you want it to.
    # CHAIN_CONFIGS format: {'func_name': {'func_param_key': 'func_param_value'}}
    'CHAIN_CONFIGS': {'append_user_data': {'disable_mode': ['train', 'validation']},
                      'append_metrics_data': {'disable_mode': ['train', 'validation']},
                      'convert_data_type': {},
                      'convert_columns_name': {},
                      'process_age': {},
                      'handle_missing_data': {},
                      'aggregate_preference': {},
                      'encode_features': {},
                      'process_user_behavior_sequence': {}}
})

JollybuyGoodsUser2ItemDINConfig = JollybuyGoodsUser2ItemGeneralConfig.update(JollybuyGoodsUser2ItemDINModelConfig) \
                                                                     .update(JollybuyGoodsUser2ItemDINPreprocessConfig)
