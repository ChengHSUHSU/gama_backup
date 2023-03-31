from bdds_recommendation.src_v2.configs import GeneralConfigs
from bdds_recommendation.src_v2.configs.models.sgd import BaseSGDConfigs
from bdds_recommendation.src_v2.configs.preprocess import BaseItem2ItemPreprocessConfigs


PlanetNewsItem2ItemGeneralConfig = GeneralConfigs.update({
    'REMOVE_COLS': ['uuid', 'click_uuid', 'y'],
    'REQUISITE_COLS': ['uuid', 'click_uuid', 'view_also_view_score', 'semantics_dot', 'cat0_dot', 'cat1_dot', 'cat2_dot', 'ner_dot', 'site_name_dot', 'ner_score', 'y'],
    'CONTENT_TYPE': 'planet_news',
    'SERVICE_TYPE': 'item2item',
    'NDCG_GROUPBY_KEY': 'uuid',
    'MONITOR_METRICS_TO_THRESHOLD_MAPPING': {'auc': 10, 'ndcg5': 10, 'ndcg10': 10, 'ndcg20': 10},
    'ALERT_MAIL_RECIPIENTS': 'xiangchen@gamania.com'
})

PlanetNewsItem2ItemSGDPreprocessConfig = BaseItem2ItemPreprocessConfigs.update({

    # EMBEDDING_PROCESS format: [{'embed_col1': 'title_embedding', 'embed_col2': 'click_title_embedding', 'mode': 'dot'}, {...}]
    'EMBEDDING_PROCESS': [
        {'embed_col1': 'title_embedding', 'embed_col2': 'click_title_embedding', 'dim': 300, 'mode': 'prod'},
        {'embed_col1': 'title_embedding', 'embed_col2': 'click_title_embedding', 'dim': 300, 'mode': 'dot'}
    ],

    # format: {mode: [col1, col2]}
    # only support 'ast', 'json', 'int', 'flaot' mode
    'TYPE_CONVERT_MODE2COLS': {
        'ast': ['cat0', 'cat1', 'cat2', 'content_ner', 'publish_time',
                'click_cat0', 'click_cat1', 'click_cat2', 'click_content_ner', 'click_publish_time', 'timestamp'],
        'json': ['title_embedding', 'click_title_embedding']
    },
    'TYPE_CONVERT_MODE2COLS_INFERENCE': {
        'ast': ['content_ner', 'click_content_ner']
    },

    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; moode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    'COL2CATS_NAMES': 'col2label.pickle',
    'CATEGORY_FEATURES_PROCESS': {
        'cat0': [False, True, 'VectorEncoding'],
        'cat1': [False, True, 'VectorEncoding'],
        'cat2': [False, True, 'VectorEncoding'],
        'content_ner_list': [False, True, 'VectorEncoding'],
        'site_name': [False, True, 'VectorEncoding'],
        'click_cat0': [False, True, 'VectorEncoding'],
        'click_cat1': [False, True, 'VectorEncoding'],
        'click_cat2': [False, True, 'VectorEncoding'],
        'click_content_ner_list': [False, True, 'VectorEncoding'],
        'click_site_name': [False, True, 'VectorEncoding'],
    },

    # {original_col: new_col}
    'CONTENT_NER_COLS_MAP': {'content_ner': 'content_ner_list', 'click_content_ner': 'click_content_ner_list'},

    # columns to be aggregated
    # {new_col_name: [col1, col2]}
    'COLS_TO_AGGREGATE': {'cat0_dot': ('cat0', 'click_cat0'), 'cat1_dot': ('cat1', 'click_cat1'), 'cat2_dot': ('cat2', 'click_cat2'),
                          'ner_dot': ('content_ner_list', 'click_content_ner_list'), 'site_name_dot': ('site_name', 'click_site_name')},

    # column pair to generate ner matching score
    # {new_col_name: [col1, col2]}
    'COLS_OF_NER_MATCHING': {'ner_score': ('content_ner', 'click_content_ner')},

    # append_candidate_data_process: it's needed in inference mode
    # format: ['process_col_name_1', 'process_col_name_2', ...]
    'APPEND_CANDIDATE_DATA_PROCESS': [],

    # chain_configs
    # holds the pipeline of preprocessor for anything you want it to.
    # CHAIN_CONFIGS format: {'func_name': {'func_param_key': 'func_param_value'}}
    'CHAIN_CONFIGS': {'append_source_data': {'disable_mode': ['train', 'validation']},
                      'append_view_also_view_data': {'disable_mode': ['train', 'validation']},
                      'convert_data_type': {},
                      'process_embedding': {},
                      'parse_ner': {},
                      'encode_features': {'prefix': 'click_'},
                      'cal_ner_matching': {},
                      'aggregate_features': {}}

})

PlanetNewsItem2ItemSGDConfig = PlanetNewsItem2ItemGeneralConfig.update(PlanetNewsItem2ItemSGDPreprocessConfig)
