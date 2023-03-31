from bdds_recommendation.src_v2.configs import GeneralConfigs
from bdds_recommendation.src_v2.configs.models.sgd import BaseSGDConfigs
from bdds_recommendation.src_v2.configs.preprocess import BaseItem2ItemPreprocessConfigs
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


JollybuyGoodsItem2ItemSGDModelConfig = BaseSGDConfigs.update({
    'PRE_PIPELINE': [ColumnTransformer(transformers=[('MinMax', MinMaxScaler(), ['price', 'click_price'])],
                                       remainder='passthrough')
                     ]
})

JollybuyGoodsItem2ItemGeneralConfig = GeneralConfigs.update({
    'REMOVE_COLS': ['uuid', 'click_uuid', 'y'],
    'REQUISITE_COLS': ['uuid', 'click_uuid', 'view_also_view_score', 'price', 'click_price',
                       'semantics_dot', 'cat0_dot', 'cat1_dot', 'y'],
    'CONTENT_TYPE': 'jollybuy_goods',
    'SERVICE_TYPE': 'item2item',
    'NDCG_GROUPBY_KEY': 'uuid',
    'MONITOR_METRICS_TO_THRESHOLD_MAPPING': {'auc': 10, 'ndcg5': 10, 'ndcg10': 10, 'ndcg20': 10},
    'ALERT_MAIL_RECIPIENTS': 'leoliou@gamania.com'
})

JollybuyGoodsItem2ItemSGDPreprocessConfig = BaseItem2ItemPreprocessConfigs.update({

    # EMBEDDING_PROCESS format: [{'embed_col1': 'title_embedding', 'embed_col2': 'click_title_embedding', 'mode': 'dot'}, {...}]
    'EMBEDDING_PROCESS': [
        {'embed_col1': 'title_embedding', 'embed_col2': 'click_title_embedding', 'dim': 300, 'mode': 'dot'}
    ],

    # format: {mode: [col1, col2]}
    # only support 'ast', 'json', 'int', 'float' mode
    'TYPE_CONVERT_MODE2COLS': {
        'ast': ['cat0', 'cat1', 'price', 'publish_time',
                'click_cat0', 'click_cat1', 'click_price', 'click_publish_time', 'timestamp'],
        'json': ['title_embedding', 'click_title_embedding']
    },

    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; mode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    'COL2CATS_NAMES': 'col2label.pickle',
    'CATEGORY_FEATURES_PROCESS': {
        'cat0': [False, True, 'VectorEncoding'],
        'cat1': [False, True, 'VectorEncoding'],
        'click_cat0': [False, True, 'VectorEncoding'],
        'click_cat1': [False, True, 'VectorEncoding']
    },

    # columns to be aggregated
    # {new_col_name: [col1, col2]}
    'COLS_TO_AGGREGATE': {'cat0_dot': ('cat0', 'click_cat0'), 'cat1_dot': ('cat1', 'click_cat1')},

    # chain_configs
    # holds the pipeline of preprocessor for anything you want it to.
    # CHAIN_CONFIGS format: {'func_name': {'func_param_key': 'func_param_value'}}
    'CHAIN_CONFIGS': {'append_source_data': {'disable_mode': ['train', 'validation']},
                      'append_view_also_view_data': {'disable_mode': ['train', 'validation']},
                      'convert_data_type': {},
                      'process_embedding': {},
                      'encode_features': {'prefix': 'click_'},
                      'aggregate_features': {}}

})

JollybuyGoodsItem2ItemSGDConfig = JollybuyGoodsItem2ItemGeneralConfig.update(JollybuyGoodsItem2ItemSGDPreprocessConfig) \
                                                                     .update(JollybuyGoodsItem2ItemSGDModelConfig)
