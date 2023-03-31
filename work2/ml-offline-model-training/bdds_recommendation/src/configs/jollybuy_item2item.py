from bdds_recommendation.src.configs import BaseItem2itemConfig
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


class JollybuyGoodsItem2ItemConfig(BaseItem2itemConfig):

    CONTENT_TYPE = 'jollybuy_goods'

    # only support `ast` and `json` for now
    # {mode: [col1, col2, ...]}
    TYPE_CONVERT_MODE2COLS = {'ast': ['cat0', 'cat1', 'price', 'publish_time',
                                      'click_cat0', 'click_cat1', 'click_price', 'click_publish_time', 'timestamp'],
                              'json': ['title_embedding', 'click_title_embedding']}

    # categorical columns
    COLUMNS_TO_ENCODE = ['cat0', 'cat1', 'click_cat0', 'click_cat1']

    # columns to be aggregated
    # {new_col_name: [col1, col2]}
    COLS_TO_AGGREGATE = {'cat0_dot': ('cat0', 'click_cat0'), 'cat1_dot': ('cat1', 'click_cat1')}

    # final columns to be used for training
    REQUISITE_COLS = ['view_also_view_score', 'price', 'click_price',
                      'semantics_dot', 'cat0_dot', 'cat1_dot', 'y']

    # all categories list file
    COL2CATS_NAMES = 'col2label.pickle'

    # Works must run before model training
    PRE_PIPELINE = [ColumnTransformer(transformers=[('MinMax', MinMaxScaler(), ['price', 'click_price'])],
                                      remainder='passthrough')]

    MONITOR_METRICS_TO_THRESHOLD_MAPPING = {'auc': 10, 'ndcg5': 10, 'ndcg10': 10, 'ndcg20': 10}
    ALERT_MAIL_RECIPIENTS = 'nickchen@gamania.com,leoliou@gamania.com'


class JollybuyGoodsMultiItem2ItemConfig(JollybuyGoodsItem2ItemConfig):

    # final columns to be used for training
    REQUISITE_COLS = ['view_also_view_score', 'buy_also_buy_score', 'price', 'click_price',
                      'semantics_dot', 'cat0_dot', 'cat1_dot', 'y']
