from . import BaseUser2itemConfig
from src.preprocess.utils.encoder import UNKNOWN_LABEL


class ClubPostUser2ItemConfig(BaseUser2itemConfig):

    CONTENT_TYPE = 'club_post'

    ZERO = 0.0
    SEMANTICS_EMBEDDING_SIZE = 128      # method one (300), method two (128)
    FEATURE_EMBEDDING_SIZE = {'age': 8, 'gender': 2, 'planet_user_level': 5, 'club_user_level': 5}

    DENSE_FEATURE = ['category_pref_score', 'hour_to_current', 'token_count', 'interaction_score', 'popularity_score']
    ONE_HOT_FEATURE = ['age', 'gender', 'planet_user_level', 'club_user_level']

    SEMANTICS_INPUT = ['user_post_embedding', 'item_title_embedding']   # method two : raw semantics input
    SEMANTICS_FEATURE = {'post': ('user_post_embedding', 'item_title_embedding')}
    SEMANTICS_INTERACTION_MODE = ['prod']

    COLUMN_TO_RENAME = {'final_score': 'popularity_score'}
    COLUMN_NEED_PARSE = []
    COLUMN_TO_FILLNA = {
        'planet_news_used': ZERO,
        'planet_comic_used': ZERO,
        'gamapay_used':  ZERO,
        'shopping_used':  ZERO,
        'game_used':  ZERO,
        'gash_used': ZERO,
        'token_count': ZERO,
        'interaction_score': ZERO,
        'popularity_score': ZERO,
        'planet_user_level': UNKNOWN_LABEL,
        'club_user_level': UNKNOWN_LABEL,
        'gender': UNKNOWN_LABEL,
        'user_category': '{}'
    }

    TYPE_CONVERT_MODE2COLS = {
        'ast': ['cat0'],
        'json': ['user_post_embedding', 'item_title_embedding'],
        'int64': ['publish_time', 'timestamp'],
        'float64': ['token_count', 'interaction_score', 'final_score', 'planet_news_used', 'planet_comic_used', 'gamapay_used', 'shopping_used', 'game_used', 'gash_used']
    }

    NORMALIZE_COLS = {'min-max': {'hour_to_current': 14 * 24}, 'z-score': {}}

    REQUISITE_COLS = ['user_post_embedding', 'gender', 'age', 'planet_user_level', 'club_user_level',
                      'cat0', 'item_title_embedding', 'token_count', 'interaction_score', 'popularity_score',
                      'category_pref_score', 'hour_to_current', 'y']

    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; moode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    COL2CATS_NAMES = 'col2label.pickle'

    CATEGORY_FEATURES_PROCESS = {
        'age': [False, True, 'LabelEncoding'],
        'gender': [False, True, 'LabelEncoding'],
        'planet_user_level': [False, True, 'LabelEncoding'],
        'club_user_level': [False, True, 'LabelEncoding']
        }

    NDCG_GROUPBY_KEY = 'userid'

    MONITOR_METRICS_TO_THRESHOLD_MAPPING = {'auc': 10, 'ndcg5': 10, 'ndcg10': 10, 'ndcg20': 10}
    ALERT_MAIL_RECIPIENTS = 'xiangchen@gamania.com'
