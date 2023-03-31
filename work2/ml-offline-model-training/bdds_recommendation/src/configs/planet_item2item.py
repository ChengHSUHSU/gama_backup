from bdds_recommendation.src.configs import BaseItem2itemConfig


class PlanetNewsItem2ItemConfig(BaseItem2itemConfig):

    CONTENT_TYPE = 'planet_news'

    # only support `ast` and `json` for now
    # {mode: [col1, col2, ...]}
    TYPE_CONVERT_MODE2COLS = {'ast': ['cat0', 'cat1', 'cat2', 'content_ner', 'publish_time', 'click_cat0',
                                      'click_cat1', 'click_cat2', 'click_content_ner', 'click_publish_time', 'timestamp'],
                              'json': ['title_embedding', 'click_title_embedding']}

    # {original_col: new_col}
    CONTENT_NER_COLS_MAP = {'content_ner': 'content_ner_list', 'click_content_ner': 'click_content_ner_list'}

    # categorical columns
    COLUMNS_TO_ENCODE = ['cat0', 'cat1', 'cat2', 'content_ner_list', 'site_name', 'click_cat0', 'click_cat1', 'click_cat2', 'click_content_ner_list', 'click_site_name']

    # columns to be aggregated
    # {new_col_name: [col1, col2]}
    COLS_TO_AGGREGATE = {'cat0_dot': ('cat0', 'click_cat0'), 'cat1_dot': ('cat1', 'click_cat1'), 'cat2_dot': ('cat2', 'click_cat2'),
                         'ner_dot': ('content_ner_list', 'click_content_ner_list'), 'site_name_dot': ('site_name', 'click_site_name')}

    # column pair to generate ner matching score
    # {new_col_name: [col1, col2]}
    COLS_OF_NER_MATCHING = {'ner_score': ('content_ner', 'click_content_ner')}

    # final columns to be used for training
    REQUISITE_COLS = ['view_also_view_score', 'semantics_dot', 'cat0_dot', 'cat1_dot', 'cat2_dot', 'ner_dot', 'site_name_dot', 'ner_score', 'y']

    # all categories list file
    COL2CATS_NAMES = 'col2label.pickle'

    MONITOR_METRICS_TO_THRESHOLD_MAPPING = {'auc': 10}
    ALERT_MAIL_RECIPIENTS = 'xiangchen@gamania.com'
