from bdds_recommendation.src.configs import BaseItem2itemConfig


class NownewsNewsItem2ItemConfig(BaseItem2itemConfig):

    MAJOR_COL_PREFIX = 'page'
    MINOR_COL_PREFIX = 'click'

    # feature can not be none in training pipeline
    REQUISITE_INPUT_FEATURE = ['page_cat0']

    # only support `ast` and `json` for now
    # {mode: [col1, col2, ...]}
    TYPE_CONVERT_MODE2COLS = {
        'ast': ['page_tags', 'page_cat0', 'page_cat1', 'click_tags', 'click_cat0', 'click_cat1'],
        'json': ['page_item_title_embedding', 'click_item_title_embedding']
    }

    # only support `prod`, `dot` and `sum` for now
    SEMANTICS_INTERACTION_MODE = ['prod', 'dot']

    CONTENT_NER_LABEL = ['person', 'location', 'event', 'organization', 'item', 'others']
    CONTENT_NER_PAIRS = ['page_content_ner', 'click_content_ner']

    # columns to be aggregated
    # {new_col_name: [col1, col2]}
    COLS_TO_AGGREGATE = {
        'cat0_dot': ('page_cat0', 'click_cat0'),
        'cat1_dot': ('page_cat1', 'click_cat1'),
        'tags_dot': ('page_tags', 'click_tags'),
    }

    # categorical columns
    COLUMNS_TO_ENCODE = ['cat0', 'cat1', 'tags']

    # view also view columns name
    VIEW_ALSO_VIEW_COL = {
        'input': 'page_view_also_view_json',
        'output': 'view_also_view_score',
        'pair': 'click_uuid'}

    # final columns to be used for training
    REQUISITE_COLS = [
        'cat0_dot', 'cat1_dot', 'tags_dot', 'semantics_dot',
        'ner_person_dot', 'ner_location_dot', 'ner_event_dot',
        'ner_organization_dot', 'ner_item_dot', 'ner_others_dot',
        'view_also_view_score', 'y'
    ]


    # process categorical: format: {col_name: [enable_padding, enable_unknown, mode]}
    # enable_padding: bool ; enable_unknown: bool ; moode: string ('LabelEncoding' or 'VectorEncoding')
    # all categories list file
    COL2CATS_NAMES = 'col2label.pickle'
