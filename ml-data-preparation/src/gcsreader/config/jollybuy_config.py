from .base import BaseConfig


class JollybuyGoodsItem2ItemConfig(BaseConfig):

    EVENT_PATH = 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=*/is_page_view=*/event=*/*.parquet'
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=jollybuy/content_type=jollybuy_goods/snapshot/*.parquet'

    # used by `filter_dataframe`
    TARGET_EVENT = {
        'candidate_pool': ['jb_goods_page_view'],
        'positive_pool': ['jb_goods_content_click']
    }

    TARGET_INFO = {
        # {info: [(original_key, alias_key), (original_key2, alias_key2)]}
        'candidate_pool': {'page_info': [('uuid', 'uuid'), ('name', 'name'), ('page', 'page')]},
        'positive_pool': {'page_info': [('uuid', 'uuid'), ('name', 'name'), ('page', 'page')], 'click_info': [('uuid', 'click_uuid'), ('name', 'click_name'), ('sec', 'sec')]}
    }

    TARGET_COLS = {
        'candidate_pool': ['uuid', 'name', 'date', 'timestamp'],
        'positive_pool': ['uuid', 'name', 'click_uuid', 'click_name', 'date', 'timestamp'],
        'content_pool': ['goods_number', 'cat0', 'cat1', 'title_embedding', 'price', 'publish_time'],
        'all_pool': ['uuid', 'name', 'click_uuid', 'click_name', 'date', 'timestamp', 'y'],
        'view_also_view': ['content_uuid', 'view_also_view_json', 'date'],
        'final_pool': ['uuid', 'name', 'cat0', 'cat1', 'title_embedding', 'price', 'publish_time',
                       'click_uuid', 'click_name', 'click_cat0', 'click_cat1', 'click_title_embedding', 'click_price', 'click_publish_time',
                       'view_also_view_score', 'y', 'timestamp', 'date']
    }

    FILTER_CONDITIONS = {
        # {target_col: [value_1, value_2, ...]}
        'candidate_pool': {'page': ['jb_goods'], 'is_additive_view': ['False']},
        'positive_pool': {'page': ['jb_goods'], 'sec': ['maybe_like']}
    }

    # Need to rename feature cols for positive sample to avoid duplicated naming for candidate and positive sample in df_positive
    # {config_key: {old_col_name1: new_col_name1, old_col_name2: new_col_name2, ...}}
    RENAMED_COLS = {
        'all_pool': {'cat0': 'click_cat0', 'cat1': 'click_cat1', 'title_embedding': 'click_title_embedding', 'publish_time': 'click_publish_time', 'price': 'click_price'},
        'negative_pool': {'name': 'click_name'},
        'view_also_view': {'content_uuid': 'uuid'},
        'content_pool': {'goods_number': 'content_id'}
    }

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {
        'content_pool': ['cat0', 'cat1']
    }


class JollybuyGoodsMultiItem2ItemConfig(BaseConfig):

    EVENT_PATH = 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=*/is_page_view=*/event=*/*.parquet'
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=jollybuy/content_type=jollybuy_goods/snapshot/*.parquet'
    # path for metrics (view also view and buy also buy)
    VIEW_ALSO_VIEW_PATH = 'gs://pipeline-PROJECT_ID/metrics/METRICS/CONTENT_TYPE/INPUT_DATE/METRICS_CONTENT_TYPE.csv'

    # used by `filter_dataframe`
    TARGET_EVENT = {
        'candidate_pool': ['jb_goods_page_view'],
    }

    TARGET_INFO = {
        # {info: [(original_key, alias_key), (original_key2, alias_key2)]}
        'candidate_pool': {'page_info': [('uuid', 'uuid'), ('name', 'name'), ('page', 'page')]}
    }

    TARGET_COLS = {
        'candidate_pool': ['uuid', 'name', 'date', 'timestamp'],
        'content_pool': ['goods_number', 'cat0', 'cat1', 'title_embedding', 'price', 'publish_time', 'title'],
        'all_pool': ['uuid', 'name', 'click_uuid', 'click_name', 'date', 'timestamp', 'y'],
        'view_also_view': ['content_uuid', 'view_also_view_json', 'date'],
        'buy_also_buy': ['content_uuid', 'buy_also_buy_json', 'date'],
        'final_pool': ['uuid', 'name', 'cat0', 'cat1', 'title_embedding', 'price', 'publish_time',
                       'click_uuid', 'click_name', 'click_cat0', 'click_cat1', 'click_title_embedding', 'click_price', 'click_publish_time',
                       'view_also_view_score', 'buy_also_buy_score', 'y', 'timestamp', 'date']
    }

    FILTER_CONDITIONS = {
        # {target_col: [value_1, value_2, ...]}
        'candidate_pool': {'page': ['jb_goods'], 'is_additive_view': ['False']},
    }

    # Need to rename feature cols for positive sample to avoid duplicated naming for candidate and positive sample in df_positive
    # {config_key: {old_col_name1: new_col_name1, old_col_name2: new_col_name2, ...}}
    RENAMED_COLS = {
        'all_pool': {'cat0': 'click_cat0', 'cat1': 'click_cat1', 'title_embedding': 'click_title_embedding',
                     'publish_time': 'click_publish_time', 'price': 'click_price'},
        'negative_pool': {'name': 'click_name'},
        'view_also_view': {'content_uuid': 'uuid'},
        'buy_also_buy': {'content_uuid': 'uuid'},
        'content_pool': {'goods_number': 'content_id'}
    }

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {
        'content_pool': ['cat0', 'cat1']
    }


class JollybuyGoodsHotItemsConfig(BaseConfig):

    EVENT_DAILY_FOLDER = 'event_daily'
    EVENT_HOURLY_FOLDER = 'event_hourly'
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=jollybuy/content_type=jollybuy_goods/snapshot/*.parquet'

    # used by `filter_dataframe`
    TARGET_EVENT = {
        'input_data_pool': ['jb_merchant_content_click', 'jb_goods_content_click', 'jb_category_content_click', 'jb_search_content_click', 'jb_home_page_content_click', 'jb_goods_item_click',
                            'jb_merchant_content_impression', 'jb_goods_content_impression', 'jb_cart_content_impression', 'jb_category_content_impression', 'jb_search_content_impression', 'jb_home_page_content_impression',
                            'jb_goods_page_view'],
    }

    TARGET_COLS = {
        'input_data_pool': ['trackid', 'openid', 'page_info', 'impression_info', 'click_info', 'hour', 'date', 'property', 'event'],
        'content_pool': ['content_id', 'title', 'cat0', 'cat1', 'price', 'count_total_sales', 'count_total_browse', 'count_total_favorite', 'count_comment']
    }

    FILTER_CONDITIONS = {
        'content_pool': {'status': ['1']}  # {target_col: [value_1, value_2, ...]}
    }

    # filter and get counts
    # Format : {event: [target_info, filter_info, content_id_col, filter_key, filter_value_list]} in accordance with `udf_info_array_of_json_map_transaction` and `udf_info_json_map_transaction`
    COUNT_FILTER_CONDITIONS = {'impression': {'jb_merchant_content_impression': ['impression_info', 'page_info', 'uuid', 'page', ['jb_store', 'jb_store_category', 'jb_merchant_search'], True],
                                              'jb_goods_content_impression': ['impression_info', 'impression_info', 'uuid', 'sec', ['recommendation', 'maybe_like'], True],
                                              'jb_cart_content_impression': ['impression_info', 'impression_info', 'uuid', 'sec', ['maybe_like'], True],
                                              'jb_category_content_impression': ['impression_info', 'page_info', 'uuid', 'page', ['jb_category'], True],
                                              'jb_search_content_impression': ['impression_info', 'page_info', 'uuid', 'page', ['jb_search'], True],
                                              'jb_home_page_content_impression': ['impression_info', 'impression_info', 'uuid', 'sec', ['limited_offer', 'maybe_like'], True]},
                               'click': {'jb_merchant_content_click': ['click_info', 'click_info', 'uuid', 'sec', ['tv_wall'], False],
                                         'jb_goods_content_click': ['click_info', 'click_info', 'uuid', 'sec', ['recommendation', 'maybe_like'], True],
                                         'jb_category_content_click': ['click_info', 'page_info', 'uuid', 'page', ['jb_category'], True],
                                         'jb_search_content_click': ['click_info', 'click_info', 'uuid', 'type', ['enterprise', 'personal'], True],
                                         'jb_home_page_content_click': ['click_info', 'click_info', 'uuid', 'sec', ['limited_offer', 'maybe_like', 'trending'], True]},
                               'booster': {'jb_goods_item_click': ['page_info', 'click_info', 'uuid', 'type', ['action_btn', 'share_btn', 'review_tab'], True]},
                               'reward': {'jb_goods_page_view': ['page_info', 'page_info', 'uuid', 'page', ['jb_goods'], True],
                                          'jb_goods_item_click': ['page_info', 'click_info', 'uuid', 'type', ['trade_btn'], True]}}

    # Need to rename feature cols for positive sample to avoid duplicated naming for candidate and positive sample in df_positive
    # {config_key: {old_col_name1: new_col_name1, old_col_name2: new_col_name2, ...}}
    RENAMED_COLS = {
        'content_pool': {'content_id': 'uuid'}
    }

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {
        'content_pool': ['cat0', 'cat1']
    }
