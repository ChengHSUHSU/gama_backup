from .base import BaseConfig


class PlanetNewsUser2ItemConfig(BaseConfig):
    CONTENT_COL = ['content_id', 'title', 'publish_time', 'author', 'site_name', 'tags', 'cat0', 'cat1', 'cat2', 'status', 'title_embedding', 'content_ner']
    EVENT_NAME = 'planet_content_page_view'
    EVENT_TO_CONDITION_MAPPING = {'page_info.page': 'planet_news'}
    USER_PROFILE = {
        'tag': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/tag/INPUT_DATE/CONTENT_PATH/*.parquet',
        'category': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/category/INPUT_DATE/CONTENT_PATH/*.parquet',
        'title_embedding': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/embedding/INPUT_DATE/CONTENT_PATH/*.parquet',
        'meta': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/gamania/meta/meta/INPUT_DATE/*.parquet'
    }
    COLUMN_TO_RENAME = {
        'item_content': [('title_embedding', 'item_title_embedding')],
        'user_title_embedding': [('userid', 'openid'), ('data', 'user_title_embedding')],
        'user_category': [('userid', 'openid'), ('data', 'user_category')],
        'user_tag': [('userid', 'openid'), ('data', 'user_tag')],
        'user_meta': [('userid', 'openid')]
    }

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {
        'content': ['cat0', 'cat1', 'cat2', 'site_name']
    }


class PlanetNewsUser2ItemOptimizedConfig(BaseConfig):

    # event config
    EVENT_PATH = 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=*/is_page_view=*/event=*/*.parquet'
    EVENT_OF_CONTENT_TYPE_CONDITIONS = {'planet_news': [{'event': 'planet_content_page_view', 'page_info_map["page"]': 'planet_news'}],
                                        'nownews_news': [{'event': 'nn_news_page_view', 'page_info_map["page"]': 'news'}]}
    # content config
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=*/content_type=CONTENT_TYPE/snapshot/*.parquet'

    # metrics config
    METRICS_POPULARITY_PATH = 'gs://pipeline-PROJECT_ID/metrics/popularity/POPULARITY_FOLDER/INPUT_DATE/INPUT_HOUR/*.csv'
    POPULARITY_FOLDER = {'planet_news': 'news',
                         'nownews_news': 'nownews_news'}

    # user profile config
    PLAENT_PROFILE_PATH = 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/PROFILE_TYPE/INPUT_DATE/BLOB_PATH/*.parquet'
    NOWNEWS_PROFILE_PATH = 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/nownews/PROFILE_TYPE/INPUT_DATE/*.parquet'
    GAMANIA_META_PROFILE_PATH = 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/gamania/meta/meta/INPUT_DATE/*.parquet',

    USER_PROFILE_CONDITIONS = {'planet_news': {'user_profile_path': PLAENT_PROFILE_PATH, 'blob_path': 'news', 'save_type': 'diff'},
                               'nownews_news': {'user_profile_path': NOWNEWS_PROFILE_PATH, 'blob_path': '', 'save_type': 'diff'},
                               'gamania_meta': {'user_profile_path': GAMANIA_META_PROFILE_PATH, 'blob_path': '', 'save_type': 'snapshot'}}

    STEPS_TO_RENAMCOLS = {'event': [('uuid', 'content_id')],
                          'content': [('title_embedding', 'item_title_embedding')],
                          'metrics': {
                              'popularity': [('final_score', 'popularity_score'), ('uuid', 'content_id')],
                              'snapshot_popularity': [('final_score', 'popularity_score'), ('uuid', 'content_id')]},
                          'user_profile': {
                              'category': [('data', 'user_category')],
                              'tag': [('data', 'user_tag')],
                              'title_embedding': [('data', 'user_title_embedding')]}}

    REQUISITE_COLS = {'event': ['userid', 'event', 'content_id', 'date', 'hour', 'timestamp'],
                      'content': ['content_id', 'title', 'publish_time', 'cat0', 'cat1', 'tags', 'content_ner', 'item_title_embedding'],
                      'metrics': {'popularity': ['content_id', 'total_click_count', 'total_view_count', 'popularity_score', 'date', 'hour'],
                                  'snapshot_popularity': ['content_id', 'total_click_count', 'total_view_count', 'popularity_score']},
                      'user_profile': {'category': ['userid', 'date', 'user_category'],
                                       'tag': ['userid', 'date', 'user_tag'],
                                       'title_embedding': ['userid', 'date', 'user_title_embedding']}}

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {'content': ['cat0', 'cat1', 'tags']}


class PlanetComicsBookUser2ItemConfig(BaseConfig):
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=beanfun/content_type=planet_comics_book/snapshot/*.parquet'
    EVENT_PATH = 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=beanfun/is_page_view=*/event=*/*.parquet'
    METRICS_POPULARITY_PATH = 'gs://pipeline-PROJECT_ID/metrics/popularity/POPULARITY_FOLDER/INPUT_DATE/INPUT_HOUR/*.csv'
    PLAENT_PROFILE_PATH = 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/PROFILE_TYPE/INPUT_DATE/BLOB_PATH/*.parquet'
    GAMANIA_META_PROFILE_PATH = 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/gamania/meta/meta/INPUT_DATE/*.parquet'

    USER_PROFILE_CONDITIONS = {'planet_comics_book': {'user_profile_path': PLAENT_PROFILE_PATH, 'blob_path': 'comic', 'save_type': 'diff'},
                               'gamania_meta': {'user_profile_path': GAMANIA_META_PROFILE_PATH, 'blob_path': '', 'save_type': 'snapshot'}}

    EVENT_OF_CONTENT_TYPE_CONDITIONS = {'planet_comics_book': [{'event': 'planet_content_page_view', 'page_info_map["page"]': 'planet_comics'}]}

    POPULARITY_FOLDER = {'planet_comics_book': 'comic'}
    CONTENT_TYPE_TO_PROPERTY_TYPE = {'planet_comics_book': 'beanfun'}

    STEPS_TO_RENAMCOLS = {
        'event': [],
        'content': [('title_embedding', 'item_title_embedding'), ('content_id', 'uuid')],
        'metrics': {'popularity': [('content_id', 'uuid')]},
        'user_profile': {'title_embedding': [('data', 'user_title_embedding')],
                         'category': [('data', 'user_category')],
                         'tag': [('data', 'user_tag')]}
    }

    PROCESS_PROFILE_TYPES = ['title_embedding', 'category', 'tag', 'meta']
    PROCESS_METRICS_TYPES = ['popularity']

    REQUISITE_COLS = {'event': ['userid', 'uuid', 'timestamp', 'date', 'hour'],
                      'content': [
                                    'uuid', 'title', 'item_title_embedding', 'cat0', 'cat1', 'cat2', 'tags', 'publish_time', 'content_ner',
                                    'is_adult', 'read_count_repeat', 'read_count_norepeat', 'like_count'
                                ],
                      'metrics': {'popularity': ['uuid', 'final_score', 'date', 'hour']},
                      'user_profile': {'title_embedding': ['userid', 'date', 'user_title_embedding'],
                                       'category': ['userid', 'date', 'user_category'],
                                       'tag': ['userid', 'date', 'user_tag'],
                                       'meta': ['userid', 'gender', 'age']}}

    BASE_PREF_USER_PROFILE_CONDITION = ('pref', 0.01)
    USER_PROFILE_LIST = ['user_category', 'user_tag', 'user_embedding']

    META_COLS = [('gender', 'data', '$.gender'), ('age', 'data', '$.age')]

    FINAL_COLS = [
        'userid', 'uuid', 'timestamp', 'date', 'age', 'gender',
        'user_category', 'user_tag', 'user_title_embedding', 'title',
        'is_adult', 'read_count_repeat', 'read_count_norepeat', 'like_count',
        'publish_time', 'cat0', 'cat1', 'cat2', 'tags', 'content_ner', 'item_title_embedding', 'y'
    ]

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {
        'content': ['cat0', 'cat1', 'cat2', 'tags']
    }


class PlanetNovelBookUser2ItemConfig(BaseConfig):
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=PROPERTY/content_type=CONTENT_TYPE/snapshot/*.parquet'
    EVENT_PATH = 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=PROPERTY/is_page_view=*/event=EVENT/*.parquet'

    CONTENT_COL = [
        'content_id', 'publish_time', 'title',  'description', 'is_adult',
        'cat1', 'tags', 'content_ner', 'series', 'title_embedding', 'author', 'author_penname', 'display_author',
        'provider', 'show_platform', 'renewal', 'chapter_newest_first_publish_time'
    ]

    EVENT_NAME = 'planet_content_page_view'
    EVENT_TO_CONDITION_MAPPING = {'page_info.page': 'planet_novels'}

    USER_PROFILE = {
        'tag': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/tag/INPUT_DATE/CONTENT_TYPE/*.parquet',
        'category': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/category/INPUT_DATE/CONTENT_TYPE/*.parquet',
        'embedding': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/embedding/INPUT_DATE/CONTENT_TYPE/*.parquet',
        'meta': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/gamania/meta/meta/INPUT_DATE/*.parquet'
    }

    COLUMN_TO_RENAME = {
        'item_content': [('title_embedding', 'item_title_embedding'), ('content_id', 'uuid')],
        'user_category': [('data', 'user_category')],
        'user_tag': [('data', 'user_tag')],
        'user_meta': []
    }

    USER_PROFILE_COLUMNS = {
        'user_category': ['user_category'],
        'user_tag': ['user_tag'],
        'user_embedding': ['user_title_embedding', 'user_tag_embedding']
    }

    FINAL_COLS = [
        'userid', 'uuid', 'timestamp', 'date', 'age', 'gender',
        'user_category', 'user_tag', 'user_title_embedding', 'user_tag_embedding',
        'publish_time', 'title',  'description', 'is_adult',
        'cat1', 'tags', 'series', 'item_title_embedding', 'author', 'author_penname', 'display_author',
        'provider', 'show_platform', 'renewal', 'chapter_newest_first_publish_time', 'y'
    ]

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {
        'content': ['cat1', 'tags']
    }


class PlanetVideoUser2ItemConfig(BaseConfig):
    CONTENT_COL = ['content_id', 'title', 'publish_time', 'author', 'site_name', 'tags', 'cat0', 'cat1', 'cat2', 'status', 'title_embedding', 'content_ner']
    EVENT_NAME = ['planet_videos_content_click', 'planet_banner_click', 'planet_videos_content_click']
    USER_PROFILE = {
        'tag': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/tag/INPUT_DATE/CONTENT_PATH/*.parquet',
        'category': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/category/INPUT_DATE/CONTENT_PATH/*.parquet',
        'title_embedding': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/embedding/INPUT_DATE/CONTENT_PATH/*.parquet',
        'meta': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/gamania/meta/meta/INPUT_DATE/*.parquet'
    }
    EVENT_OF_CONTENT_TYPE_CONDITION = [
        {'event': 'planet_banner_click', 'click_info_map.sec': 'video'},
        {'event': 'planet_content_click', 'click_info_map.type': 'video', 'page_info_map.page': 'planet'},
        {'event': 'planet_videos_content_click'}
    ]
    COLUMN_TO_RENAME = {
        'item_content': [('title_embedding', 'item_title_embedding')],
        'user_video_title_embedding': [('userid', 'openid'), ('data', 'user_video_title_embedding')],
        'user_video_category': [('userid', 'openid'), ('data', 'user_video_category')],
        'user_video_tag': [('userid', 'openid'), ('data', 'user_video_tag')],
        'user_news_title_embedding': [('userid', 'openid'), ('data', 'user_news_title_embedding')],
        'user_news_category': [('userid', 'openid'), ('data', 'user_news_category')],
        'user_meta': [('userid', 'openid')]
    }

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {
        'content': ['tags', 'cat0', 'cat1']
    }


class PlanetNewsItem2ItemConfig(BaseConfig):

    EVENT_PATH = 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=beanfun/is_page_view=*/event=*/*.parquet'
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=beanfun/content_type=planet_news/snapshot/*.parquet'

    # used by `filter_dataframe`
    TARGET_EVENT = {
        'candidate_pool': ['planet_content_page_view'],
        'positive_pool': ['planet_content_click']
    }

    TARGET_INFO = {
        'candidate_pool': {'page_info': [('uuid', 'uuid'), ('name', 'name'), ('page', 'page')]},  # {info: [(original_key, alias_key), (original_key2, alias_key2)]}
        'positive_pool': {'page_info': [('uuid', 'uuid'), ('name', 'name'), ('page', 'page')], 'click_info': [('uuid', 'click_uuid'), ('name', 'click_name'), ('sec', 'sec')]}
    }

    TARGET_COLS = {
        'candidate_pool': ['uuid', 'name', 'date', 'timestamp'],
        'positive_pool': ['uuid', 'name', 'click_uuid', 'click_name', 'date', 'timestamp'],
        'content_pool': ['content_id', 'site_name', 'cat0', 'cat1', 'cat2', 'title_embedding', 'content_ner', 'publish_time'],
        'all_pool': ['uuid', 'name', 'click_uuid', 'click_name', 'date', 'timestamp', 'y'],
        'view_also_view': ['content_uuid', 'view_also_view_json', 'date'],
        'final_pool': ['uuid', 'name', 'cat0', 'cat1', 'cat2', 'title_embedding', 'content_ner', 'site_name', 'publish_time',
                       'click_uuid', 'click_name', 'click_cat0', 'click_cat1', 'click_cat2', 'click_title_embedding', 'click_content_ner', 'click_site_name', 'click_publish_time',
                       'view_also_view_score', 'y', 'timestamp', 'date']
    }

    FILTER_CONDITIONS = {
        'candidate_pool': {'page': ['planet_news'], 'is_additive_view': ['False']},  # {target_col: [value_1, value_2, ...]}
        'positive_pool': {'page': ['planet_news'], 'sec': ['rec_item_to_item']}
    }

    # Need to rename feature cols for positive sample to avoid duplicated naming for candidate and positive sample in df_positive
    # {config_key: {old_col_name1: new_col_name1, old_col_name2: new_col_name2, ...}}
    RENAMED_COLS = {
        'all_pool': {'site_name': 'click_site_name', 'cat0': 'click_cat0', 'cat1': 'click_cat1', 'cat2': 'click_cat2',
                     'title_embedding': 'click_title_embedding', 'content_ner': 'click_content_ner', 'publish_time': 'click_publish_time'},
        'negative_pool': {'name': 'click_name'},
        'view_also_view': {'content_uuid': 'uuid'}
    }

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {
        'content_pool': ['cat0', 'cat1', 'cat2', 'site_name']
    }
