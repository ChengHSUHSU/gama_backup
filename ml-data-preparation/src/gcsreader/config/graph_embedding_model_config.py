from src.gcsreader.config.base import BaseConfig


class GraphEmbeddingModelConfig(BaseConfig):

    # base config
    CONTENT_TYPE_TO_PROPERTY_TYPE = {'jollybuy_goods': 'jollybuy', 'nownews_news': 'nownews'}

    STEPS_TO_RENAMCOLS = {'event': [('uuid', 'content_id')],
                          'metrics': {'popularity': [('final_score', 'popularity_score')]}}

    REQUISITE_COLS = {'event': ['userid', 'event', 'content_id', 'date', 'timestamp'],
                      'metrics': {'popularity': ['uuid', 'popularity_score', 'date', 'hour']},
                      'user_profile': {'category': ['userid', 'profile', 'date', 'data']}}

    REQUISITE_CONTENT_COLS = {'planet_comics_book': ['content_id', 'title', 'publish_time', 'cat0', 'cat1', 'cat2', 'tags'],
                              'jollybuy_goods': ['content_id', 'title', 'publish_time', 'cat0', 'cat1', 'tags',
                                                 'count_30d_sales', 'count_2d_sales', 'count_total_sales',
                                                 'count_2d_favorite', 'count_total_favorite', 'count_2d_browse', 'count_total_browse',
                                                 'count_comment', 'is_promotion'],
                              'default': ['content_id', 'title', 'publish_time', 'cat0', 'cat1', 'tags']}

    # event config
    EVENT_PATH = 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=*/is_page_view=*/event=*/*.parquet'
    EVENT_OF_CONTENT_TYPE_CONDITIONS = {'planet_news': [{'event': 'planet_content_page_view', 'page_info_map["page"]': 'planet_news'}],
                                        'planet_comics_book': [{'event': 'planet_intro_page_view', 'page_info_map["page"]': 'planet_comics_intro'},
                                                               {'event': 'web_planet_intro_page_view', 'page_info_map["page"]': 'web_planet_comics_intro'}],
                                        'planet_novel_book': [{'event': 'planet_intro_page_view', 'page_info_map["page"]': 'planet_novels_intro'}],
                                        'planet_video': [{'event': 'planet_banner_click', 'click_info_map["sec"]': 'video', 'click_info_map["type"]': 'play'},
                                                         {'event': 'planet_banner_click', 'click_info_map["sec"]': 'video', 'click_info_map["type"]': 'function_panel'},
                                                         {'event': 'planet_content_click', 'page_info_map["page"]': 'planet', 'click_info_map["type"]': 'video'},
                                                         {'event': 'planet_videos_content_click', 'click_info_map["type"]': 'play'},
                                                         {'event': 'planet_videos_content_click', 'click_info_map["type"]': 'function_panel'},
                                                         {'event': 'home_page_content_click', 'click_info_map["type"]': 'planet_video'},
                                                         {'event': 'home_page_content_click', 'click_info_map["type"]': 'planet_video_comment'},
                                                         {'event': 'home_page_content_play', 'page_info_map["type"]': 'planet_video', 'page_info_map["action"]': 'auto_play'}],
                                        'jollybuy_goods': [{'event': 'jb_goods_page_view', 'page_info_map["page"]': 'jb_goods'}],
                                        'nownews_news': [{'event': 'nn_news_page_view', 'page_info_map["page"]': 'news'}]}

    # content config
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=*/content_type=CONTENT_TYPE/snapshot/*.parquet'

    # metrics config
    METRICS_POPULARITY_PATH = 'gs://pipeline-PROJECT_ID/metrics/popularity/POPULARITY_FOLDER/INPUT_DATE/INPUT_HOUR/*.csv'
    POPULARITY_FOLDER = {'planet_news': 'news',
                         'planet_comics_book': 'comic',
                         'planet_novel_book': 'novel',
                         'planet_video': 'video',
                         'jollybuy_goods': 'jollybuy_goods',
                         'nownews_news': 'nownews_news'}

    # user profile config
    PLAENT_PROFILE_PATH = 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/PROFILE_TYPE/INPUT_DATE/BLOB_PATH/*.parquet'
    NOWNEWS_PROFILE_PATH = 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/nownews/PROFILE_TYPE/INPUT_DATE/*.parquet'
    JOLLYBUY_PROFILE_PATH = 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/jollybuy/PROFILE_TYPE/INPUT_DATE/BLOB_PATH/*.parquet'
    GAMANIA_META_PROFILE_PATH = 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/gamania/meta/meta/INPUT_DATE/*.parquet',

    USER_PROFILE_CONDITIONS = {'planet_news': {'user_profile_path': PLAENT_PROFILE_PATH, 'blob_path': 'news', 'save_type': 'diff'},
                               'planet_novel_book': {'user_profile_path': PLAENT_PROFILE_PATH, 'blob_path': 'novel', 'save_type': 'diff'},
                               'planet_comics_book': {'user_profile_path': PLAENT_PROFILE_PATH, 'blob_path': 'comic', 'save_type': 'diff'},
                               'planet_video': {'user_profile_path': PLAENT_PROFILE_PATH, 'blob_path': 'video', 'save_type': 'diff'},
                               'nownews_news': {'user_profile_path': NOWNEWS_PROFILE_PATH, 'blob_path': '', 'save_type': 'diff'},
                               'jollybuy_goods': {'user_profile_path': JOLLYBUY_PROFILE_PATH, 'blob_path': 'goods', 'save_type': 'diff'},
                               'gamania_meta': {'user_profile_path': GAMANIA_META_PROFILE_PATH, 'blob_path': '', 'save_type': 'snapshot'}}

    # model specific config
    PROCESS_CONTENT_TYPES = ['planet_news', 'planet_comics_book', 'planet_novel_book', 'planet_video', 'jollybuy_goods', 'nownews_news']
    PROCESS_PROFILE_TYPES = ['category']
    PROCESS_METRICS_TYPES = ['popularity']
