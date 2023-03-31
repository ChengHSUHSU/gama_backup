from src.gcsreader.config.base import GeneralConfig
from src.gcsreader.config.event import BaseEventConfig
from src.gcsreader.config.content import BaseContentConfig
from src.gcsreader.config.profile import BaseProfileConfig
from src.gcsreader.config.metrics import BaseMetricsConfig


# Planet news hot model
PlanetNewsHotModelGeneralConfig = GeneralConfig.update({
    'PROCESS_CONTENT_TYPES': ['planet_news'],
    'PROCESS_METRICS_TYPES': ['snapshot_popularity', 'snapshot_statistics'],
    'CONTENT_TYPE_TO_PROPERTY_TYPE': {'planet_news': 'planet'},

    # columns to be renamed at each step in format of {'step': [(old_col_name, new_col_name), ...]}
    'STEPS_TO_RENAMCOLS':  {'event': [('uuid', 'content_id')],
                            'metrics': {'snapshot_statistics': [('uuid', 'content_id')]}},

    # columns to find distinct values for encoding purpose at data preprocess stage
    'COLS_TO_ENCODE': {
        'content': ['cat0', 'cat1', 'tags']
    },
    'REQUISITE_COLS': {
        'event': ['userid', 'content_id', 'date', 'hour', 'event'],
        'content': ['content_id', 'cat0', 'cat1', 'tags'],
        'metrics': {'statistics': '*'}
    }
})

PlanetNewsHotModelEventConfig = BaseEventConfig.update({
    'EVENT_PATH': 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=beanfun/is_page_view=*/event=*/*.parquet',
    'HOURLY_EVENT_PATH': 'gs://event-PROJECT_ID/event_hourly/date=INPUT_DATE/hour=INPUT_HOUR/property=beanfun/is_page_view=*/event=*/*.parquet',

    # events and condictions
    'EVENT_OF_CONTENT_TYPE_CONDITIONS': {'planet_news': [
                {'event': 'planet_home_page_impression', 'impression_info_map["sec"]': 'news'},
                {'event': 'planet_content_impression'},
                {'event': 'home_page_impression', 'impression_info_map["type"]': 'planet_news'},
                {'event': 'exploration_page_content_impression', 'impression_info_map["type"]': 'planet_news'},
                {'event': 'planet_content_click', 'click_info_map["type"]': 'news'},
                {'event': 'home_page_content_click', 'click_info_map["type"]': 'planet_news'},
                {'event': 'home_page_content_click', 'click_info_map["type"]': 'planet_news_comment'},
                {'event': 'exploration_page_content_click', 'click_info_map["type"]': 'planet_news'}]}
})

PlanetNewsHotModelProfileConfig = BaseProfileConfig.update({
    'USER_PROFILE_CONDITIONS': {'planet_news': {'user_profile_path': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/PROFILE_TYPE/INPUT_DATE/BLOB_PATH/*.parquet',
                                                'blob_path': 'news',
                                                'save_type': 'diff'}},
    'META_COLS': []
})

PlanetNewsHotModelContentConfig = BaseContentConfig.update({
    'CONTENT_PATH': 'gs://content-PROJECT_ID/content_daily/property=beanfun/content_type=planet_news/snapshot/*.parquet'
})

PlanetNewsHotModelMetricsConfig = BaseMetricsConfig.update({
    'METRICS_POPULARITY_PATH': 'gs://pipeline-PROJECT_ID/metrics/popularity/POPULARITY_FOLDER/INPUT_DATE/INPUT_HOUR/popularity_news.csv',
    'STATISTICS_PATH': 'gs://pipeline-PROJECT_ID/statistics/STATISTICS_FOLDER/INPUT_DATE/INPUT_HOUR/statistics_news.csv',
    'POPULARITY_FOLDER': {'planet_news': 'news'},
    'STATISTICS_FOLDER': {'planet_news': 'news'},
    'STATISTICS_COLS': ['1_last_day_total_view_count', '3_last_day_total_view_count', '7_last_day_total_view_count', '30_last_day_total_view_count', 
                        '1_last_day_total_click_count', '3_last_day_total_click_count', '7_last_day_total_click_count', '30_last_day_total_click_count']
})

PlanetNewsHotModelConfig = PlanetNewsHotModelGeneralConfig.update(PlanetNewsHotModelEventConfig) \
                                                                  .update(PlanetNewsHotModelProfileConfig) \
                                                                  .update(PlanetNewsHotModelContentConfig) \
                                                                  .update(PlanetNewsHotModelMetricsConfig)
