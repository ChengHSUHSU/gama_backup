from src.gcsreader.config.base import GeneralConfig
from src.gcsreader.config.event import BaseEventConfig
from src.gcsreader.config.content import BaseContentConfig
from src.gcsreader.config.profile import BaseProfileConfig
from src.gcsreader.config.metrics import BaseMetricsConfig

ShowUser2ItemGeneralConfig = GeneralConfig.update({
    'PROCESS_CONTENT_TYPES': ['show'],
    'PROCESS_PROFILE_TYPES': ['title_embedding', 'category', 'meta'],
    'PROCESS_METRICS_TYPES': ['popularity', 'statistics'],
    'CONTENT_TYPE_TO_PROPERTY_TYPE': {'show': 'beanfun'},

    'STEPS_TO_RENAMCOLS': {
        'event': [],
        'content': [('title_embedding', 'item_title_embedding'), ('content_id', 'uuid')],
        'metrics': {'popularity': [('content_id', 'uuid')]},
        'user_profile': {'title_embedding': [('data', 'user_title_embedding')],
                         'category': [('data', 'user_category')]},
        'club_user_profile': {'title_embedding': [('user_title_embedding', 'club_user_title_embedding')],
                              'category': [('user_category', 'club_user_category')]}
    },

    'COLS_TO_ENCODE': {
        'content': ['cat0']
    },

    'USER_PROFILE_LIST': ['user_category', 'user_embedding', 'club_user_category', 'club_user_embedding'],

    'REQUISITE_COLS': {
        'event': ['userid', 'uuid', 'timestamp', 'date', 'hour'],
        'content': [
                    'uuid', 'title', 'item_title_embedding', 'cat0', 'publish_time', 'content_ner',
                    'is_adult'
                ],
        'metrics': {'popularity': ['uuid', 'final_score', 'date', 'hour'], 
                    'statistics': ['uuid', '1_last_day_total_view_count', '1_last_day_total_click_count', 
                                    '1_last_day_interaction_count', '3_last_day_total_view_count', '3_last_day_total_click_count', 
                                    '5_last_day_total_view_count', '5_last_day_total_click_count', '5_last_day_interaction_count', 'date',
                                    'hour']
                },
        'user_profile': {'title_embedding': ['userid', 'date', 'user_title_embedding'],
                        'category': ['userid', 'date', 'user_category'],
                        'meta': ['userid', 'gender', 'age', 'gamapay_used', 'planet_news_used', 'planet_comic_used',
                                 'planet_novel_used', 'game_used', 'shopping_used', 'gash_used']}
    },
    
    'FINAL_COLS': [
        'userid', 'uuid', 'timestamp', 'date', 'hour', 'age', 'gender', 'title',
        'final_score', '1_last_day_total_view_count', '1_last_day_total_click_count', 
        '1_last_day_interaction_count', '3_last_day_total_view_count', '3_last_day_total_click_count', 
        '5_last_day_total_view_count', '5_last_day_total_click_count', '5_last_day_interaction_count', 
        'user_category', 'user_title_embedding', 
        'is_adult', 'publish_time', 'cat0', 'content_ner', 'item_title_embedding', 'y',
        'gamapay_used', 'planet_news_used', 'planet_comic_used', 'planet_novel_used', 'game_used',
        'shopping_used', 'gash_used', 'club_user_category' ,'club_user_title_embedding'
    ],

    'PROCESS_META_CONTENT_TYPES_INFO': [
        {'content_type': 'club', 'profile_types': ['category', 'title_embedding'], 'rename_by': 'club_user_profile'}
    ]
})

ShowUser2ItemEventConfig = BaseEventConfig.update({
    'EVENT_PATH': 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=beanfun/is_page_view=*/event=*/*.parquet',
    'EVENT_OF_CONTENT_TYPE_CONDITIONS': {'show': [{'event': 'show_detail_page_view', 'page_info_map["page"]': 'show_detail'}]}
})

ShowUser2ItemProfileConfig = BaseProfileConfig.update({
    'USER_PROFILE_CONDITIONS': {'show': {'user_profile_path': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/show/PROFILE_TYPE/INPUT_DATE/*.parquet', 
                                         'blob_path': '', 
                                         'save_type': 'diff'}, # 記得 planet comic 效果好
                                'beanfun_meta': {'user_profile_path': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/meta/meta/INPUT_DATE/*.parquet', 
                                                'blob_path': '', 
                                                'save_type': 'snapshot'},
                                'club': {'user_profile_path': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/club/PROFILE_TYPE/INPUT_DATE/*.parquet', 
                                         'blob_path': '', 
                                         'save_type': 'diff'}
                             },
    'META_COLS': [('gender', 'data', '$.gender'), 
                  ('age', 'data', '$.age'), 
                  ('gamapay_used', 'data', '$.gamapay_used'),
                  ('planet_news_used', 'data', '$.planet_news_used'),
                  ('planet_comic_used', 'data', '$.planet_comic_used'),
                  ('planet_novel_used', 'data', '$.planet_novel_used'),
                  ('game_used', 'data', '$.game_used'),
                  ('shopping_used', 'data', '$.shopping_used'),
                  ('gash_used', 'data', '$.gash_used')]
})

ShowUser2ItemContentConfig = BaseContentConfig.update({
    'CONTENT_PATH': 'gs://content-PROJECT_ID/content_daily/property=beanfun/content_type=show/snapshot/*.parquet'
})

ShowUser2ItemMetricsConfig = BaseMetricsConfig.update({
    'METRICS_POPULARITY_PATH': 'gs://pipeline-PROJECT_ID/metrics/popularity/POPULARITY_FOLDER/INPUT_DATE/INPUT_HOUR/*.csv',
    'STATISTICS_PATH': 'gs://pipeline-PROJECT_ID/statistics/STATISTICS_FOLDER/INPUT_DATE/INPUT_HOUR/*.csv',
    'POPULARITY_FOLDER': {'show': 'show'},
    'STATISTICS_FOLDER': {'show': 'show'}
})

ShowUser2ItemConfig = (ShowUser2ItemGeneralConfig.update(ShowUser2ItemEventConfig)
                                                 .update(ShowUser2ItemProfileConfig)
                                                 .update(ShowUser2ItemContentConfig)
                                                 .update(ShowUser2ItemMetricsConfig)
                    )
