from src.gcsreader.config.base import GeneralConfig
from src.gcsreader.config.event import BaseEventConfig
from src.gcsreader.config.content import BaseContentConfig
from src.gcsreader.config.profile import BaseProfileConfig
from src.gcsreader.config.metrics import BaseMetricsConfig

# Jollybuy goods user2item
JollybuyGoodsUser2ItemGeneralConfig = GeneralConfig.update({
    'PROCESS_CONTENT_TYPES': ['jollybuy_goods'],
    'PROCESS_PROFILE_TYPES': ['title_embedding', 'category', 'tag', 'meta'],
    'PROCESS_METRICS_TYPES': ['popularity', 'snapshot_popularity'],
    'CONTENT_TYPE_TO_PROPERTY_TYPE': {'jollybuy_goods': 'jollybuy'},

    # columns to be renamed at each step in format of {'step': [(old_col_name, new_col_name), ...]}
    'STEPS_TO_RENAMCOLS':  {'event': [('uuid', 'content_id')],
                            'content': [('title_embedding', 'item_title_embedding')],
                            'metrics': {'popularity': [('uuid', 'content_id')],
                                        'snapshot_popularity': [('uuid', 'content_id')]},
                            'user_profile': {'title_embedding': [('data', 'user_title_embedding')],
                                             'category': [('data', 'user_category')],
                                             'tag': [('data', 'user_tag')]}},

    # filtering
    # filter preference profile, such as category profile, tag profile, etc.
    # format: {tag_type: ('pref', threshold)}
    'PREFERENCE_FILTER_CONDITIONS': {
        'user_tag': {'editor': ('pref', 0.02)}
    },

    # filter out unused profile or content meta data by user_id or content_id
    'PRUNED_BY_USERID_COLS': ['user_title_embedding', 'user_category', 'user_tag', 'user_meta'],
    'PRUNED_BY_CONTENT_ID_COLS': ['item_content'],

    # columns to find distinct values for encoding purpose at data preprocess stage
    'COLS_TO_ENCODE': {
        'content': ['cat0', 'cat1', 'tags']
    },
    'REQUISITE_COLS': {'event': ['userid', 'content_id', 'date', 'hour'],
                       'content': ['content_id', 'item_title_embedding', 'cat0', 'cat1', 'tags', 'publish_time'],
                       'metrics': {'popularity': ['content_id', 'final_score', 'date', 'hour'],
                                   'snapshot_popularity': ['content_id', 'final_score']},
                       'user_profile': {'title_embedding': ['userid', 'date', 'user_title_embedding'],
                                        'category': ['userid', 'date', 'user_category'],
                                        'tag': ['userid', 'date', 'user_tag'],
                                        'meta': ['userid', 'gender', 'age']}},
    'FINAL_COLS': ['userid', 'content_id', 'age', 'gender', 'user_title_embedding',
                   'user_category', 'user_tag', 'final_score', 'tags', 'item_title_embedding',
                   'cat0', 'cat1', 'y', 'publish_time'],

})

JollybuyGoodsUser2ItemEventConfig = BaseEventConfig.update({
    # TODO: Make it compatible to property equal to jollybuy or beanfun
    'EVENT_PATH': 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=jollybuy/is_page_view=*/event=*/*.parquet',

    # if we only consider 'jb_home_page_content_click' and 'jb_home_page_content_impression' with sec == 'maybe_like'
    # the sample size would be too small (around 500 positive samples for 30 days)
    'EVENT_OF_CONTENT_TYPE_CONDITIONS': {'jollybuy_goods': [{'event': 'jb_goods_page_view', 'page_info_map["page"]': 'jb_goods'}]}
})

JollybuyGoodsUser2ItemProfileConfig = BaseProfileConfig.update({
    'USER_PROFILE_CONDITIONS': {'jollybuy_goods': {'user_profile_path': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/jollybuy/PROFILE_TYPE/INPUT_DATE/BLOB_PATH/*.parquet',
                                                   'blob_path': 'goods',
                                                   'save_type': 'diff'},
                                'gamania_meta': {'user_profile_path': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/gamania/meta/meta/INPUT_DATE/*.parquet', 'blob_path': '', 'save_type': 'snapshot'}},
    'META_COLS': [('gender', 'data', '$.gender'), ('age', 'data', '$.age')]
})

JollybuyGoodsUser2ItemContentConfig = BaseContentConfig.update({
    'CONTENT_PATH': 'gs://content-PROJECT_ID/content_daily/property=jollybuy/content_type=jollybuy_goods/snapshot/*.parquet'
})

JollybuyGoodsUser2ItemMetricsConfig = BaseMetricsConfig.update({
    'METRICS_POPULARITY_PATH': 'gs://pipeline-PROJECT_ID/metrics/popularity/POPULARITY_FOLDER/INPUT_DATE/INPUT_HOUR/popularity_jollybuy_goods.csv',
    'POPULARITY_FOLDER': {'jollybuy_goods': 'jollybuy_goods'}
})

JollybuyGoodsUser2ItemConfig = JollybuyGoodsUser2ItemGeneralConfig.update(JollybuyGoodsUser2ItemEventConfig) \
                                                                  .update(JollybuyGoodsUser2ItemProfileConfig) \
                                                                  .update(JollybuyGoodsUser2ItemContentConfig) \
                                                                  .update(JollybuyGoodsUser2ItemMetricsConfig)
