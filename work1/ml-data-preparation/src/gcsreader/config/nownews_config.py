from .base import BaseConfig


class NownewsNewsItem2ItemConfig(BaseConfig):
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=nownews/content_type=nownews_news/snapshot/*.parquet'
    EVENT_PATH = 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=PROPERTY/is_page_view=*/event=EVENT/*.parquet'

    CONTENT_COL = [
        'content_id', 'publish_time',
        'tags', 'cat0', 'cat1',
        'title_embedding', 'content_ner', 'title'
        ]

    EVENT_NAME = 'nn_page_content_click'

    COLUMN_TO_RENAME = {
        'user_event': [('trackid', 'userid')],
        'item_content': [('title_embedding', 'item_title_embedding')]
    }

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {
        'content': ['cat0', 'cat1', 'tags']
    }


class NownewsNewsUser2ItemConfig(BaseConfig):
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=nownews/content_type=nownews_news/snapshot/*.parquet'
    EVENT_PATH = 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=PROPERTY/is_page_view=*/event=EVENT/*.parquet'

    CONTENT_COL = [
        'content_id', 'publish_time',
        'cat0', 'cat1', 'tags', 'seo_keywords', 'category_name',
        'title_embedding', 'content_ner', 'title'
    ]

    EVENT_NAME = 'nn_news_page_view'

    USER_PROFILE = {
        'tag': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/nownews/tag/INPUT_DATE/*.parquet',
        'category': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/nownews/category/INPUT_DATE/*.parquet',
        'embedding': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/nownews/embedding/INPUT_DATE/*.parquet',
        'meta': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/gamania/meta/meta/INPUT_DATE/*.parquet'
    }

    COLUMN_TO_RENAME = {
        'item_content': [('title_embedding', 'item_title_embedding')],
        'user_embedding': [('data', 'user_title_embedding')],
        'user_category': [('data', 'user_category')],
        'user_tag': [('data', 'user_tag')],
        'user_meta': []
    }

    FINAL_COLS = [
        'userid', 'content_id', 'timestamp', 'date', 'age', 'gender',
        'user_category', 'user_tag', 'user_title_embedding', 'title',
        'publish_time', 'cat0', 'cat1', 'tags', 'category_name',
        'seo_keywords', 'content_ner', 'item_title_embedding', 'y'
    ]

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {
        'content': ['cat0', 'cat1', 'tags']
    }
