from .base import BaseConfig


class ClubPostUser2ItemConfig(BaseConfig):

    POPULARITY_FOLDER = 'beanfun_club_post'
    PROPERTY = 'beanfun'

    CONTENT_COL = ['content_id', 'title', 'publish_time', 'cat0', 'title_embedding']
    POPULARITY_COLS = ['uuid', 'token_count', 'interaction_score', 'final_score', 'date', 'hour']
    GAMANIA_META_COL = ['userid', 'gender', 'age', 'planet_news_used', 'planet_comic_used', 'gamapay_used', 'shopping_used', 'game_used', 'gash_used']
    BEANFUN_META_COL = ['userid', 'planet_user_level', 'club_user_level']

    EVENT_NAME = 'club_post_page_view'
    EVENT_TO_CONDITION_MAPPING = {'page_info.page': ['club_post', 'club_post_comment']}

    USER_PROFILE = {
        'post_embedding': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/club/embedding/INPUT_DATE/*.parquet',
        'news_embedding': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/planet/embedding/INPUT_DATE/news/*.parquet',
        'category': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/club/category/INPUT_DATE/*.parquet',
        'gamania_meta': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/gamania/meta/meta/INPUT_DATE/*.parquet',
        'beanfun_meta': 'gs://pipeline-PROJECT_ID/user_profiles_cassandra/beanfun/meta/meta/INPUT_DATE/*.parquet'
    }

    COLUMN_TO_RENAME = {
        'content': [('title_embedding', 'item_title_embedding')],
        'popularity': [('uuid', 'content_id')],
        'user_post_embedding': [('data', 'user_post_embedding')],
        'user_category': [('data', 'user_category')],
    }

    # columns to find distinct values for encoding purpose at data preprocess stage
    COLS_TO_ENCODE = {
        'content': ['cat0']
    }
