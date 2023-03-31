
class NSFWTextConfig():
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=nownews/content_type=nownews_news/snapshot/*.parquet'
    CONTENT_COL = ['content_id', 'title', 'contents', 'is_other_source', 'is_adult', 'category_name',
                   'cat0', 'cat1', 'cat2', 'tags', 'seo_keywords', 'publish_time', 'content_ner']