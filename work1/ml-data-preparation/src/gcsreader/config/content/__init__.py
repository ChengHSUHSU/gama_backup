from src.gcsreader.config.base import Config

BaseContentConfig = Config({
    'CONTENT_PATH': 'gs://content-PROJECT_ID/content_daily/property=*/content_type=CONTENT_TYPE/snapshot/*.parquet'
})
