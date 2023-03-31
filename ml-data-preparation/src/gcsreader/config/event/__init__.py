from src.gcsreader.config.base import Config

BaseEventConfig = Config({
    'EVENT_PATH': 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=*/is_page_view=*/event=*/*.parquet',
    'HOURLY_EVENT_PATH': 'gs://event-PROJECT_ID/event_hourly/date=INPUT_DATE/hour=*/property=*/is_page_view=*/event=*/*.parquet',
    'EVENT_OF_CONTENT_TYPE_CONDITIONS': {}
})
