from src.gcsreader.config.base import Config

BaseMetricsConfig = Config({
    'METRICS_POPULARITY_PATH': 'gs://pipeline-PROJECT_ID/metrics/popularity/POPULARITY_FOLDER/INPUT_DATE/INPUT_HOUR/*.csv',
    'POPULARITY_FOLDER': {},
    'VIEW_ALSO_VIEW_PATH': 'gs://pipeline-PROJECT_ID/metrics/METRICS/CONTENT_TYPE/INPUT_DATE/METRICS_CONTENT_TYPE.csv'
})
