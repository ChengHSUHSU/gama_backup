from pyspark.sql.functions import col, udf, get_json_object
from pyspark.sql.types import StringType, MapType, FloatType
from src.gcsreader.config.nownews_config import NownewsNewsItem2ItemConfig
from utils import get_partition_path_with_check
from src.gcsreader import GcsReader
from datetime import datetime, timedelta
import json


VIEW_ALSO_VIEW_COL = 'view_also_view_json'


class NownewsNewsItem2ItemReader(GcsReader):

    CONFIG = NownewsNewsItem2ItemConfig
    CONTENT_TYPE = 'nownews_news'
    PROPERTY = 'nownews'

    def __init__(self, project_id, sql_context, run_time, days=30):

        self.project_id = project_id
        self.sql_context = sql_context
        self.days = days
        self.event_name = self.CONFIG.EVENT_NAME
        self.run_time = (datetime.strptime(run_time, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')  # training on date (run_time-1)

    def get_event_data(self, userid='openid'):
        input_path = self.CONFIG.EVENT_PATH.replace('PROJECT_ID', self.project_id) \
            .replace('PROPERTY', self.PROPERTY) \
            .replace('EVENT', self.event_name)

        input_path = input_path.replace('is_page_view=*', 'is_page_view=True') \
            if 'page_view' in self.event_name else input_path.replace('is_page_view=*', 'is_page_view=False')

        base_path = input_path[:input_path.find('date=')]
        path_list, _ = get_partition_path_with_check(input_path[:input_path.find('*')], self.run_time, self.days)
        path_list = [p + '*.parquet' for p in path_list]

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        if self.event_name.split('_')[-1] == 'view':
            info_prefix = 'page'
        elif self.event_name.split('_')[-1] == 'click':
            info_prefix = 'click'

        df = df.select(
            userid,
            get_json_object('page_info', '$.uuid').alias('page_uuid'),
            get_json_object(f'{info_prefix}_info', '$.uuid').alias('click_uuid'),
            get_json_object(f'{info_prefix}_info', '$.sec').alias('sec'),
            get_json_object(f'{info_prefix}_info', '$.type').alias('type'),
            'date',
            'timestamp'
        )

        condition = ((df.sec == 'extend') & (df.type == 'content'))
        df = df.filter(condition)
        df = df.withColumn('date', df.date.cast('string'))
        df = df.drop('sec').drop('type')

        return df

    def get_content_data(self):
        df_content = self.sql_context.read.parquet(
            self.CONFIG.CONTENT_PATH.replace('PROJECT_ID', self.project_id))
        df_content = df_content.select(self.CONFIG.CONTENT_COL)
        return df_content

    def get_view_also_view_data(self):

        input_path = self.CONFIG.VIEW_ALSO_VIEW_PATH \
                        .replace('PROJECT_ID', self.project_id) \
                        .replace('INPUT_DATE', self.run_time) \
                        .replace('CONTENT_TYPE', self.CONTENT_TYPE)

        df = self.sql_context.read.options(**{'header': 'true', 'escape': '"'}).csv(input_path)

        df = df.withColumn(
            VIEW_ALSO_VIEW_COL,
            udf(lambda x: json.loads(x), MapType(StringType(), FloatType(), False))(col(VIEW_ALSO_VIEW_COL)))

        return df
