from pyspark.sql.functions import col, get_json_object, udf, input_file_name
from pyspark.sql.types import StringType
from src.gcsreader.config.nownews_config import NownewsNewsUser2ItemConfig
from src.gcsreader import GcsReader
from datetime import datetime, timedelta
import logging
import json


class NownewsNewsUser2ItemReader(GcsReader):

    CONFIG = NownewsNewsUser2ItemConfig
    PROPERTY = 'nownews'
    CONTENT_TYPE = 'nownews_news'

    def __init__(self, project_id, sql_context, content_type, run_time, days=30, logger=logging):

        self.project_id = project_id
        self.sql_context = sql_context
        self.content_type = content_type
        self.days = days

        self.run_time = (datetime.strptime(run_time, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')  # training on date (run_time-1)
        self.event_name = self.CONFIG.EVENT_NAME
        self.logger = logger

    def get_event_data(self):

        self.logger.info('[Data Preparation][Nownews News User2Item] Get event data')

        input_path = self.CONFIG.EVENT_PATH \
            .replace('PROPERTY', self.PROPERTY) \
            .replace('PROJECT_ID', self.project_id) \
            .replace('EVENT', self.event_name)

        base_path = input_path[:input_path.find('date=')]

        self.logger.info(f'[Data Preparation][Nownews News User2Item] input_path={input_path}')
        self.logger.info(f'[Data Preparation][Nownews News User2Item] base_path={base_path}')

        path_list = self._get_existed_blobs(input_path, prefix=f'event_daily/date=INPUT_DATE/property={self.PROPERTY}', bucket_idx=2)

        self.logger.info(f'[Data Preparation][Nownews News User2Item] path_list={path_list}')

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        df = self._add_userid(df)

        df = df.select(
            'timestamp', 'date', 'userid',
            get_json_object('page_info', '$.uuid').alias('content_id')
        )
        df = df.withColumn('date', df.date.cast('string'))

        return df.select('userid', 'content_id', 'date', 'timestamp')

    def get_content_data(self):

        self.logger.info('[Data Preparation][Nownews News User2Item] Get content data')

        input_path = self.CONFIG.CONTENT_PATH \
            .replace('PROJECT_ID', self.project_id) \
            .replace('CONTENT_TYPE', self.CONTENT_TYPE)

        df_content = self.sql_context.read.parquet(input_path)
        df_content = df_content.select(self.CONFIG.CONTENT_COL)
        return df_content

    def get_user_embedding(self):

        self.logger.info('[Data Preparation][Nownews News User2Item] Get user embedding')

        EMBEDDINH_KEY = 'title'

        path_list, base_path = self._get_user_profile_input_path(user_profile_type='embedding', save_type='diff')

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        df = self._add_process_date_to_profile(df, profile_type='embedding')

        df = df.withColumn('data', udf(lambda x: json.loads(x).get(EMBEDDINH_KEY, None), StringType())(col('data')))

        df = df.select(['userid', 'date', 'data'])

        return df

    def get_user_category(self):

        self.logger.info('[Data Preparation][Nownews News User2Item] Get user category')

        path_list, base_path = self._get_user_profile_input_path(user_profile_type='category', save_type='diff')

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        df = self._add_process_date_to_profile(df, profile_type='category')

        df = df.select(['userid', 'date', 'data'])

        return df

    def get_user_tag(self):

        self.logger.info('[Data Preparation][Nownews News User2Item] Get user tag')

        path_list, base_path = self._get_user_profile_input_path(user_profile_type='tag', save_type='diff')

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        df = self._add_process_date_to_profile(df, profile_type='tag')

        df = df.select(['userid', 'date', 'data'])

        return df

    def get_user_meta_data(self):

        self.logger.info('[Data Preparation][Nownews News User2Item] Get user meta')

        path_list, base_path = self._get_user_profile_input_path(user_profile_type='meta', save_type='snapshot')

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        df = df.select(
            'userid',
            get_json_object(df.data, '$.gender').alias('gender'),
            get_json_object(df.data, '$.age').alias('age')
        )

        df = df.select(['userid', 'age', 'gender'])

        return df

    def _get_user_profile_input_path(self, user_profile_type='category', save_type='diff'):

        input_path = self.CONFIG.USER_PROFILE[user_profile_type].replace('PROJECT_ID', self.project_id).replace('CONTENT_PATH', self.content_type)
        base_path = input_path[:input_path.find('INPUT_DATE')]

        if save_type == 'diff':

            path_list = self._get_existed_blobs(input_path, prefix=f'user_profiles_cassandra/nownews/{user_profile_type}/INPUT_DATE', bucket_idx=2)

        elif save_type == 'snapshot':

            input_path = input_path.replace('INPUT_DATE', self.run_time)
            path_list = [input_path]

        return path_list, base_path

    def _add_process_date_to_profile(self, df, profile_type='category'):
        df = df.withColumn('date', input_file_name())
        df = df.withColumn('date', self._extract_date(f'{profile_type}/', df.date))
        return df
