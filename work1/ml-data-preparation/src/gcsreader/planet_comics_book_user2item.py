import pyspark.sql.functions as f
import pyspark.sql.types as t
from src.gcsreader.config.planet_configs import PlanetComicsBookUser2ItemConfig
from src.gcsreader import GcsReader
from utils import DAY_STRING_FORMAT
from datetime import datetime, timedelta
import logging
import json


def shift_datetime(date, days):
    # shift date forward with positive days input, vice versa
    date = (datetime.strptime(str(date), DAY_STRING_FORMAT) + timedelta(days=days)).strftime(DAY_STRING_FORMAT)
    return date


shift_datetime_udf = f.udf(shift_datetime, t.StringType())


class PlanetComicsBookUser2ItemReader(GcsReader):

    CONFIG = PlanetComicsBookUser2ItemConfig
    PROPERTY = 'beanfun'
    CONTENT_SNAPSHOT_NAME = 'planet_comics_book'
    CONTENT_TYPE = 'comic'

    def __init__(self, project_id, sql_context, run_time, days=30, logger=logging):

        self.project_id = project_id
        self.sql_context = sql_context
        self.days = days

        self.run_time = (datetime.strptime(run_time, DAY_STRING_FORMAT) - timedelta(days=1)).strftime(DAY_STRING_FORMAT)  # training on date (run_time-1)
        self.event_name = self.CONFIG.EVENT_NAME
        self.logger = logger

    def get_event_data(self):

        self.logger.info('[Data Preparation][Planet Comic User2Item] Get event data')

        input_path = self.CONFIG.EVENT_PATH \
            .replace('PROPERTY', self.PROPERTY) \
            .replace('PROJECT_ID', self.project_id) \
            .replace('EVENT', self.event_name)

        base_path = input_path[:input_path.find('date=')]

        self.logger.info(f'[Data Preparation][Planet Comic User2Item] input_path={input_path}')
        self.logger.info(f'[Data Preparation][Planet Comic User2Item] base_path={base_path}')

        path_list = self._get_existed_blobs(input_path, prefix=f'event_daily/date=INPUT_DATE/property={self.PROPERTY}', bucket_idx=2)

        self.logger.info(f'[Data Preparation][Planet Comic User2Item] path_list={path_list}')

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        df = self._add_userid(df)

        df = df.select(
            'timestamp', 'date', 'userid',
            f.get_json_object('page_info', '$.uuid').alias('uuid'),
            f.get_json_object('page_info', '$.page').alias('page'),
        )

        for key, val in self.CONFIG.EVENT_TO_CONDITION_MAPPING.items():
            info_field = key.split('.')
            if len(info_field) == 2:
                df = df.filter(f.col(info_field[1]) == val)

        df = df.withColumn('date', df.date.cast('string'))

        return df.select('userid', 'uuid', 'date', 'timestamp')

    def get_content_data(self):
        self.logger.info('[Data Preparation][Planet Comic User2Item] Get content data')
        input_path = self.CONFIG.CONTENT_PATH \
            .replace('PROJECT_ID', self.project_id) \
            .replace('CONTENT_TYPE', self.CONTENT_SNAPSHOT_NAME)

        df_content = self.sql_context.read.parquet(input_path)
        df_content = df_content.select(self.CONFIG.CONTENT_COL)

        return df_content

    def get_user_profile(self, profile_type):
        # Support category, tag, and embedding profile
        self.logger.info(f'[Data Preparation][Planet Comic User2Item] Get user {profile_type}')
        path_list, base_path = self._get_user_profile_input_path(user_profile_type=profile_type, save_type='diff')
        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)
        df = self._add_process_date_to_profile(df, profile_type=profile_type)
        if profile_type == 'embedding':
            df = df.withColumn('data', f.udf(lambda x: json.loads(x).get('title', None), t.StringType())(f.col('data')))
        df = df.select(['userid', 'date', 'data'])
        return df

    def get_user_meta_data(self):
        self.logger.info('[Data Preparation][Planet Comic User2Item] Get user meta')
        path_list, base_path = self._get_user_profile_input_path(user_profile_type='meta', save_type='snapshot')
        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)
        df = df.select(
            'userid',
            f.get_json_object(df.data, '$.gender').alias('gender'),
            f.get_json_object(df.data, '$.age').alias('age')
        )
        df = df.select(['userid', 'age', 'gender'])
        return df

    def _get_user_profile_input_path(self, user_profile_type='category', save_type='diff'):
        input_path = self.CONFIG.USER_PROFILE[user_profile_type].replace('PROJECT_ID', self.project_id).replace('CONTENT_TYPE', self.CONTENT_TYPE)
        base_path = input_path[:input_path.find('INPUT_DATE')]
        if save_type == 'diff':
            path_list = self._get_existed_blobs(input_path, prefix=f'user_profiles_cassandra/beanfun/planet/{user_profile_type}/INPUT_DATE/{self.CONTENT_TYPE}', bucket_idx=2)

        elif save_type == 'snapshot':
            input_path = input_path.replace('INPUT_DATE', self.run_time)
            path_list = [input_path]
        return path_list, base_path

    def get_content_popular_data(self):
        """Get previous day statistics data from content"""
        self.logger.info('[Data Preparation][Planet Comic User2Item] Get content statistics data')
        input_path = self.CONFIG.CONTENT_POPULARITY_PATH.replace('PROJECT_ID', self.project_id).replace('CONTENT_TYPE', self.CONTENT_SNAPSHOT_NAME)
        base_path = input_path[:input_path.find('date')]
        path_list = self._get_existed_blobs(
            input_path,
            prefix=f'content_daily/property={self.PROPERTY}/content_type={self.CONTENT_SNAPSHOT_NAME}/date=INPUT_DATE',
            bucket_idx=2, days=31)
        df_popular = self.sql_context.read.option('basePath', base_path).parquet(*path_list)
        df_popular = df_popular.select(self.CONFIG.CONTENT_DAILY_POPULAR_COL)

        # Shift date forward a day, make sure the model can't see data from future
        df_popular = df_popular.withColumn('date', shift_datetime_udf(f.col('date'), f.lit(1)))

        return df_popular

    def _add_process_date_to_profile(self, df, profile_type='category'):
        df = df.withColumn('date', f.input_file_name())
        df = df.withColumn('date', self._extract_date(f'{profile_type}/', df.date))
        return df
