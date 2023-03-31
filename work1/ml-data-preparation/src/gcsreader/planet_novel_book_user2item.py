import pyspark.sql.functions as f
import pyspark.sql.types as t
import json
import logging
from datetime import datetime, timedelta
from src.gcsreader import GcsReader
from src.gcsreader.config.planet_configs import PlanetNovelBookUser2ItemConfig
from utils import DAY_STRING_FORMAT


class PlanetNovelBookUser2ItemReader(GcsReader):

    CONFIG = PlanetNovelBookUser2ItemConfig
    PROPERTY = 'beanfun'
    CONTENT_SNAPSHOT_NAME = 'planet_novel_book'
    CONTENT_TYPE = 'novel'

    def __init__(self, project_id, sql_context, run_time, days=30, logger=logging):

        self.project_id = project_id
        self.sql_context = sql_context
        self.days = days

        self.run_time = (datetime.strptime(run_time, DAY_STRING_FORMAT) - timedelta(days=1)).strftime(DAY_STRING_FORMAT)  # training on date (run_time-1)
        self.event_name = self.CONFIG.EVENT_NAME
        self.logger = logger

    def get_event_data(self):

        self.logger.info('[Data Preparation][Planet Novel User2Item] Get event data')

        input_path = self.CONFIG.EVENT_PATH \
            .replace('PROPERTY', self.PROPERTY) \
            .replace('PROJECT_ID', self.project_id) \
            .replace('EVENT', self.event_name)

        base_path = input_path[:input_path.find('date=')]

        self.logger.info(f'[Data Preparation][Planet Novel User2Item] input_path={input_path}')
        self.logger.info(f'[Data Preparation][Planet Novel User2Item] base_path={base_path}')

        path_list = self._get_existed_blobs(input_path, prefix=f'event_daily/date=INPUT_DATE/property={self.PROPERTY}', bucket_idx=2)

        self.logger.info(f'[Data Preparation][Planet Novel User2Item] path_list={path_list}')

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
        self.logger.info('[Data Preparation][Planet Novel User2Item] Get content data')
        input_path = self.CONFIG.CONTENT_PATH \
            .replace('PROPERTY', self.PROPERTY) \
            .replace('PROJECT_ID', self.project_id) \
            .replace('CONTENT_TYPE', self.CONTENT_SNAPSHOT_NAME)

        df_content = self.sql_context.read.parquet(input_path)
        df_content = df_content.select(self.CONFIG.CONTENT_COL)

        return df_content

    def get_user_profile(self, profile_type):
        # Support category, tag, and embedding profile
        self.logger.info(f'[Data Preparation][Planet Novel User2Item] Get user {profile_type}')
        path_list, base_path = self._get_user_profile_input_path(user_profile_type=profile_type, save_type='diff')
        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)
        df = self._add_process_date_to_profile(df, profile_type=profile_type)
        if profile_type == 'embedding':
            df = df.withColumn('user_title_embedding', f.udf(lambda x: json.loads(x).get('title', None), t.StringType())(f.col('data')))
            df = df.withColumn('user_tag_embedding', f.udf(lambda x: json.loads(x).get('tag', None), t.StringType())(f.col('data')))
            df = df.select(['userid', 'user_title_embedding', 'user_tag_embedding', 'date'])
        else:
            df = df.select(['userid', 'data', 'date'])
        return df

    def get_user_meta_data(self):
        self.logger.info('[Data Preparation][Planet Novel User2Item] Get user meta')
        path_list, base_path = self._get_user_profile_input_path(user_profile_type='meta', save_type='snapshot')
        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)
        df = df.select(
            'userid',
            f.get_json_object(df.data, '$.age').alias('age'),
            f.get_json_object(df.data, '$.gender').alias('gender')
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

    def _add_process_date_to_profile(self, df, profile_type='category'):
        df = df.withColumn('date', f.input_file_name())
        df = df.withColumn('date', self._extract_date(f'{profile_type}/', df.date))
        return df
