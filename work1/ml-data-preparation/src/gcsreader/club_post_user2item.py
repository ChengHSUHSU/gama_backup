import json
import logging
import pyspark.sql.functions as f
from pyspark.sql.types import StringType
from datetime import datetime, timedelta
from src.gcsreader import GcsReader
from src.gcsreader.config.club_configs import ClubPostUser2ItemConfig
from utils import *

LOGGER_NAME = '[Data Preparation][beanfun! club user2item]'
HOUR_STRING_FORMAT = '%Y%m%d%H'


class ClubPostUser2ItemReader(GcsReader):

    CONFIG = ClubPostUser2ItemConfig
    PROPERTY = 'beanfun'
    CONTENT_TYPE = 'club_post'

    def __init__(self, project_id, sql_context, content_type, property_name, run_time, days=30, logger=logging):

        self.project_id = project_id
        self.sql_context = sql_context
        self.content_type = content_type
        self.property_name = property_name
        self.days = days
        self.logger = logger

        self.run_time = (datetime.strptime(run_time, DAY_STRING_FORMAT) - timedelta(days=1)).strftime(DAY_STRING_FORMAT)  # training on date (run_time-1)
        self.event_name = self.CONFIG.EVENT_NAME

    def get_event_data(self):

        self.logger.info(f'{LOGGER_NAME} get event data')

        input_path = self.CONFIG.EVENT_PATH \
            .replace('PROPERTY', self.PROPERTY) \
            .replace('PROJECT_ID', self.project_id) \
            .replace('EVENT', self.event_name)

        base_path = input_path[:input_path.find('date=')]

        self.logger.info(f'{LOGGER_NAME} input_path={input_path}')
        self.logger.info(f'{LOGGER_NAME} base_path={base_path}')

        path_list = self._get_existed_blobs(input_path, prefix=f'event_daily/date=INPUT_DATE/property={self.PROPERTY}', bucket_idx=2)

        self.logger.info(f'{LOGGER_NAME} path_list={path_list}')

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        df = self._add_userid(df)

        df = df.select('timestamp', 'date', 'hour', 'userid', f.get_json_object('page_info', '$.uuid').alias('content_id'))

        df = df.withColumn('date', df.date.cast('string')) \
               .withColumn('hour', df.hour.cast('string'))

        return df.select(['userid', 'content_id', 'date', 'hour', 'timestamp'])

    def get_content_data(self):

        self.logger.info(f'{LOGGER_NAME} get content data')

        input_path = self.CONFIG.CONTENT_PATH \
                         .replace('PROJECT_ID', self.project_id) \
                         .replace('CONTENT_TYPE', self.content_type)

        self.logger.info(f'{LOGGER_NAME} content_path={input_path}')

        df_content = self.sql_context.read.parquet(input_path)

        return df_content.select(self.CONFIG.CONTENT_COL)

    def get_user_embedding(self, embedding_key='post', content_type='club_post', user_profile_type='post_embedding'):
        """
        Args:
            embedding_key (str, optional): support [post, news] here , relate to `club_configs.USER_PROFILE` . Defaults to 'post'.
            content_type (str, optional): content type. Defaults to None.
            user_profile_type (str, optional): _description_. Defaults to 'post_embedding'.
        """

        self.logger.info(f'{LOGGER_NAME} get user {embedding_key} embedding')

        path_list, base_path = self._get_user_profile_input_path(user_profile_type=user_profile_type, save_type='diff', content_type=content_type)

        self.logger.info(f'{LOGGER_NAME} user embedding path : {path_list}')

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        df = self._add_process_date_to_profile(df, profile_type='embedding')

        df = df.withColumn('data', f.udf(lambda x: json.loads(x).get(embedding_key, None), StringType())(f.col('data')))

        return df.select(['userid', 'date', 'data'])

    def get_user_category(self):

        self.logger.info(f'{LOGGER_NAME} get user category')

        path_list, base_path = self._get_user_profile_input_path(user_profile_type='category', save_type='diff')

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        df = self._add_process_date_to_profile(df, profile_type='category')

        return df.select(['userid', 'date', 'data'])

    def get_user_meta_data(self, user_profile_type='gamania_meta'):

        user_profile_type_to_column_mapping = {'gamania_meta': self.CONFIG.GAMANIA_META_COL,
                                               'beanfun_meta': self.CONFIG.BEANFUN_META_COL}

        self.logger.info(f'{LOGGER_NAME} get user {user_profile_type}')

        user_meta_col = user_profile_type_to_column_mapping[user_profile_type]

        path_list, base_path = self._get_user_profile_input_path(user_profile_type=user_profile_type, save_type='snapshot')

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        for column_name in user_meta_col:
            if column_name not in ['openid', 'userid']:
                df = df.withColumn(column_name, f.get_json_object(df.data, f'$.{column_name}'))

        return df.select(user_meta_col)

    def get_popularity_data(self):
        """Get hourly popularity score data
        """
        self.logger.info(f'{LOGGER_NAME} get popularity score data')

        all_dates = []
        all_hours = []
        all_path_list = []

        popularity_path = self.CONFIG.METRICS_POPULARITY_PATH.replace('PROJECT_ID', self.project_id).replace('POPULARITY_FOLDER', self.CONFIG.POPULARITY_FOLDER)

        for d in range(int(self.days)):
            current_date = (datetime.strptime(self.run_time, DAY_STRING_FORMAT) - timedelta(days=d)).strftime(DAY_STRING_FORMAT)

            for h in range(24):
                current_hour = (datetime.strptime(str(h), '%H')).strftime('%H')
                prefix = f'metrics/popularity/{self.CONFIG.POPULARITY_FOLDER}/{current_date}/{current_hour}'
                path = popularity_path.replace('INPUT_DATE', current_date).replace('INPUT_HOUR', current_hour)
                cur_path_list, _ = get_partition_path_with_blobs_check(path, prefix, bucket_idx=2)

                if len(cur_path_list) > 0:
                    all_path_list.extend(cur_path_list)
                    all_dates.append(current_date)
                    all_hours.append(current_hour)

        for i, (path, cur_date, cur_hour) in enumerate(zip(all_path_list, all_dates, all_hours)):
            if i == 0:
                df_popular = self.sql_context.read.options(**{'header': 'true', 'escape': '"'}).csv(path)
                df_popular = df_popular.withColumn('date', f.lit(cur_date)).withColumn('hour', f.lit(cur_hour))
            else:
                df = self.sql_context.read.options(**{'header': 'true', 'escape': '"'}).csv(path)
                df = df.withColumn('date', f.lit(cur_date)).withColumn('hour', f.lit(cur_hour))
                df_popular = df_popular.unionByName(df, allowMissingColumns=True)

        return df_popular.select(self.CONFIG.POPULARITY_COLS)

    def _get_user_profile_input_path(self, user_profile_type='category', save_type='diff', content_type=None):

        input_content_type = self.content_type if content_type is None else content_type
        input_path = self.CONFIG.USER_PROFILE[user_profile_type].replace('PROJECT_ID', self.project_id).replace('CONTENT_PATH', input_content_type)
        base_path = input_path[:input_path.find('INPUT_DATE')]

        if save_type == 'diff':

            path_list, _ = get_partition_path_with_check(input_path[:input_path.find('*')], self.run_time, self.days)
            path_list = [f'{p}*.parquet' for p in path_list]

        elif save_type == 'snapshot':

            input_path = input_path.replace('INPUT_DATE', self.run_time)
            path_list = [input_path]

        return path_list, base_path

    def _add_process_date_to_profile(self, df, profile_type='category'):
        df = df.withColumn('date', f.input_file_name())
        df = df.withColumn('date', self._extract_date(f'{profile_type}/', df.date))
        return df

    def _extract_date(self, prefix, data):
        """extract date imformation for user profile data"""
        offset = len(prefix)
        date_offset = len('yyyymmdd')
        return f.udf(lambda x: x[x.find(f'{prefix}')+offset:x.find(f'{prefix}')+offset+date_offset], StringType())(data)
