import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
from src.gcsreader.base_reader import GcsReader
from src.gcsreader.utils import rename_cols, extend_generated_time
from src.gcsreader.udf import udf_set_user_profile_join_date
from utils import get_partition_path_with_check
import logging
import json


class UserProfileReader(GcsReader):

    def __init__(self, project_id, sql_context, run_time, days=30, config=None, logger=logging):
        super().__init__(project_id, sql_context, run_time, days, config, logger)
        self.data = dict()

    def process(self, content_type: str, profile_type: str = 'category', user_profile_run_time: str = None):
        """Function to process user profile reader pipeline

        Args:
            content_type (str)
            profile_type (str, optional) Defaults to 'category'.
            user_profile_run_time (str, optional):
                user profile might use another time period different from self.run_time
                if it is `None`, use self.run_time. Defaults to None.
        """
        path_profile_type = 'embedding' if 'embedding' in profile_type else profile_type
        self.user_profile_run_time = user_profile_run_time if user_profile_run_time else self.run_time
        self.logger.info(f'[Data Preparation][UserProfile] set user profile run time: {self.user_profile_run_time}')

        self.logger.info(f'[Data Preparation][UserProfile] get user {content_type} {profile_type} profile')
        path_list, base_path = self._get_existed_user_profile_path(content_type=content_type, user_profile_type=path_profile_type)

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)
        df = self._parse_profile(df, profile_type)
        self.logger.info('[Data Preparation][UserProfile] extend generated time')
        df = extend_generated_time(df, prefix_name=path_profile_type, time_format='date')

        if self.config.USER_PROFILE_CONDITIONS[content_type]['save_type'] == 'diff':
            self.logger.info(f'[Data Preparation][UserProfile] set {self.config.USER_PROFILE_JOIN_DAY_DIFF} day range before event')
            df = df.withColumn('date', udf_set_user_profile_join_date(day_ranges=self.config.USER_PROFILE_JOIN_DAY_DIFF)(f.col('date')))

        self.logger.info('[Data Preparation][UserProfile] rename columns')
        df = rename_cols(df, rename_cols=self.config.STEPS_TO_RENAMCOLS.get('user_profile', {}).get(profile_type, []))

        self.logger.info('[Data Preparation][UserProfile] select requisite columns')
        self.data[profile_type] = df.select(self.config.REQUISITE_COLS.get('user_profile', {}).get(profile_type, '*'))

    def _get_existed_user_profile_path(self, content_type: str, user_profile_type: str = 'category') -> tuple:
        """Private function to get the existing user profile path

        Args:
            content_type (str)
            user_profile_type (str, optional) Defaults to 'category'.

        Returns:
            tuple
        """
        user_profile_conditions = self.config.USER_PROFILE_CONDITIONS[content_type]
        user_profile_path = user_profile_conditions['user_profile_path']
        blob_path = user_profile_conditions['blob_path']
        save_type = user_profile_conditions['save_type']

        input_path = user_profile_path \
            .replace('PROJECT_ID', self.project_id) \
            .replace('BLOB_PATH', blob_path) \
            .replace('PROFILE_TYPE', user_profile_type)

        base_path = input_path[:input_path.find('INPUT_DATE')]

        if save_type == 'diff':

            path_list, _ = get_partition_path_with_check(input_path[:input_path.find('*')], self.user_profile_run_time, self.days)
            path_list = [f'{p}*.parquet' for p in path_list]

        elif save_type == 'snapshot':

            input_path = input_path.replace('INPUT_DATE', self.user_profile_run_time)
            path_list = [input_path]

        self.logger.info(f'existed user profile path:\n {path_list}')
        return path_list, base_path

    def _parse_profile(self, df: DataFrame, profile_type: str) -> DataFrame:
        if profile_type == 'meta':
            for meta_column in self.config.META_COLS:
                df = df.withColumn(meta_column[0], f.get_json_object(f.col(meta_column[1]), meta_column[2]))

        elif 'embedding' in profile_type:
            embedding_type = profile_type.split('_')[0]
            if embedding_type not in ['title', 'tag', 'post']:
                raise ValueError(f'{embedding_type} embedding type not support')

            df = df.withColumn('data', f.udf(lambda x: json.loads(x).get(embedding_type, None), StringType())(f.col('data')))

        return df
