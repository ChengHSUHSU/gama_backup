from pyspark.sql.functions import *
from pyspark.sql.types import StringType, MapType
from datetime import datetime, timedelta
import json

from src.gcsreader.config.planet_configs import PlanetVideoUser2ItemConfig
from utils import get_partition_path_with_check
from src.gcsreader import GcsReader


class PlanetVideoUser2ItemReader(GcsReader):

    CONFIG = PlanetVideoUser2ItemConfig

    def __init__(self, project_id, sql_context, content_type, property_name, run_time, days=30):

        self.project_id = project_id
        self.sql_context = sql_context
        self.content_type = content_type
        self.property_name = property_name
        self.days = days

        self.run_time = (datetime.strptime(run_time, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')  # training on date (run_time-1)
        self.event_name = self.CONFIG.EVENT_NAME

    def get_event_data(self, days='30'):

        df = None
        for event in self.event_name:
            input_path = self.CONFIG.EVENT_PATH \
                .replace('PROJECT_ID', self.project_id) \
                .replace('PROPERTY', self.property_name) \
                .replace('EVENT', event)

            input_path = input_path.replace('is_page_view=*', 'is_page_view=True') \
                if 'page_view' in event else input_path.replace('is_page_view=*', 'is_page_view=False')

            base_path = input_path[:input_path.find('date=')]
            path_list, _ = get_partition_path_with_check(input_path[:input_path.find('*')], self.run_time, days)
            path_list = [p + '*.parquet' for p in path_list]

            df_temp = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

            if event.split('_')[-1] == 'view':
                info_prefix = 'page'
            elif event.split('_')[-1] == 'click':
                info_prefix = 'click'

            df_temp = df_temp.withColumn('page_info_map', from_json(col('page_info'), MapType(StringType(),StringType())))
            df_temp = df_temp.withColumn('click_info_map', from_json(col('click_info'), MapType(StringType(),StringType())))

            df_temp = df_temp.select(
                'openid',
                get_json_object(f'{info_prefix}_info', '$.uuid').alias('uuid'),
                get_json_object(f'{info_prefix}_info', '$.name').alias('name'),
                'page_info_map',
                'click_info_map',
                'date',
                'timestamp'
            )

            for cond in self.CONFIG.EVENT_OF_CONTENT_TYPE_CONDITION:
                if cond['event'] == event:
                    event_cond = cond

            for key in event_cond:
                if key == 'event':
                    continue
                info_field = key.split('.')
                df_temp = df_temp.filter(col(info_field[0]).getItem(info_field[1]) == event_cond[key])

            df_temp = df_temp.select('openid', 'uuid', 'name', 'date', 'timestamp')
            if df is None:
                df = df_temp
            else:
                df = df.union(df_temp)

        df = df.withColumn('date', df.date.cast('string'))

        return df


    def get_content_data(self):
        input_path = self.CONFIG.CONTENT_PATH \
                .replace('PROJECT_ID', self.project_id) \
                .replace('CONTENT_TYPE', f'planet_{self.content_type}')

        df_content = self.sql_context.read.parquet(input_path)
        df_content = df_content.select(self.CONFIG.CONTENT_COL)

        return df_content


    def get_user_title_embedding(self, content_type='video'):
        path_list, base_path = self._get_user_profile_input_path(user_profile_type='title_embedding', save_type='diff', content_type=content_type)

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)
        df = self._add_process_date_to_profile(df, profile_type='embedding')
        df = df.withColumn('data', udf(lambda x: json.loads(x).get('title', None), StringType())(col('data')))
        df = df.select(['userid', 'date', 'data'])

        return df


    def get_user_category(self, content_type='video'):
        path_list, base_path = self._get_user_profile_input_path(user_profile_type='category', save_type='diff', content_type=content_type)

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)
        df = self._add_process_date_to_profile(df, profile_type='category')
        df = df.select(['userid', 'date', 'data'])

        return df


    def get_user_tag(self, content_type='video'):
        path_list, base_path = self._get_user_profile_input_path(user_profile_type='tag', save_type='diff', content_type=content_type)

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)
        df = self._add_process_date_to_profile(df, profile_type='tag')
        df = df.select(['userid', 'date', 'data'])

        return df


    def get_user_meta_data(self):
        path_list, base_path = self._get_user_profile_input_path(user_profile_type='meta', save_type='snapshot')

        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)
        df = df.select(
            'userid',
            get_json_object(df.data, '$.gender').alias('gender'),
            get_json_object(df.data, '$.age').alias('age')
        )

        df = df.select(['userid', 'age', 'gender'])

        return df


    def _get_user_profile_input_path(self, user_profile_type='category', save_type='diff', content_type='video'):
        input_path = self.CONFIG.USER_PROFILE[user_profile_type].replace('PROJECT_ID', self.project_id).replace('CONTENT_PATH', content_type)
        base_path = input_path[:input_path.find('INPUT_DATE')]

        if save_type == 'diff':
            path_list, _ = get_partition_path_with_check(input_path[:input_path.find('*')], self.run_time, self.days)
            path_list = [p + '*.parquet' for p in path_list]
        elif save_type == 'snapshot':
            input_path = input_path.replace('INPUT_DATE', self.run_time)
            path_list = [input_path]

        return path_list, base_path


    def _add_process_date_to_profile(self, df, profile_type='category'):
        df = df.withColumn('date', input_file_name())
        df = df.withColumn('date', self._extract_date(f'{profile_type}/', df.date))

        return df


    def _extract_date(self, prefix, data):
        """extract date imformation for user profile data"""
        offset = len(prefix)
        date_offset = len('yyyymmdd')

        return udf(lambda x: x[x.find(f'{prefix}')+offset:x.find(f'{prefix}')+offset+date_offset], StringType())(data)

