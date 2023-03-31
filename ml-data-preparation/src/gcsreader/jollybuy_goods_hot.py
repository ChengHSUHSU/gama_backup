import logging
import pyspark.sql.functions as f
from datetime import datetime, timedelta
from google.cloud import storage
from utils import DAY_STRING_FORMAT, HOUR_STRING_FORMAT, join_df
from src.gcsreader import GcsReader
from src.gcsreader.utils import filter_dataframe, udf_info_json_map_transaction, udf_info_array_of_json_map_transaction
from .config.jollybuy_config import JollybuyGoodsHotItemsConfig
from .config.base import BaseConfig


class JollybuyGoodsHotItemReader(GcsReader):

    CONFIG = {
        'jollybuy_goods': JollybuyGoodsHotItemsConfig
    }

    def __init__(self, project_id, sql_context, input_bucket, content_type, property_name, run_time, days, logger=logging):

        self.project_id = project_id
        self.sql_context = sql_context
        self.input_bucket = input_bucket
        self.content_type = content_type
        self.content_property = property_name
        self.run_time = run_time
        self.days = days
        self.logger = logger

        self.config = self.CONFIG.get(self.content_type, BaseConfig)

    def get_event_data(self):
        bucket_name = self.input_bucket
        daily_event_folder = self.config.EVENT_DAILY_FOLDER
        hourly_event_folder = self.config.EVENT_HOURLY_FOLDER

        file_locations = []
        missing_data = []

        # collect all data in same day folders of input event stream
        # blobs format: {dailyDirPrefix}/{executiveFolderNamePrefix},
        # for example, data/event_stream/2021022523

        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=f'{daily_event_folder}')

        existed_paths = []
        folder_time_string_position = 1 + daily_event_folder.count('/')

        for blob in blobs:
            # for planet only
            if 'planet' in blob.name:
                folder_time_str = blob.name.split('/')[folder_time_string_position]
                existed_path = f'{daily_event_folder}/{folder_time_str}'
                existed_paths.append(existed_path)

        for i in range(int(self.days)):
            day_string = (datetime.strptime(self.run_time, HOUR_STRING_FORMAT) - timedelta(days=i+1)).strftime(DAY_STRING_FORMAT)
            file_location = f'gs://{bucket_name}/{daily_event_folder}/date={day_string}/property={self.content_property}/is_page_view=*/event=*'
            folder = f'{daily_event_folder}/date={day_string}'
            if file_location not in file_locations:
                if folder in existed_paths:
                    file_locations.append(file_location)
                else:
                    if folder not in missing_data:
                        missing_data.append(file_location)

        base_path = f'gs://{bucket_name}/{daily_event_folder}/'
        self.logger.info(f'[popularityCalculator] base_path: {base_path}')
        self.logger.info(f'[popularityCalculator] daily file_location: {file_locations}')
        self.logger.info(f'[popularityCalculator] missing daily location: {missing_data}')

        df_daily = self.sql_context.read.option('basePath', base_path).parquet(*file_locations)
        date_str = datetime.strptime(self.run_time, HOUR_STRING_FORMAT).strftime(DAY_STRING_FORMAT)
        hourly_file_location = f'gs://{bucket_name}/{hourly_event_folder}/date={date_str}/hour=*/property={self.content_property}/is_page_view=*/event=*'

        self.logger.info(f'[popularityCalculator] hourly file_location: {hourly_file_location}')

        try:
            df_hourly = self.sql_context.read.option('basePath', f'gs://{bucket_name}/{hourly_event_folder}/').parquet(hourly_file_location)
            df_all = df_daily.unionByName(df_hourly)
        except:
            self.logger.error(f'[popularityCalculator] no hourly data: {df_daily.count()}')
            return df_daily

        return df_all

    def get_content_data(self, config_key='content_pool'):

        # get content data from content_daily snapshot
        content_path = self.config.CONTENT_PATH.replace('PROJECT_ID', self.project_id)
        self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] content_path={content_path}')

        self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] Get content data')
        df_content = self.sql_context.read.parquet(content_path)

        if config_key:
            self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] Filter content data')
            df_content = filter_dataframe(df_content, config_key, self.config)

            renamed_cols = getattr(self.config, 'RENAMED_COLS', {}).get(config_key, {})
            if renamed_cols:
                for old_col, new_col in renamed_cols.items():
                    df_content = df_content.withColumnRenamed(old_col, new_col)

        self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] Content data: {df_content.count()}')

        return df_content

    def get_candidates(self, df, config_key='input_data_pool'):

        # get positive data from parsed df by event
        if config_key:
            self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] Filter candidate data')
            df_candidates = filter_dataframe(df, config_key, self.config)
        else:
            df_candidates = df

        self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] Candidates data: {df_candidates.count()}')
        df_candidates.limit(5).show()

        return df_candidates

    def get_impression_cnts(self, df, config_key='impression', cnt_col_suffix='_impression_cnt', unified_id_col='uuid', time_session_col='hour', date_col='date'):

        i = 0
        count_filter_conditions = getattr(self.config, 'COUNT_FILTER_CONDITIONS', {})

        for event, (target_info, filter_info, id_col, filter_key, filter_value_list, _) in count_filter_conditions.get(config_key, {}).items():
            df_info = df.filter(f.col('event') == event)
            df_info = df_info.withColumn(id_col,
                                         udf_info_array_of_json_map_transaction(id_col, filter_key, filter_value_list)(f.col(target_info), f.col(filter_info)))

            df_info = df_info.select(f.explode(f.col(id_col)).alias(unified_id_col), time_session_col, date_col)

            df_info = df_info.dropna(subset=[unified_id_col])
            df_info = df_info.filter(f"{unified_id_col} != ''")
            df_info = df_info.groupBy(unified_id_col, time_session_col, date_col).count().select(f.col(unified_id_col),
                                                                                                 f.col(time_session_col),
                                                                                                 f.col(date_col),
                                                                                                 f.col('count').alias(event + cnt_col_suffix))

            if i == 0:
                df_impression = df_info
            else:
                df_impression = join_df(df_impression, df_info, on=[unified_id_col, time_session_col, date_col], how='left')

            i += 1

        if i > 0:
            df_impression = df_impression.fillna(0)

            self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] process {config_key} count data')

            df_impression.cache()
            df_impression.limit(5).show(5, False)
            self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] {config_key} count data: {df_impression.count()}')

            return df_impression
        else:
            self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] No valid {config_key} count data')
            return None

    def get_click_cnts(self, df, config_key='click', cnt_col_suffix='_click_cnt', unified_id_col='uuid', time_session_col='hour', date_col='date'):

        i = 0
        count_filter_conditions = getattr(self.config, 'COUNT_FILTER_CONDITIONS', {})

        for event, (target_info, filter_info, id_col, filter_key, filter_value_list, is_positive_forward) in count_filter_conditions.get(config_key, {}).items():

            df_info = df.filter(f.col('event') == event)

            if is_positive_forward:
                df_info = df_info.withColumn(id_col,
                                             udf_info_json_map_transaction(id_col, filter_key, filter_value_list)(f.col(target_info), f.col(filter_info))) \
                    .withColumnRenamed(id_col, unified_id_col)
            else:
                df_info = df_info.withColumn(filter_key, f.get_json_object(f.col(filter_info), f'$.{filter_key}')) \
                                 .filter(~f.col(filter_key).isin(filter_value_list)) \
                                 .withColumn(unified_id_col, f.get_json_object(f.col(target_info), f'$.{id_col}'))

            df_info = df_info.dropna(subset=[unified_id_col])
            df_info = df_info.filter(f"{unified_id_col} != ''")

            df_info = df_info.select(unified_id_col, time_session_col, date_col)
            df_info = df_info.groupBy([unified_id_col, time_session_col, date_col]).count().select(f.col(unified_id_col),
                                                                                                   f.col(time_session_col),
                                                                                                   f.col(date_col),
                                                                                                   f.col('count').alias(event + cnt_col_suffix))

            if i == 0:
                df_merge = df_info
            else:
                df_merge = join_df(df_merge, df_info, on=[unified_id_col, time_session_col, date_col], how='left')

            i += 1

        if i > 0:
            df_merge = df_merge.fillna(0)

            self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] process {config_key} count data')

            df_merge.cache()

            df_merge.limit(5).show(5, False)
            self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] {config_key} count data: {df_merge.count()}')

            return df_merge
        else:
            self.logger.info(f'[Data Preparation][Jollybuy Goods Hot Items] No valid {config_key} count data')
            return None

    def get_booster_cnts(self, df, config_key='booster', cnt_col_suffix='_booster_cnt', unified_id_col='uuid', time_session_col='hour', date_col='date'):

        df_booster = self.get_click_cnts(df, config_key, cnt_col_suffix, unified_id_col, time_session_col, date_col)

        return df_booster

    def get_reward_cnts(self, df, config_key='reward', cnt_col_suffix='_reward_cnt', unified_id_col='uuid', time_session_col='hour', reward_col='reward', date_col='date'):

        df_reward = self.get_click_cnts(df, config_key, cnt_col_suffix, unified_id_col, time_session_col, date_col)

        count_filter_conditions = getattr(self.config, 'COUNT_FILTER_CONDITIONS', {})
        reward_keys = [key + cnt_col_suffix for key in count_filter_conditions.get(reward_col, {}).keys()]

        df_reward = df_reward.withColumn(reward_col, sum(f.col(key) for key in reward_keys)) \
                             .select([unified_id_col, time_session_col, date_col, reward_col])

        return df_reward
