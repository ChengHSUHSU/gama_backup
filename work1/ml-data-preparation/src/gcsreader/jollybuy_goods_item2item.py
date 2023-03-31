import logging
import json
import pyspark.sql.functions as f
from pyspark.sql.types import MapType, StringType, FloatType
from datetime import datetime, timedelta
from utils import DAY_STRING_FORMAT, get_partition_path_with_blobs_check, aggregate_view_also_view_udf
from src.gcsreader import GcsReader
from src.gcsreader.utils import filter_dataframe
from src.negative_sampler import generate_negative_samples
from .config.jollybuy_config import JollybuyGoodsItem2ItemConfig
from .config.base import BaseConfig


class JollybuyGoodsItem2ItemReader(GcsReader):

    CONFIG = {
        'jollybuy_goods': JollybuyGoodsItem2ItemConfig
    }

    def __init__(self, project_id, sql_context, spark_session, content_type, property_name, run_time, days, logger=logging):

        self.project_id = project_id
        self.spark_session = spark_session
        self.sql_context = sql_context
        self.content_type = content_type
        self.content_property = property_name
        self.run_time = run_time
        self.days = days
        self.logger = logger

        self.config = self.CONFIG.get(self.content_type, BaseConfig)

    def get_event_data(self, config_key=''):

        all_path_list = []
        event_path = self.config.EVENT_PATH.replace('PROJECT_ID', self.project_id)
        event_base_path = event_path[:event_path.find('date=')]

        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] event_path={event_path}')
        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] base_path={event_base_path}')

        for d in range(int(self.days)):
            current_date = (datetime.strptime(self.run_time, DAY_STRING_FORMAT) - timedelta(days=d)).strftime(DAY_STRING_FORMAT)
            prefix = f'event_daily/date={current_date}/property={self.content_property}'
            path = event_path.replace('INPUT_DATE', current_date)

            cur_path_list, _ = get_partition_path_with_blobs_check(path, prefix, bucket_idx=2)
            all_path_list.extend(cur_path_list)

        self.logger.info('[Data Preparation][Jollybuy Goods Item2Item] All exist paths')
        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] path_list={all_path_list}')

        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Get event data')
        df_event = self.sql_context.read.option("basePath", event_base_path).parquet(*all_path_list)

        return df_event

    def get_content_data(self, config_key=''):

        # get content data from content_daily snapshot
        content_path = self.config.CONTENT_PATH.replace('PROJECT_ID', self.project_id)
        content_base_path = content_path[:content_path.find('property=')]

        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] content_path={content_path}')
        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] content_base_path={content_base_path}')

        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Get content data')
        df_content = self.sql_context.read.parquet(content_path)

        if config_key:
            self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Filter content data')
            df_content = filter_dataframe(df_content, config_key, self.config)

            renamed_cols = getattr(self.config, 'RENAMED_COLS', {}).get(config_key, {})
            if renamed_cols:
                for old_col, new_col in renamed_cols.items():
                    df_content = df_content.withColumnRenamed(old_col, new_col)

        # current content daily data contains data with null title_embedding
        df_content = df_content.filter(f.col('title_embedding').isNotNull())

        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Content data: {df_content.count()}')

        return df_content

    def get_view_also_view_data(self, config_key=''):

        all_dates = []
        all_path_list = []

        # get view-also-view scores
        view_also_view_path = self.config.VIEW_ALSO_VIEW_PATH.replace('PROJECT_ID', self.project_id) \
                                                             .replace('CONTENT_TYPE', self.content_type)

        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] view_also_view_path={view_also_view_path}')
        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Get view_also_view data')

        for d in range(1, int(self.days) + 1):
            current_date = (datetime.strptime(self.run_time, DAY_STRING_FORMAT) - timedelta(days=d)).strftime(DAY_STRING_FORMAT)
            prefix = f'metrics/view_also_view/{self.content_type}/{current_date}'
            path = view_also_view_path.replace('INPUT_DATE', current_date)

            cur_path_list, _ = get_partition_path_with_blobs_check(path, prefix, bucket_idx=2)

            if len(cur_path_list) != 0:
                all_path_list.extend(cur_path_list)
                all_dates.append(current_date)

        self.logger.info('[Data Preparation][Jollybuy Goods Item2Item] All exist paths')
        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] view_also_view path_list={all_path_list}')
        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] view_also_view dates_list={all_dates}')

        for i, (path, cur_date) in enumerate(zip(all_path_list, all_dates)):
            if i == 0:
                df_view_also_view = self.sql_context.read.options(**{'header': 'true', 'escape': '"'}).csv(path)
                df_view_also_view = df_view_also_view.withColumn('date', f.lit(cur_date))
            else:
                df_view_also_view_tmp = self.sql_context.read.options(**{'header': 'true', 'escape': '"'}).csv(path)
                df_view_also_view_tmp = df_view_also_view_tmp.withColumn('date', f.lit(cur_date))

                df_view_also_view = df_view_also_view.unionByName(df_view_also_view_tmp)

        if config_key:
            df_view_also_view = filter_dataframe(df_view_also_view, config_key, self.config)

            renamed_cols = getattr(self.config, 'RENAMED_COLS', {}).get(config_key, {})
            if renamed_cols:
                for old_col, new_col in renamed_cols.items():
                    df_view_also_view = df_view_also_view.withColumnRenamed(old_col, new_col)

        df_view_also_view = df_view_also_view.withColumn('view_also_view_json',
                                                         f.udf(lambda x: json.loads(x), MapType(StringType(), FloatType(), False))(f.col('view_also_view_json')))
        df_view_also_view = df_view_also_view.groupby('uuid').agg(f.collect_list(f.struct(f.col('view_also_view_json'),
                                                                                          f.col('date'))).alias('view_also_view_json'))
        df_view_also_view = df_view_also_view.withColumn('view_also_view_json', aggregate_view_also_view_udf()(f.struct(f.col('uuid'), f.col('view_also_view_json'))))

        df_view_also_view.cache()
        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] View-Also-View Data: {df_view_also_view.count()}')
        df_view_also_view.limit(5).show()

        return df_view_also_view

    def get_candidates(self, df, config_key='', candidate_sample_ratio=1):

        # get positive data from parsed df by event
        if config_key:
            self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Filter candidate data')
            df_candidates = filter_dataframe(df, config_key, self.config)
        else:
            df_candidates = df

        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Candidates data: {df_candidates.count()}')
        df_candidates.limit(5).show()

        if candidate_sample_ratio < 1:
            df_candidates = df_candidates.sample(fraction=candidate_sample_ratio)
            self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Candidates data after sampling: {df_candidates.count()}')

        return df_candidates

    def get_positive_data(self, df, config_key=''):

        # get positive data from parsed df by event
        if config_key:
            self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Get all positive data')
            df_positive = filter_dataframe(df, config_key, self.config)
        else:
            df_positive = df

        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Positive data: {df_positive.count()}')
        df_positive.limit(5).show()

        return df_positive

    def get_negative_data(self, df_candidates_pool, df_positive_pool, neg_sample_size, major_col, candidate_col, time_col, prefix=''):

        # generate negative samples from candidate pool
        # we'll need to filter out the combination that overlap with positive pairs
        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Generate negative samples from all candidates')
        df_candidates_pd = df_candidates_pool.select('uuid', 'date', 'timestamp').toPandas()
        df_negative = generate_negative_samples(df_candidates_pd, sample_size=neg_sample_size, major_col=major_col,
                                                candidate_col=candidate_col, time_col=time_col, prefix=prefix)
        df_negative = self.spark_session.createDataFrame(df_negative)

        # filter out positive samples
        # we also take swap of page uuid and click uuid into account
        df_positive_pairs = df_positive_pool.select([major_col, prefix + candidate_col]) \
            .unionByName(df_positive_pool.select(f.col(prefix + candidate_col).alias(major_col),
                                                 f.col(major_col).alias(prefix + candidate_col)))
        df_negative = df_negative.subtract(df_positive_pairs)

        self.logger.info(f'[Data Preparation][Jollybuy Goods Item2Item] Negative samples: {df_negative.count()}')

        return df_negative
