# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'datapreparation.zip')

import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from datetime import datetime, timedelta
from pyspark.sql.types import StringType, FloatType
from src.options.graph_embedding_model_options import GraphEmbeddingModelOptions
from src.gcsreader.config.graph_embedding_model_config import GraphEmbeddingModelConfig
from src.gcsreader.graph_embedding_model_aggregator import GraphEmbeddingModelAggregator
from src.gcsreader import dedup_by_col
from utils import initial_spark, dump_pickle, upload_to_gcs, HOUR_STRING_FORMAT, DAY_STRING_FORMAT
from utils.logger import logger


def merge_popularity_to_content(df_content: DataFrame, df_popularity: DataFrame) -> DataFrame:
    """Function to process and merge popularity data to content data

    Args:
        df_content (DataFrame): content data
        df_popularity (DataFrame): popularity data

    Returns:
        DataFrame
    """

    df_popularity = df_popularity.withColumn('update_time',
                                             f.udf(lambda date, hour: f'{date}{hour}', StringType())
                                             (f.col('date'), f.col(f'hour')))

    df_popularity = df_popularity.withColumn('update_time',
                                             f.udf(lambda x: datetime.strptime(x, HOUR_STRING_FORMAT).timestamp()*1000, FloatType())
                                             (f.col('update_time')))

    df_popularity = df_popularity.withColumnRenamed('uuid', 'content_id')
    df_popularity = dedup_by_col(df_popularity, unique_col_base=['content_id'], time_col='update_time')
    df_content = df_content.join(df_popularity.select(['content_id', 'popularity_score']), on=['content_id'], how='left')

    return df_content


def get_raw_data(configs=GraphEmbeddingModelConfig) -> dict:
    """Function to get raw data from gcs

    Args:
        configs (GraphEmbeddingModelConfig): preparation configs. Defaults to GraphEmbeddingModelConfig.

    Returns:
        dict
    """

    process_content_types = configs.PROCESS_CONTENT_TYPES
    process_profile_types = configs.PROCESS_PROFILE_TYPES
    process_metrics_types = configs.PROCESS_METRICS_TYPES
    content_type_to_property_type = configs.CONTENT_TYPE_TO_PROPERTY_TYPE
    raw_data = {'event': None, 'content': None, 'user_category': None}

    for content_type in process_content_types:

        logger.info(f'process {content_type}')
        content_property = content_type_to_property_type.get(content_type, 'beanfun')

        # process gcs data
        aggregator.read(content_property, content_type, process_profile_types, process_metrics_types)
        data = aggregator.get()

        df_event, df_content, df_popularity, df_user_profile = \
            data['event'], data['content'], data['metrics'], data['user_profile']

        # post process
        df_category = df_user_profile['category']
        df_content = merge_popularity_to_content(df_content, df_popularity['popularity'])     # merge content and popularity

        df_event = df_event.withColumn('content_type', f.lit(content_type))
        df_content = df_content.withColumn('content_type', f.lit(content_type))
        df_category = df_category.withColumn('content_type', f.lit(content_type))

        # union different content_type dataset
        raw_data['event'] = df_event if raw_data['event'] is None else raw_data['event'].unionByName(df_event)
        raw_data['content'] = df_content if raw_data['content'] is None else raw_data['content'].unionByName(df_content, allowMissingColumns=True)
        raw_data['user_category'] = df_category if raw_data['user_category'] is None else raw_data['user_category'].unionByName(df_category)

    return raw_data


if __name__ == '__main__':

    opt = GraphEmbeddingModelOptions().parse()

    project_id = opt.project_id
    run_time = (datetime.strptime(opt.run_time, DAY_STRING_FORMAT) - timedelta(days=1)).strftime(DAY_STRING_FORMAT)
    days = int(opt.days)
    base_path = f'{opt.checkpoints_dir}/{opt.experiment_name}'

    sql_context, spark_session = initial_spark(cores='5', memory='27G', overhead='5G', shuffle_partitions='1000', num_executors='3')
    sql_context.sql('SET spark.sql.autoBroadcastJoinThreshold = -1')  # disable broadcast join

    aggregator = GraphEmbeddingModelAggregator(project_id, sql_context, run_time, days=days, config=GraphEmbeddingModelConfig, logger=logger)
    raw_data = get_raw_data(configs=GraphEmbeddingModelConfig)

    for data_type, spark_dataframe in raw_data.items():
        dataset = spark_dataframe.toPandas()
        file_name = f'{data_type}_data'
        dump_pickle(file_name, dataset, base_path=base_path)
        logger.info(f'{data_type} dataset length: {len(dataset)}')
        logger.info(f'dump {data_type} dataset success')

    # Upload checkpoint, dataset to GCS
    if opt.upload_gcs:
        upload_to_gcs(
            base_path,
            f'machine-learning-models-{project_id}',
            f'dataset/graph_embedding_model/{run_time}'
        )
