import sys
sys.path.insert(0, 'datapreparation.zip')

from datetime import datetime
from utils import HOUR_STRING_FORMAT
import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from utils.logger import Logger
from utils.parser import ContentDistinctValueParser
from src.gcsreader.utils import udf_get_candidate_arms
from utils import initial_spark, dump_pickle, upload_to_gcs

from src.options.planet_news_options import NewsHotOptions
from src.gcsreader.config.planet_news.planet_news_hot_model import PlanetNewsHotModelConfig
from src.gcsreader.planet_news_hot_model_aggregator import PlanetNewsHotModelAggregator


def get_raw_data(configs: PlanetNewsHotModelConfig,
                 aggregator: PlanetNewsHotModelAggregator,
                 opt: NewsHotOptions) -> dict:
    """  Args:
                configs: PlanetNewsHotModelConfig.
                aggregator: PlanetNewsHotModelAggregator.
                content_type: str.
                content_property: str.
        Returns:
                raw_data: dict. it includes event_count, statistics_count, context_data
    """
    process_metrics_types = configs.PROCESS_METRICS_TYPES

    aggregator.read(opt.content_property, opt.content_type, process_metrics_types)
    data = aggregator.get()

    df_event, df_content, df_metrics = data['event'], data['content'], data['metrics']
    df_statistics = df_metrics['snapshot_statistics']

    # build event count for each content_id
    df_event_count = aggregator.get_event_count(df_event, dropDuplicates=True)

    # build reward column
    df_reward = aggregator.get_reward_cnts(df_event_count)

    # build df_context_data
    df_context_data = df_content.select([f.col('content_id'), f.col('cat0'), f.col('cat1')])

    raw_data = {
        'user_event': df_event,
        'item_content': df_content,
        'df_reward': df_reward,
        'df_event_count': df_event_count,
        'df_context_data': df_context_data,
        'df_statistics_cnt': df_statistics
    }
    return raw_data


def join_raw_data(raw_data: dict, aggregator: PlanetNewsHotModelAggregator, opt: NewsHotOptions) -> DataFrame:
    """  Args:
                raw_data: dict, it collects raw data.
                aggregator: PlanetNewsHotModelAggregator
                opt: NewsHotOptions
        Returns:
                df_data: DataFrame, (columns: content_id, date, event_count_col, statistics_col)
    """
    df_reward = raw_data['df_reward']
    df_event_count = raw_data['df_event_count']
    df_context_data = raw_data['df_context_data']
    df_statistics_cnt = raw_data['df_statistics_cnt']

    df_data = df_reward.join(df_event_count, on=['date', 'hour', 'content_id'], how='left')
    df_data = df_data.join(df_statistics_cnt, on=['content_id'], how='left')
    df_data = df_data.join(df_context_data, on=['content_id'], how='left')
    df_data = df_data.na.drop(subset=['content_id']) 

    # get negative samples
    df_negative_candidates = aggregator.get_negative_candidates(df_data)

    # left join df_negative_candidates
    df_data = df_data.join(df_negative_candidates, on=['date', 'hour'], how='left')

    # add column content_id_neg
    df_data = df_data.withColumn('content_id_neg',
                            udf_get_candidate_arms(int(opt.negative_sample_size),
                                                    neg_candidate_col='content_id_neg',
                                                    pos_id_col='content_id')(f.struct(f.col('content_id'),
                                                                             f.col('content_id_neg'))))
    df_data = df_data.sort(f.col('reward').desc())
    return df_data


if __name__ == '__main__':
    # config, option
    config = PlanetNewsHotModelConfig
    opt = NewsHotOptions().parse()
    project_id = opt.project_id
    run_time = opt.run_time
    days = int(opt.days)
    opt.input_bucket = 'event-bf-data-prod-001'

    # run_time
    if run_time == '':
        run_time = datetime.utcnow().strftime(HOUR_STRING_FORMAT)

    base_path = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    logger = Logger(logger_name=opt.logger_name, dev_mode=True)

    sql_context, spark_session = initial_spark(cores='5', memory='27G',
                                               overhead='5G', shuffle_partitions='1000',
                                               num_executors='3')
    sql_context.sql("SET spark.sql.autoBroadcastJoinThreshold = -1")  # disable broadcast join

    # get raw data
    logger.info(f'[Data Preparation][Planet News Hot Item] get raw data from gcs.')
    aggregator = PlanetNewsHotModelAggregator(project_id, sql_context, run_time, days, config=config, logger=logger)
    raw_data = get_raw_data(configs=config, aggregator=aggregator, opt=opt)

    # join raw data
    logger.info(f'[Data Preparation][Planet News Hot Item] join raw data.')
    df_data = join_raw_data(raw_data, aggregator, opt)
    df_data.show(10, False)
    df_pd = df_data.toPandas()

    # collect all distinct cat0, cat1, tags and store_id
    logger.info(f'[Data Preparation][Planet News Hot Item] build col2label.')
    distinct_content_parser = ContentDistinctValueParser()
    col2label = distinct_content_parser.parse(raw_data['item_content'], config.COLS_TO_ENCODE['content'], add_ner=False)

    # user event data
    user_event = raw_data['user_event'].toPandas()

    # dump pickle
    logger.info(f'[Data Preparation][Planet News Hot Item] dump pickle.')
    if opt.save:
        dump_pickle(f'dataset', df_pd, base_path=base_path)
        dump_pickle(f'col2label', col2label, base_path=base_path)
        dump_pickle(f'user_event', user_event, base_path=base_path)

    # Upload checkpoint, dataset to GCS
    logger.info(f'[Data Preparation][Planet News Hot Item] upload dataset to GCS.')
    if opt.upload_gcs:
        upload_to_gcs(
            base_path,
            f'machine-learning-models-{project_id}',
            f'dataset/{opt.content_type}/hot/{run_time}'
        )
