# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'datapreparation.zip')

import pyspark.sql.functions as f
from utils.logger import Logger
from utils.parser import ContentDistinctValueParser
from utils import dump_pickle, upload_to_gcs, join_df
from utils import initial_spark
from src.gcsreader.jollybuy_goods_hot import JollybuyGoodsHotItemReader
from src.gcsreader.utils import udf_get_candidate_arms
from src.options.jollybuy_goods_options import GoodsHotOptions
from utils import HOUR_STRING_FORMAT
from datetime import datetime, timedelta


TARGET_FEATURES = ['impression', 'click', 'booster', 'reward']
UNIFIED_ID_COL = 'uuid'
TIME_SESSION_COL = 'hour'
DATE_COL = 'date'
NEGATIVE_CANDIDATES_COL = 'uuids'


def get_raw_data(reader):

    # event data
    df_event = reader.get_event_data()

    # content data
    df_content = reader.get_content_data(config_key='content_pool')

    # candidate data
    df_candidates = reader.get_candidates(df_event, 'input_data_pool')

    # impression data
    df_impression = reader.get_impression_cnts(df_candidates)

    # click data
    df_click = reader.get_click_cnts(df_candidates)

    # booster data
    df_booster = reader.get_booster_cnts(df_candidates)

    # reward data
    df_reward = reader.get_reward_cnts(df_candidates)

    raw_data = {
        'event': df_event,
        'content': df_content,
        'candidates': df_candidates,
        'impression': df_impression,
        'click': df_click,
        'booster': df_booster,
        'reward': df_reward
    }
    return raw_data


if __name__ == '__main__':

    # define arguments
    opt = GoodsHotOptions().parse()
    project_id = opt.project_id
    run_time = opt.run_time
    content_type = opt.content_type
    content_property = opt.content_property
    input_bucket = opt.input_bucket
    negative_sample_size = int(opt.negative_sample_size)
    days = int(opt.days)

    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    # TODO: Add filtering for blindbox
    item_content_type = opt.item_content_type  # preserve for filtering blindbox (not for use now)

    # define logger
    LOGGER = Logger(logger_name=opt.logger_name, dev_mode=True)
    LOGGER.info(str(opt))

    # define reader
    sql_context, spark_session = initial_spark(memory='20G', overhead='15G')
    sql_context.sql("SET spark.sql.autoBroadcastJoinThreshold = -1")  # disable broadcast join
    log4jLogger = sql_context._jvm.org.apache.log4j

    run_time = run_time if run_time else (datetime.now()-timedelta(hours=1)).strftime(HOUR_STRING_FORMAT)
    reader = JollybuyGoodsHotItemReader(project_id, sql_context, input_bucket, content_type, content_property, run_time, days, LOGGER)

    # get raw data
    raw_data = get_raw_data(reader)

    # aggregate data
    for i, feature_key in enumerate(TARGET_FEATURES):

        if i == 0:
            df_data = raw_data[feature_key]
        else:
            df_data = join_df(df_data, raw_data[feature_key], on=[UNIFIED_ID_COL, TIME_SESSION_COL, DATE_COL], how='left')

    df_data = df_data.fillna(0)
    df_data = df_data.filter(f.col('reward') != 0)

    LOGGER.info(f'[Data Preparation][Jollybuy Goods Hot Items] prepared data: {df_data.count()}')
    df_data.limit(5).show(5, False)

    # join with df_content to get cat0, cat1, and title (for debug usage)
    df_data = join_df(df_data, raw_data['content'], on=[UNIFIED_ID_COL], how='inner')
    LOGGER.info(f'[Data Preparation][Jollybuy Goods Hot Items] Join content data: {df_data.count()}')

    # get negative arms
    df_negative_candidates = df_data.groupby([DATE_COL, TIME_SESSION_COL]) \
                                    .agg(f.collect_set(f.col(UNIFIED_ID_COL)).alias(NEGATIVE_CANDIDATES_COL))
    df_data = df_data.join(df_negative_candidates, on=[DATE_COL, TIME_SESSION_COL], how='left')
    df_data = df_data.withColumn(NEGATIVE_CANDIDATES_COL,
                                 udf_get_candidate_arms(negative_sample_size,
                                                        neg_candidate_col=NEGATIVE_CANDIDATES_COL,
                                                        pos_id_col=UNIFIED_ID_COL)(f.struct(f.col(UNIFIED_ID_COL),
                                                                                   f.col(NEGATIVE_CANDIDATES_COL))))

    df_data.cache()
    LOGGER.info(f'[Data Preparation][Jollybuy Goods Hot Items] prepared data inner join with df_content: {df_data.count()}')
    df_data.limit(5).show(5, False)

    # transform
    df_pd = df_data.toPandas()

    LOGGER.info(f'[Data Preparation][Jollybuy Goods Hot Items] Total Data: {len(df_pd)}')

    # collect all distinct cat0, cat1
    distinct_content_parser = ContentDistinctValueParser()
    col2label = distinct_content_parser.parse(raw_data['content'], reader.config.COLS_TO_ENCODE['content_pool'], add_ner=False)

    ### dump data ###
    if opt.save:
        dump_pickle(f'dataset', df_pd, base_path=BASE_PATH)
        dump_pickle(f'col2label', col2label, base_path=BASE_PATH)

    if opt.upload_gcs:
        upload_to_gcs(
            BASE_PATH,
            f'machine-learning-models-{project_id}',
            f'dataset/{content_type}/hot/{run_time}'
        )
