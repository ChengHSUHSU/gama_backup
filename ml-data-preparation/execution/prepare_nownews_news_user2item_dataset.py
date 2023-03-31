# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'datapreparation.zip')

import pyspark.sql.functions as f
from pyspark.sql.functions import lit, col, udf
from pyspark.sql.types import StringType
from src.options.nownews_news_option import NowNewsNewsPreparationOptions
from src.gcsreader.config.nownews_config import NownewsNewsUser2ItemConfig
from src.gcsreader.nownews_news_user2item import NownewsNewsUser2ItemReader
from src.gcsreader import dedup_by_col, join_event_with_user_profile
from src.negative_sampler import generate_negative_samples
from utils import initial_spark, dump_pickle, upload_to_gcs, sample_positive_data, prune_preference_user_profile, filter_by_interaction_count
from utils.parser import ContentDistinctValueParser
from utils.logger import logger


CONTENT_TYPE = 'nownews_news'
USER_PROFILE_LIST_TO_GENERATE_PREVIOUS_DATE = ['user_category', 'user_tag', 'user_embedding']
BASE_PREF_USER_PROFILE_CONDITION = ('pref', 0.01)


def get_raw_data_from_gcs():

    raw_data = {
        'user_event': gcsreader.get_event_data(),
        'item_content': gcsreader.get_content_data(),
        'user_category': gcsreader.get_user_category(),
        'user_tag': gcsreader.get_user_tag(),
        'user_embedding': gcsreader.get_user_embedding(),
        'user_meta': gcsreader.get_user_meta_data()
    }

    for data_name, column_to_rename_list in NownewsNewsUser2ItemConfig.COLUMN_TO_RENAME.items():
        for renamed_tuple in column_to_rename_list:
            raw_data[data_name] = raw_data[data_name].withColumnRenamed(renamed_tuple[0], renamed_tuple[1])

    return raw_data


if __name__ == '__main__':

    opt = NowNewsNewsPreparationOptions().parse()

    PROJECT_ID = opt.project_id
    RUN_TIME = opt.run_time
    DAYS = int(opt.days)
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'

    NEGATIVE_SAMPLE_SIZE = int(opt.negative_sample_size)
    REQUISITE_SEQUENCE_LENGTH = int(opt.requisite_sequence_length)
    DAILY_SAMPLE_SEED = int(opt.daily_sample_seed)
    DAILY_POSITIVE_SAMPLE_SIZE = int(opt.daily_positive_sample_size)

    sql_context, spark_session = initial_spark(cores='5', memory='27G', overhead='5G', shuffle_partitions='1000', num_executors='3')
    sql_context.sql("SET spark.sql.autoBroadcastJoinThreshold = -1")  # disable broadcast join
    sql_context.sql("SET spark.sql.broadcastTimeout = 600000ms")
    sql_context.sql("SET spark.sql.shuffle.partitions = 1000")

    # Process raw data from gcs
    logger.info(f'[Data Preparation][Nownews User2Item] Get raw data from GCS')
    gcsreader = NownewsNewsUser2ItemReader(PROJECT_ID, sql_context, CONTENT_TYPE, RUN_TIME, DAYS, logger=logger)
    raw_data = get_raw_data_from_gcs()

    raw_data['user_event'] = filter_by_interaction_count(raw_data['user_event'], primary_col='userid',
                                                         requisite_sequence_length=REQUISITE_SEQUENCE_LENGTH)
    logger.info(f'event count of user sequence length more than {REQUISITE_SEQUENCE_LENGTH} : {raw_data["user_event"].count()}')

    raw_data['user_event'] = sample_positive_data(
        raw_data['user_event'],
        daily_positive_sample_size=DAILY_POSITIVE_SAMPLE_SIZE,
        random_seed=DAILY_SAMPLE_SEED)

    logger.info(f'positive event sample size : {raw_data["user_event"].count()}')

    logger.info(f'[Data Preparation][Nownews User2Item] Nownews user profile pruning')

    raw_data['user_tag'] = raw_data['user_tag'].dropDuplicates(subset=['userid', 'date'])   # nn user tag profile has duplicates data

    # prune user tag profile
    raw_data['user_tag'] = raw_data['user_tag'].withColumn('user_tag', udf(
        lambda x: prune_preference_user_profile(x, tag_type=['editor'], condition=BASE_PREF_USER_PROFILE_CONDITION), StringType())(col('user_tag')))

    df_distinct_userid = raw_data['user_event'].select('userid').dropDuplicates(subset=['userid'])

    logger.info(f'Distinct userid count : {df_distinct_userid.count()}')

    for profile_type in USER_PROFILE_LIST_TO_GENERATE_PREVIOUS_DATE:
        logger.info(f'Origin {profile_type} profile count : {raw_data[profile_type].count()}')
        raw_data[profile_type] = raw_data[profile_type].join(f.broadcast(df_distinct_userid), how='inner', on=['userid'])
        logger.info(f'Pruned {profile_type} profile count : {raw_data[profile_type].count()}')

        raw_data[profile_type].cache()

    logger.info(f'[Data Preparation][Nownews User2Item] Join positive event data with user profile data')

    df_positive = raw_data['user_event']
    for key, df_profile in raw_data.items():
        if key in USER_PROFILE_LIST_TO_GENERATE_PREVIOUS_DATE:
            logger.info(f'processing join {key} data')
            condition = [df_positive.userid == df_profile.userid, df_positive.date == df_profile.date]
            df_positive = join_event_with_user_profile(df_positive, df_profile, cond=condition, how='left')
            df_positive.cache()

    user_profile_col = df_positive.columns
    df_positive = df_positive.join(raw_data['item_content'], on=['content_id'], how='inner')
    df_positive = df_positive.join(raw_data['user_meta'], on=['userid'], how='left')

    logger.info(f'[Data Preparation][Nownews User2Item] Generate negative dataset. negative_sample_size : {NEGATIVE_SAMPLE_SIZE}')
    df_dedup_positive = dedup_by_col(df_positive, unique_col_base=['userid'], time_col='timestamp')
    df_positive_pandas = df_positive.select(['userid', 'content_id', 'publish_time']).toPandas()

    df_neg_pair = generate_negative_samples(
        df_positive_pandas,
        major_col='userid',
        candidate_col='content_id',
        time_col='publish_time',
        sample_size=NEGATIVE_SAMPLE_SIZE
    )

    df_negative = spark_session.createDataFrame(df_neg_pair)
    df_negative = df_negative.join(df_dedup_positive[user_profile_col], on=['userid'], how='left').drop(df_dedup_positive.content_id)
    df_negative = df_negative.join(raw_data['item_content'], on=['content_id'], how='left')
    df_negative = df_negative.join(raw_data['user_meta'], on=['userid'], how='left')

    logger.info('[Data Preparation][Nownews User2Item] Integration positive and negative dataset')
    df_positive = df_positive.withColumn('y', lit(1))
    df_negative = df_negative.withColumn('y', lit(0))
    dataset = df_positive.unionByName(df_negative)
    dataset = dataset.select(NownewsNewsUser2ItemConfig.FINAL_COLS)

    logger.info('[Data Preparation][Nownews User2Item] Convert pyspark dataframe to pandas dataframe')
    dataset = dataset.toPandas()

    logger.info(f'Len of positive sample: {len(dataset[dataset["y"]==1])}')
    logger.info(f'Len of negative sample: {len(dataset[dataset["y"]==0])}')
    logger.info(f'Len of dataset: {len(dataset)}')

    # collect distinct label
    distinct_content_parser = ContentDistinctValueParser()
    col2label = distinct_content_parser.parse(raw_data['item_content'],
                                              NownewsNewsUser2ItemConfig.COLS_TO_ENCODE['content'],
                                              add_ner=False)

    logger.info('[Data Preparation][Nownews User2Item] Upload dataset to GCS')
    dump_pickle(f'dataset', dataset, base_path=BASE_PATH)
    dump_pickle(f'col2label', col2label, base_path=BASE_PATH)

    # Upload checkpoint, dataset to GCS
    if opt.upload_gcs:
        upload_to_gcs(
            BASE_PATH,
            f'machine-learning-models-{PROJECT_ID}',
            f'dataset/nownews_news/user2item/{RUN_TIME}'
        )
