# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'datapreparation.zip')

import pyspark.sql.functions as f
from src.options.club_post_options import PostUser2ItemOptions
from src.gcsreader.config.club_configs import ClubPostUser2ItemConfig
from src.gcsreader.club_post_user2item import ClubPostUser2ItemReader
from src.gcsreader import dedup_by_col, join_event_with_user_profile
from src.spark_negative_sampler import generate_negative_samples
from utils import initial_spark, dump_pickle, upload_to_gcs, sample_positive_data
from utils.parser import ContentDistinctValueParser
from utils.logger import logger


CONTENT_PROPERTY = 'beanfun'
CONTENT_TYPE = 'club_post'
USER_PROFILE_TO_JOIN = ['user_category', 'user_post_embedding']


def get_raw_data():

    df_event = gcsreader.get_event_data()
    df_content = gcsreader.get_content_data()
    df_popularity = gcsreader.get_popularity_data()
    df_user_category = gcsreader.get_user_category()
    df_user_gamania_meta = gcsreader.get_user_meta_data('gamania_meta')
    df_user_beanfun_meta = gcsreader.get_user_meta_data('beanfun_meta')
    df_user_post_embedding = gcsreader.get_user_embedding(embedding_key='post', content_type='club_post', user_profile_type='post_embedding')

    raw_data = {
        'event': df_event,
        'content': df_content,
        'popularity': df_popularity,
        'user_category': df_user_category,
        'user_gamania_meta': df_user_gamania_meta,
        'user_beanfun_meta': df_user_beanfun_meta,
        'user_post_embedding': df_user_post_embedding}

    for data_name, column_to_rename_list in ClubPostUser2ItemConfig.COLUMN_TO_RENAME.items():
        for renamed_tuple in column_to_rename_list:
            raw_data[data_name] = raw_data[data_name].withColumnRenamed(renamed_tuple[0], renamed_tuple[1])

    return raw_data


if __name__ == '__main__':

    opt = PostUser2ItemOptions().parse()

    PROJECT_ID = opt.project_id
    RUN_TIME = opt.run_time
    DAYS = int(opt.days)
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    NEGATIVE_SAMPLE_SIZE = int(opt.negative_sample_size)
    DAILY_SAMPLE_SEED = int(opt.daily_sample_seed)
    DAILY_POSITIVE_SAMPLE_SIZE = int(opt.daily_positive_sample_size)

    sql_context, spark_session = initial_spark(cores='5', memory='27G', overhead='5G', shuffle_partitions='1000', num_executors='3')
    sql_context.sql('SET spark.sql.autoBroadcastJoinThreshold = -1')  # disable broadcast join
    sql_context.sql('SET spark.sql.broadcastTimeout = 600000ms')
    sql_context.sql('SET spark.sql.shuffle.partitions = 1000')

    gcsreader = ClubPostUser2ItemReader(PROJECT_ID, sql_context, CONTENT_TYPE, CONTENT_PROPERTY, RUN_TIME, days=DAYS, logger=logger)
    raw_data = get_raw_data()
    raw_data['event'] = sample_positive_data(
        raw_data['event'],
        daily_positive_sample_size=DAILY_POSITIVE_SAMPLE_SIZE,
        random_seed=DAILY_SAMPLE_SEED)

    for key, data in raw_data.items():
        logger.info(f'[Data Preparation][beanfun! club user2item] {key} data count : {data.count()}')

    # process positive data
    df_distinct_userid = raw_data['event'].select('userid').dropDuplicates(subset=['userid'])

    for profile_type in USER_PROFILE_TO_JOIN:
        logger.info(f'Origin {profile_type} profile count : {raw_data[profile_type].count()}')
        raw_data[profile_type] = raw_data[profile_type].join(f.broadcast(df_distinct_userid), how='inner', on=['userid'])
        logger.info(f'Pruned {profile_type} profile count : {raw_data[profile_type].count()}')
        raw_data[profile_type].cache()

    logger.info(f'[Data Preparation][beanfun! club user2item] Join positive event data with user profile data')

    df_positive = raw_data['event']
    for key, df_profile in raw_data.items():
        if key in USER_PROFILE_TO_JOIN:
            logger.info(f'processing join {key} data')
            condition = [df_positive.userid == df_profile.userid, df_positive.date == df_profile.date]
            df_positive = join_event_with_user_profile(df_positive, df_profile, cond=condition, how='left')
            df_positive.cache()

    logger.info(f'[Data Preparation][beanfun! club user2item] positive data count : {df_positive.count()}')

    user_profile_col = df_positive.columns
    df_positive = df_positive.join(raw_data['user_gamania_meta'], on=['userid'], how='left')
    df_positive = df_positive.join(raw_data['user_beanfun_meta'], on=['userid'], how='left')
    df_positive = df_positive.join(raw_data['content'], on=['content_id'], how='inner')
    df_positive = df_positive.join(raw_data['popularity'], on=['content_id', 'date', 'hour'], how='left')

    # process negative data
    df_dedup_positive = dedup_by_col(df_positive, unique_col_base=['userid'], time_col='publish_time')

    df_negative = generate_negative_samples(
        df_event=df_positive,
        df_content=raw_data['content'],
        major_col='userid',
        candidate_col='content_id',
        time_col='publish_time',
        sample_size=NEGATIVE_SAMPLE_SIZE)

    logger.info(f'[Data Preparation][beanfun! club user2item] negative data count : {df_negative.count()}')

    df_negative = df_negative.join(df_dedup_positive[user_profile_col], on=['userid'], how='left').drop(df_dedup_positive.content_id)
    df_negative = df_negative.join(raw_data['user_gamania_meta'], on=['userid'], how='left')
    df_negative = df_negative.join(raw_data['user_beanfun_meta'], on=['userid'], how='left')
    df_negative = df_negative.join(raw_data['content'], on=['content_id'], how='inner')
    df_negative = df_negative.join(raw_data['popularity'], on=['content_id', 'date', 'hour'], how='left')

    # integration positive and negative data
    logger.info('[Data Preparation][beanfun! club user2item] Integration positive and negative dataset')
    df_positive = df_positive.withColumn('y', f.lit(1))
    df_negative = df_negative.withColumn('y', f.lit(0))
    dataset = df_positive.unionByName(df_negative)

    logger.info('[Data Preparation][beanfun! club user2item] Convert pyspark dataframe to pandas dataframe')
    dataset = dataset.toPandas()

    logger.info(f'Len of positive sample: {len(dataset[dataset["y"]==1])}')
    logger.info(f'Len of negative sample: {len(dataset[dataset["y"]==0])}')
    logger.info(f'Len of dataset: {len(dataset)}')

    # collect distinct label
    distinct_content_parser = ContentDistinctValueParser()
    col2label = distinct_content_parser.parse(raw_data['content'],
                                              ClubPostUser2ItemConfig.COLS_TO_ENCODE['content'],
                                              add_ner=False)

    logger.info('[Data Preparation][beanfun! club user2item] Upload dataset to GCS')
    dump_pickle(f'dataset', dataset, base_path=BASE_PATH)
    dump_pickle(f'col2label', col2label, base_path=BASE_PATH)

    # Upload checkpoint, dataset to GCS
    if opt.upload_gcs:
        upload_to_gcs(
            BASE_PATH,
            f'machine-learning-models-{PROJECT_ID}',
            f'dataset/club_post/user2item/{RUN_TIME}'
        )
