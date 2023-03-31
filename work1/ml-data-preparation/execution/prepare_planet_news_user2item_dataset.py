# -*- coding: utf-8 -*-
import sys

sys.path.insert(0, 'datapreparation.zip')

from datetime import datetime, timedelta

import pyspark.sql.functions as f
from pyspark.sql.functions import lit
from pyspark.sql.types import StringType

from src.gcsreader import dedup_by_col, join_event_with_user_profile
from src.gcsreader.config.planet_configs import PlanetNewsUser2ItemOptimizedConfig
from src.gcsreader.planet_news_user2item_aggregator import PlanetNewsUser2ItemAggregator
from src.gcsreader.samples.samples_pool import get_similarity_samples_pool
from src.options.planet_news_options import NewsUser2ItemOptions
from src.spark_negative_sampler import generate_negative_samples
from utils import (DAY_STRING_FORMAT, dump_pickle, filter_by_interaction_count, initial_spark, sample_positive_data, upload_to_gcs)
from utils.logger import logger
from utils.parser import ContentDistinctValueParser


def main():

    logger.info('[Data Preparation][planet news user2item] Raw data aggregation')
    aggregator = PlanetNewsUser2ItemAggregator(project_id, sql_context, run_time, days=days, logger=logger, config=PlanetNewsUser2ItemOptimizedConfig)

    # get nownews content for news tag set integration
    aggregator.read(property_name='nownews', content_type='nownews_news', processed_data=['content'])
    nownews_news_data = aggregator.data.copy()

    # get main raw dataset
    aggregator.read(property_name=content_property, content_type=content_type, processed_data=['event', 'content', 'metrics', 'user_profile'])
    raw_data = aggregator.data

    # strictly sampling
    raw_data['event'] = filter_by_interaction_count(raw_data['event'], primary_col='userid', requisite_sequence_length=requisite_sequence_length)
    raw_data['event'] = sample_positive_data(raw_data['event'], daily_positive_sample_size=daily_positive_sample_size, random_seed=daily_sample_seed)

    logger.info(f'[Data Preparation][planet news user2item] Prepare positive dataset')

    df_positive = raw_data['event']

    for profile_key, df_profile in raw_data['user_profile'].items():

        if profile_key == 'meta':
            continue

        logger.info(f'processing join {profile_key} data')
        condition = [df_positive.userid == df_profile.userid, df_positive.date == df_profile.date]
        df_positive = join_event_with_user_profile(df_positive, df_profile, cond=condition, how='left')

    user_profile_col = df_positive.columns
    df_positive = df_positive.join(raw_data['user_profile']['meta'], on=['userid'], how='left')
    df_positive = df_positive.join(raw_data['metrics']['popularity'], on=['content_id', 'date', 'hour'], how='left')
    df_positive = df_positive.join(raw_data['content'], on=['content_id'], how='inner')

    df_dedup_positive = dedup_by_col(df_positive, unique_col_base=['userid'], time_col='publish_time')

    logger.info(f'[Data Preparation][planet news user2item] Prepare positive and negative sample pool')
    df_content_pool = df_positive.select(['content_id', 'publish_time']).distinct()

    df_freshness_content_pool = df_content_pool.orderBy([f.col('publish_time')], ascending=False).limit(topk_freshness)

    df_popular_content_pool = raw_data['metrics']['snapshot_popularity'].orderBy([f.col('popularity_score')], ascending=False).limit(topk_popular)
    df_popular_content_pool = df_popular_content_pool.join(raw_data['content'], on=['content_id'], how='inner')
    df_popular_content_pool = df_popular_content_pool.select(['content_id', 'publish_time'])
    df_similar_event_pool = get_similarity_samples_pool(df_positive, topk_similar)

    logger.info(f'[Data Preparation][planet news user2item] sample by top-{topk_popular} popular data')
    df_positive_popular = df_positive.join(df_popular_content_pool.select('content_id'), on=['content_id'], how='inner')

    logger.info(f'[Data Preparation][planet news user2item] sample by top-{topk_freshness} freshness data')
    df_positive_freshness = df_positive.join(df_freshness_content_pool.select('content_id'), on=['content_id'], how='inner')

    df_positive = df_positive.withColumn('sample_type', f.lit('positive'))
    df_positive_popular = df_positive_popular.withColumn('sample_type', f.lit('popular'))
    df_positive_freshness = df_positive_freshness.withColumn('sample_type', f.lit('freshness'))
    df_positive = df_positive \
        .unionByName(df_positive_popular) \
        .unionByName(df_positive_freshness)

    logger.info(f'[Data Preparation][planet news user2item] sample by top-{topk_similar} similar data')
    df_negative_similar = generate_negative_samples(df_event=df_positive,
                                                    df_content=df_similar_event_pool.select(['content_id', 'publish_time']),
                                                    major_col='userid',
                                                    candidate_col='content_id',
                                                    time_col='publish_time',
                                                    sample_size=similar_sample_size)

    logger.info(f'[Data Preparation][planet news user2item] similar sample ratio {similar_sample_ratio}')
    df_negative_similar = df_negative_similar.sample(fraction=float(similar_sample_ratio))

    logger.info('[Data Preparation][planet news user2item] process random sample')
    df_negative_random = generate_negative_samples(df_event=df_positive,
                                                   df_content=df_content_pool,
                                                   major_col='userid',
                                                   candidate_col='content_id',
                                                   time_col='publish_time',
                                                   sample_size=random_sample_size)

    df_negative_similar = df_negative_similar.withColumn('sample_type', f.lit('similar'))
    df_negative_random = df_negative_random.withColumn('sample_type', f.lit('random'))

    logger.info(f'[Data Preparation][planet news user2item] length of similar sample {df_negative_similar.count()}')
    logger.info(f'[Data Preparation][planet news user2item] length of random sample {df_negative_random.count()}')

    df_negative = df_negative_random.unionByName(df_negative_similar)

    df_negative = df_negative.join(df_dedup_positive[user_profile_col], on=['userid'], how='left').drop(df_dedup_positive.content_id)
    df_negative = df_negative.join(raw_data['user_profile']['meta'], on=['userid'], how='left')
    df_negative = df_negative.join(raw_data['metrics']['popularity'], on=['content_id', 'date', 'hour'], how='left')
    df_negative = df_negative.join(raw_data['content'], on=['content_id'], how='inner')

    logger.info(f'[Data Preparation][planet news user2item] Integrate positive and negative dataset')
    df_positive = df_positive.withColumn('y', lit(1))
    df_negative = df_negative.withColumn('y', lit(0))
    dataset = df_positive.unionByName(df_negative)

    logger.info(f'[Data Preparation][planet news user2item] Convert pyspark dataframe to pandas dataframe')
    dataset = dataset.toPandas()

    logger.info(f'Len of positive sample: {len(dataset[dataset["y"]==1])}')
    logger.info(f'Len of negative sample: {len(dataset[dataset["y"]==0])}')
    logger.info(f'Len of dataset: {len(dataset)}')

    # collect distinct label
    distinct_content_parser = ContentDistinctValueParser()
    content_to_encode = raw_data['content'].unionByName(nownews_news_data['content'])
    col2label = distinct_content_parser.parse(content_to_encode, PlanetNewsUser2ItemOptimizedConfig.COLS_TO_ENCODE['content'], add_ner=False)

    logger.info('[Data Preparation][planet news user2item] Upload dataset to GCS')
    dump_pickle('dataset', dataset, base_path=base_path)
    dump_pickle('col2label', col2label, base_path=base_path)

    # Upload checkpoint, dataset to GCS
    if opt.upload_gcs:
        upload_to_gcs(base_path,
                      f'machine-learning-models-{project_id}',
                      f'dataset/{content_type}/{service_type}/{run_time}')


if __name__ == '__main__':
    opt = NewsUser2ItemOptions().parse()

    # base arguments
    project_id = opt.project_id
    run_time = opt.run_time
    content_type = opt.content_type
    content_property = opt.content_property
    service_type = opt.service_type
    days = int(opt.days)
    negative_sample_size = int(opt.negative_sample_size)
    base_path = f'{opt.checkpoints_dir}/{opt.experiment_name}'

    # sample arguments
    requisite_sequence_length = opt.requisite_sequence_length
    daily_positive_sample_size = opt.daily_positive_sample_size
    daily_sample_seed = opt.daily_sample_seed
    topk_freshness = opt.topk_freshness
    topk_popular = opt.topk_popular
    topk_similar = opt.topk_similar
    similar_sample_ratio = opt.similar_sample_ratio
    similar_sample_size = opt.similar_sample_size
    random_sample_size = opt.random_sample_size

    # Spark initialize
    sql_context, spark_session = initial_spark()
    sql_context, spark_session = initial_spark(cores='5', memory='27G', overhead='5G', shuffle_partitions='1000', num_executors='3')
    sql_context.sql('SET spark.sql.broadcastTimeout = 600000ms')

    main()
