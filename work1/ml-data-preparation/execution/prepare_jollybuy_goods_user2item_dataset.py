import os
import sys
sys.path.insert(0, 'datapreparation.zip')
os.environ['NUMEXPR_MAX_THREADS'] = '16'

from src.spark_negative_sampler import generate_negative_samples
from utils.logger import Logger
from utils.parser import ContentDistinctValueParser
from utils import initial_spark, dump_pickle, upload_to_gcs, sample_positive_data, prune_preference_user_profile, filter_by_interaction_count
from src.gcsreader.jollybuy_goods_user2item_aggregator import JollybuyGoodsUser2ItemAggregator
from src.gcsreader.config.jollybuy_goods.jollybuy_goods_user2item import JollybuyGoodsUser2ItemConfig

from src.gcsreader import dedup_by_col, join_event_with_user_profile
from src.options.jollybuy_goods_options import GoodsUser2ItemOptions
from pyspark.sql.types import StringType
import pyspark.sql.functions as f


USER_PROFILES_TO_JOINED_WITH_DATE = ['user_title_embedding', 'user_category', 'user_tag']
POP_NEGATIVE_MULTIPLIER = 100


def get_raw_data(configs: JollybuyGoodsUser2ItemConfig, aggregator: JollybuyGoodsUser2ItemAggregator, content_type: str, content_property: str) -> dict:
    process_metrics_types = configs.PROCESS_METRICS_TYPES
    process_profile_types = configs.PROCESS_PROFILE_TYPES
    aggregator.read(content_property, content_type, process_profile_types, process_metrics_types)
    data = aggregator.get()
    df_event, df_content, df_popularity, df_user_profile = \
        data['event'], data['content'], data['metrics'], data['user_profile']
    raw_data = {
        'user_event': df_event,
        'item_content': df_content,
        'user_title_embedding': df_user_profile['title_embedding'],
        'user_category': df_user_profile['category'],
        'user_tag': df_user_profile['tag'],
        'user_meta': df_user_profile['meta'],
        'popularity': df_popularity['popularity'],
        'snapshot_popularity': df_popularity['snapshot_popularity']
    }

    return raw_data


if __name__ == '__main__':
    config = JollybuyGoodsUser2ItemConfig
    opt = GoodsUser2ItemOptions().parse()
    project_id = opt.project_id
    run_time = opt.run_time
    content_type = opt.content_type
    days = int(opt.days)
    content_negative_sample_size = int(opt.content_negative_sample_size)
    pop_negative_sample_size = int(opt.pop_negative_sample_size)
    requisite_event_sequence_length = int(opt.requisite_sequence_length)
    enable_positive_sampling = bool(opt.enable_positive_sampling)
    daily_positive_sample_size = int(opt.daily_positive_sample_size)
    daily_sample_seed = int(opt.daily_sample_seed)

    base_path = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    logger = Logger(logger_name=opt.logger_name, dev_mode=True)

    sql_context, spark_session = initial_spark(cores='5', memory='27G',
                                               overhead='5G', shuffle_partitions='1000',
                                               num_executors='3')
    sql_context.sql("SET spark.sql.autoBroadcastJoinThreshold = -1")  # disable broadcast join

    # get raw data
    logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Get raw data from GCS')
    aggregator = JollybuyGoodsUser2ItemAggregator(project_id, sql_context, run_time, days, config=config, logger=logger)
    raw_data = get_raw_data(configs=config, aggregator=aggregator, content_type=opt.content_type, content_property=opt.content_property)

    # generate positive data
    df_positive = raw_data['user_event']

    if bool(requisite_event_sequence_length):
        logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Filter out event with small click count')
        logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Filter threshold {requisite_event_sequence_length}')
        df_positive = filter_by_interaction_count(df_positive, primary_col='userid', requisite_sequence_length=requisite_event_sequence_length)
        logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Positive event count {df_positive.count()}')

    if opt.enable_positive_sampling:
        logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Positive Sampling by Date')
        logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Random seed: {daily_sample_seed}')
        logger.info(f'[Data Preparation][Jollybuy Goods User2Item] {daily_positive_sample_size} samples per day')
        df_positive = sample_positive_data(df_positive, daily_positive_sample_size, daily_sample_seed)

    # prune data by userid and content_id
    df_distinct_userid = raw_data['user_event'].select('userid').dropDuplicates(subset=['userid'])
    df_distinct_content_id = raw_data['user_event'].select('content_id').dropDuplicates(subset=['content_id'])

    pruned_by_userid_cols = getattr(config, 'PRUNED_BY_USERID_COLS', [])
    pruned_by_content_id_cols = getattr(config, 'PRUNED_BY_CONTENT_ID_COLS', [])

    for profile_type in raw_data.keys():

        logger.info(f'Origin {profile_type} profile count : {raw_data[profile_type].count()}')

        if profile_type in pruned_by_userid_cols:
            raw_data[profile_type] = raw_data[profile_type].join(df_distinct_userid, how='inner', on=['userid'])
        elif profile_type in pruned_by_content_id_cols:
            raw_data[profile_type] = raw_data[profile_type].join(df_distinct_content_id, how='inner', on=['content_id'])

        logger.info(f'Pruned {profile_type} profile count : {raw_data[profile_type].count()}')

    # prune preference profile
    logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Preference profile filtering')

    for profile_type, type2cond in getattr(config, 'PREFERENCE_FILTER_CONDITIONS', {}).items():
        logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Filter {profile_type}')
        for tag_type, cond in type2cond.items():
            raw_data[profile_type] = raw_data[profile_type].withColumn(profile_type,
                                                                       f.udf(lambda x: prune_preference_user_profile(x, tag_type=tag_type, condition=cond), StringType())(f.col(profile_type)))

    # join user profile with event
    # during preparation, we save all user profiles
    logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Join positive event data with user profile data')

    for profile_key in USER_PROFILES_TO_JOINED_WITH_DATE:
        logger.info(f'processing join {profile_key} data')

        df_profile = raw_data[profile_key]
        condition = [df_positive.userid == df_profile.userid, df_positive.date == df_profile.date]

        df_positive = join_event_with_user_profile(df_positive, df_profile, cond=condition, how='left')

    # append meta
    user_profile_col = df_positive.columns

    df_positive = df_positive.join(raw_data['popularity'], on=['content_id', 'date', 'hour'], how='left')
    df_positive = df_positive.join(raw_data['user_meta'], on=['userid'], how='left')
    df_positive = df_positive.join(raw_data['item_content'], on=['content_id'], how='inner')

    df_positive.cache()
    df_positive.show()
    df_positive.count()

    # negative sampling
    df_content_pool = df_positive.select(['content_id', 'publish_time']).distinct()

    # negative sampling
    df_dedup_positive = dedup_by_col(df_positive, unique_col_base=['userid'], time_col='publish_time')

    # generate negative by top-k popular items
    logger.info(
        f'[Data Preparation][Jollybuy Goods User2Item] Generate negative sample from top-{pop_negative_sample_size * POP_NEGATIVE_MULTIPLIER} popular items')

    df_snapshot_pop = raw_data['snapshot_popularity']
    df_snapshot_pop = df_snapshot_pop.limit(pop_negative_sample_size * POP_NEGATIVE_MULTIPLIER)
    df_snapshot_pop = df_snapshot_pop.join(raw_data['item_content'], on=['content_id'], how='inner').select(['content_id', 'publish_time'])
    df_pop_negative = generate_negative_samples(df_event=df_positive, df_content=df_snapshot_pop,
                                                major_col='userid', candidate_col='content_id',
                                                time_col='publish_time', sample_size=pop_negative_sample_size)

    logger.info('[Data Preparation][Jollybuy Goods User2Item] pop negative samples')
    df_pop_negative.show()
    # generate negative by all content pool
    logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Generate negative sample from content pool')

    df_negative = generate_negative_samples(df_event=df_positive, df_content=df_content_pool,
                                            major_col='userid', candidate_col='content_id',
                                            time_col='publish_time', sample_size=content_negative_sample_size)

    logger.info('[Data Preparation][Jollybuy Goods User2Item] content negative samples')
    df_negative.show()
    logger.info('[Data Preparation][Jollybuy Goods User2Item] final negative sample')
    df_negative_total = df_negative.unionByName(df_pop_negative)
    df_negative_total.cache()
    df_negative_total.show()

    # append profile and meta to negative samples
    df_negative_total = df_negative_total.join(df_dedup_positive[user_profile_col], on=['userid'], how='left') \
                                         .drop(df_dedup_positive.content_id)
    df_negative_total = df_negative_total.join(raw_data['user_meta'], on=['userid'], how='left')
    df_negative_total = df_negative_total.join(raw_data['item_content'], on=['content_id'], how='inner')
    df_negative_total = df_negative_total.join(raw_data['popularity'], on=['content_id', 'date', 'hour'], how='left')

    # integration positive and negative data
    logger.info('[Data Preparation][Jollybuy Goods User2Item] Integration positive and negative dataset')
    df_positive = df_positive.withColumn('y', f.lit(1))
    df_negative_total = df_negative_total.withColumn('y', f.lit(0))
    dataset = df_positive.unionByName(df_negative_total)

    dataset = dataset.select(getattr(config, 'FINAL_COLS', '*'))
    dataset = dataset.toPandas()

    logger.info(f"Len of positive sample: {len(dataset[dataset['y']==1])}")
    logger.info(f"Len of negative sample: {len(dataset[dataset['y']==0])}")
    logger.info(f"Len of dataset: {len(dataset)}")

    ### collect all distinct cat0, cat1, tags and store_id ###
    distinct_content_parser = ContentDistinctValueParser()
    col2label = distinct_content_parser.parse(raw_data['item_content'], config.COLS_TO_ENCODE['content'], add_ner=False)

    if opt.save:
        dump_pickle(f'dataset', dataset, base_path=base_path)
        dump_pickle(f'col2label', col2label, base_path=base_path)

    # Upload checkpoint, dataset to GCS
    if opt.upload_gcs:
        upload_to_gcs(
            base_path,
            f'machine-learning-models-{project_id}',
            f'dataset/{content_type}/user2item/{run_time}'
        )
