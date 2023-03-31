import sys
sys.path.insert(0, 'datapreparation.zip')

import pyspark.sql.functions as f
from pyspark.sql.types import StringType
from src.options.planet_comics_book_options import ComicsBookUser2ItemOptions
from src.gcsreader.config.planet_configs import PlanetComicsBookUser2ItemConfig
from src.gcsreader.planet_comics_book_user2item_aggregator import PlanetComicsBookUser2ItemAggregator
from src.gcsreader import dedup_by_col, join_event_with_user_profile
from src.spark_negative_sampler import generate_negative_samples
from utils import initial_spark, dump_pickle, upload_to_gcs, sample_positive_data, prune_preference_user_profile, filter_by_interaction_count
from utils.parser import ContentDistinctValueParser
from utils.logger import logger


def get_raw_data(
                    aggregator: PlanetComicsBookUser2ItemAggregator,
                    config: PlanetComicsBookUser2ItemConfig,
                    content_type: str,
                    content_property: str) -> dict:
    process_profile_type = config.PROCESS_PROFILE_TYPES
    process_metrics_type = config.PROCESS_METRICS_TYPES

    aggregator.read(content_property, content_type, process_profile_type, process_metrics_type)
    data = aggregator.get()

    df_event, df_content, df_popularity, df_user_profile = data['event'], data['content'], data['metrics'], data['user_profile']

    raw_data = {
        'user_event': df_event,
        'item_content': df_content,
        'popularity': df_popularity['popularity'],
        'user_category': df_user_profile['category'],
        'user_tag': df_user_profile['tag'],
        'user_embedding': df_user_profile['title_embedding'],
        'user_meta': df_user_profile['meta']
    }

    return raw_data


if __name__ == '__main__':
    config = PlanetComicsBookUser2ItemConfig
    opt = ComicsBookUser2ItemOptions().parse()

    project_id = opt.project_id
    run_time = opt.run_time
    days = int(opt.days)
    base_path = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    content_type = opt.content_type

    NEGATIVE_SAMPLE_SIZE = int(opt.negative_sample_size)
    REQUISITE_SEQUENCE_LENGTH = int(opt.requisite_sequence_length)
    DAILY_SAMPLE_SEED = int(opt.daily_sample_seed)
    DAILY_POSITIVE_SAMPLE_SIZE = int(opt.daily_positive_sample_size)

    sql_context, spark_session = initial_spark(cores='5', memory='27G', overhead='5G', shuffle_partitions='1000', num_executors='3')
    sql_context.sql('SET spark.sql.autoBroadcastJoinThreshold = -1')  # disable broadcast join
    sql_context.sql('SET spark.sql.broadcastTimeout = 600000ms')
    sql_context.sql('SET spark.sql.shuffle.partitions = 1000')
    sql_context.sql('SET spark.sql.adaptive.enabled = false')

    aggregator = PlanetComicsBookUser2ItemAggregator(project_id,
                                                     sql_context,
                                                     run_time,
                                                     days=days,
                                                     config=PlanetComicsBookUser2ItemConfig,
                                                     logger=logger)

    raw_data = get_raw_data(aggregator, config, opt.content_type, opt.content_property)
    raw_data['user_event'] = filter_by_interaction_count(raw_data['user_event'], primary_col='userid',
                                                         requisite_sequence_length=REQUISITE_SEQUENCE_LENGTH)
    logger.info(f'event count of user sequence length more than {REQUISITE_SEQUENCE_LENGTH} : {raw_data["user_event"].count()}')

    raw_data['user_event'] = sample_positive_data(
        raw_data['user_event'],
        daily_positive_sample_size=DAILY_POSITIVE_SAMPLE_SIZE,
        random_seed=DAILY_SAMPLE_SEED)

    logger.info(f'positive event sample size : {raw_data["user_event"].count()}')

    logger.info('[Data Preparation][Planet Comic User2Item] Planet comic user profile pruning')

    # prune user tag profile
    raw_data['user_tag'] = (
        raw_data['user_tag'].withColumn('user_tag',
                                        f.udf(
                                            lambda x: prune_preference_user_profile(x,
                                                                                    tag_type=['editor'],
                                                                                    condition=config.BASE_PREF_USER_PROFILE_CONDITION
                                                                                    ), StringType())(f.col('user_tag')))
    )

    df_distinct_userid = raw_data['user_event'].select('userid').dropDuplicates(subset=['userid'])
    logger.info(f'Distinct userid count : {df_distinct_userid.count()}')

    # prune user profile by distinct userid
    for profile_type in config.USER_PROFILE_LIST:
        logger.info(f'Origin {profile_type} profile count : {raw_data[profile_type].count()}')
        raw_data[profile_type] = raw_data[profile_type].join(f.broadcast(df_distinct_userid), how='inner', on=['userid'])
        logger.info(f'Pruned {profile_type} profile count : {raw_data[profile_type].count()}')

        raw_data[profile_type].cache()

    logger.info('[Data Preparation][Planet Comic User2Item] Join positive event data with user profile data')

    df_positive = raw_data['user_event']
    for key, df_profile in raw_data.items():
        if key in config.USER_PROFILE_LIST:
            logger.info(f'Joining {key} data')
            condition = [df_positive.userid == df_profile.userid, df_positive.date == df_profile.date]
            df_positive = join_event_with_user_profile(df_positive, df_profile, cond=condition, how='left')
            df_positive.cache()

    user_profile_col = df_positive.columns

    df_positive = df_positive.join(raw_data['item_content'], on=['uuid'], how='inner')
    df_positive = df_positive.join(raw_data['user_meta'], on=['userid'], how='left')
    df_positive = df_positive.join(raw_data['popularity'], on=['uuid', 'date', 'hour'], how='left')

    logger.info(f'[Data Preparation][Planet Comic User2Item] Generate negative dataset. negative_sample_size : {NEGATIVE_SAMPLE_SIZE}')
    df_dedup_positive = dedup_by_col(df_positive, unique_col_base=['userid'], time_col='timestamp')

    df_negative = generate_negative_samples(
        df_positive.select(['userid', 'uuid', 'publish_time']),
        raw_data['item_content'],
        major_col='userid',
        candidate_col='uuid',
        time_col='publish_time',
        sample_size=NEGATIVE_SAMPLE_SIZE
    )

    df_negative = df_negative.join(df_dedup_positive[user_profile_col], on=['userid'], how='left').drop(df_dedup_positive.uuid)
    df_negative = df_negative.join(raw_data['item_content'], on=['uuid'], how='left')
    df_negative = df_negative.join(raw_data['user_meta'], on=['userid'], how='left')
    df_negative = df_negative.join(raw_data['popularity'], on=['uuid', 'date', 'hour'], how='left')

    logger.info('[Data Preparation][Planet Comic User2Item] Integrate positive and negative dataset')
    df_positive = df_positive.withColumn('y', f.lit(1))
    df_negative = df_negative.withColumn('y', f.lit(0))
    dataset = df_positive.unionByName(df_negative)
    dataset = dataset.select(config.FINAL_COLS)
    dataset.show(10, False)
    logger.info('[Data Preparation][Planet Comic User2Item] Convert pyspark dataframe to pandas dataframe')
    dataset = dataset.toPandas()

    logger.info(f'Len of positive sample: {len(dataset[dataset["y"]==1])}')
    logger.info(f'Len of negative sample: {len(dataset[dataset["y"]==0])}')
    logger.info(f'Len of dataset: {len(dataset)}')

    # collect distinct label
    distinct_content_parser = ContentDistinctValueParser()
    col2label = distinct_content_parser.parse(raw_data['item_content'],
                                              PlanetComicsBookUser2ItemConfig.COLS_TO_ENCODE['content'],
                                              add_ner=False)

    logger.info('[Data Preparation][Planet Comic User2Item] Upload dataset to GCS')
    dump_pickle(f'dataset', dataset, base_path=base_path)
    dump_pickle(f'col2label', col2label, base_path=base_path)

    # Upload checkpoint, dataset to GCS
    if opt.upload_gcs:
        upload_to_gcs(
            base_path,
            f'machine-learning-models-{project_id}',
            f'dataset/{opt.content_type}/user2item/{run_time}'
        )
