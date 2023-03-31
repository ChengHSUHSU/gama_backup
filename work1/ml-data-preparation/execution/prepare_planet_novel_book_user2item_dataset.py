# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'datapreparation.zip')

from utils.logger import logger
from utils.parser import ContentDistinctValueParser
from utils import initial_spark, filter_by_interaction_count, sample_positive_data, prune_preference_user_profile, \
    dump_pickle, upload_to_gcs
from src.spark_negative_sampler import generate_negative_samples
from src.options.planet_novel_book_options import NovelBookUser2ItemOptions
from src.gcsreader.config.planet_configs import PlanetNovelBookUser2ItemConfig
from src.gcsreader.planet_novel_book_user2item import PlanetNovelBookUser2ItemReader
from src.gcsreader import join_event_with_user_profile, dedup_by_col
import pyspark.sql.types as t
import pyspark.sql.functions as f


USER_PROFILE_LIST = ['user_category', 'user_tag', 'user_embedding']
UNIQUE_COL_SET = ['userid', 'uuid', 'timestamp']
BASE_PREF_USER_PROFILE_CONDITION = ('pref', 0.01)


def get_raw_data_from_gcs():

    raw_data = {
        'user_event': gcsreader.get_event_data(),
        'item_content': gcsreader.get_content_data(),
        'user_category': gcsreader.get_user_profile('category'),
        'user_tag': gcsreader.get_user_profile('tag'),
        'user_embedding': gcsreader.get_user_profile('embedding'),
        'user_meta': gcsreader.get_user_meta_data()
    }

    for data_name, column_to_rename_list in PlanetNovelBookUser2ItemConfig.COLUMN_TO_RENAME.items():
        for renamed_tuple in column_to_rename_list:
            raw_data[data_name] = raw_data[data_name].withColumnRenamed(renamed_tuple[0], renamed_tuple[1])

    return raw_data


if __name__ == '__main__':

    opt = NovelBookUser2ItemOptions().parse()

    PROJECT_ID = opt.project_id
    RUN_TIME = opt.run_time
    DAYS = int(opt.days)
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'

    NEGATIVE_SAMPLE_SIZE = int(opt.negative_sample_size)
    REQUISITE_SEQUENCE_LENGTH = int(opt.requisite_sequence_length)
    DAILY_SAMPLE_SEED = int(opt.daily_sample_seed)
    DAILY_POSITIVE_SAMPLE_SIZE = int(opt.daily_positive_sample_size)

    sql_context, spark_session = initial_spark()

    # Process raw data from gcs
    logger.info('[Data Preparation][Planet Novel User2Item] Get raw data from GCS')
    gcsreader = PlanetNovelBookUser2ItemReader(PROJECT_ID, sql_context, RUN_TIME, DAYS, logger=logger)
    raw_data = get_raw_data_from_gcs()

    raw_data['user_event'] = filter_by_interaction_count(raw_data['user_event'], primary_col='userid',
                                                         requisite_sequence_length=REQUISITE_SEQUENCE_LENGTH)
    logger.info(
        f'[Data Preparation][Planet Novel User2Item] Event count of user sequence length more than {REQUISITE_SEQUENCE_LENGTH} : {raw_data["user_event"].count()}')

    raw_data['user_event'] = sample_positive_data(raw_data['user_event'],
                                                  daily_positive_sample_size=DAILY_POSITIVE_SAMPLE_SIZE,
                                                  random_seed=DAILY_SAMPLE_SEED)
    logger.info(f'[Data Preparation][Planet Novel User2Item] Positive event sample size : {raw_data["user_event"].count()}')

    logger.info('[Data Preparation][Planet Novel User2Item] Planet novel user profile pruning')

    # Prune user tag profile
    raw_data['user_tag'] = raw_data['user_tag'].withColumn('user_tag', f.udf(
        lambda x: prune_preference_user_profile(x, tag_type=['editor'], condition=BASE_PREF_USER_PROFILE_CONDITION),
        t.StringType())(f.col('user_tag')))

    # Prune user profile by distinct userid
    df_distinct_userid = raw_data['user_event'].select('userid').dropDuplicates(subset=['userid'])
    logger.info(f'[Data Preparation][Planet Novel User2Item] Distinct userid count : {df_distinct_userid.count()}')
    for profile_type in USER_PROFILE_LIST:
        logger.info(f'[Data Preparation][Planet Novel User2Item] Origin {profile_type} profile count : {raw_data[profile_type].count()}')
        raw_data[profile_type] = raw_data[profile_type].join(f.broadcast(df_distinct_userid), how='inner', on=['userid'])
        logger.info(f'[Data Preparation][Planet Novel User2Item] Pruned {profile_type} profile count : {raw_data[profile_type].count()}')

        raw_data[profile_type].cache()
    logger.info('[Data Preparation][Planet Novel User2Item] Planet novel user profile pruned')

    logger.info('[Data Preparation][Planet Novel User2Item] Join positive event data with user profile data')
    df_positive = raw_data['user_event']
    user_profile_cols = PlanetNovelBookUser2ItemConfig.USER_PROFILE_COLUMNS
    for key, df_profile in raw_data.items():
        if key in USER_PROFILE_LIST:
            logger.info(f'Joining {key} data')

            # Rename column for de-dup user profile of same user
            df_profile = df_profile.withColumn('profile_date', f.col('date'))

            # Join user profile if event date >= profile date (later than)
            condition = [df_positive.userid == df_profile.userid, df_positive.date >= df_profile.profile_date]
            df_positive = join_event_with_user_profile(df_positive, df_profile, cond=condition, how='left')

            # De-dup thus keep newest user profile
            df_positive = dedup_by_col(df_positive, UNIQUE_COL_SET, time_col='profile_date').drop('profile_date')
            UNIQUE_COL_SET.extend(user_profile_cols[key])
            df_positive.cache()

    user_profile_col = df_positive.columns
    df_positive = df_positive.join(raw_data['item_content'], on=['uuid'], how='inner')
    df_positive = df_positive.join(raw_data['user_meta'], on=['userid'], how='left')

    logger.info(f'[Data Preparation][Planet Novel User2Item] Generate negative dataset. negative_sample_size : {NEGATIVE_SAMPLE_SIZE}')
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

    logger.info('[Data Preparation][Planet Novel User2Item] Integrate positive and negative dataset')
    df_positive = df_positive.withColumn('y', f.lit(1))
    df_negative = df_negative.withColumn('y', f.lit(0))
    dataset = df_positive.unionByName(df_negative)
    dataset = dataset.select(PlanetNovelBookUser2ItemConfig.FINAL_COLS)
    dataset.show(10)
    logger.info('[Data Preparation][Planet Novel User2Item] Convert pyspark dataframe to pandas dataframe')
    dataset = dataset.toPandas()

    logger.info(f'Len of positive sample: {len(dataset[dataset["y"] == 1])}')
    logger.info(f'Len of negative sample: {len(dataset[dataset["y"] == 0])}')
    logger.info(f'Len of dataset: {len(dataset)}')

    # collect distinct label
    distinct_content_parser = ContentDistinctValueParser()
    col2label = distinct_content_parser.parse(raw_data['item_content'],
                                              PlanetNovelBookUser2ItemConfig.COLS_TO_ENCODE['content'],
                                              add_ner=False)

    logger.info('[Data Preparation][Planet Novel User2Item] Upload dataset to GCS')
    dump_pickle('dataset', dataset, base_path=BASE_PATH)
    dump_pickle(f'col2label', col2label, base_path=BASE_PATH)

    # Upload checkpoint, dataset to GCS
    if opt.upload_gcs:
        upload_to_gcs(
            BASE_PATH,
            f'machine-learning-models-{PROJECT_ID}',
            f'dataset/{opt.content_type}/user2item/{RUN_TIME}'
        )
