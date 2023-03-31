# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'datapreparation.zip')

from src.options.base_options import BaseOptions
from pyspark.sql.functions import col, get_json_object, lit
import ast

from utils import initial_spark, dump_pickle, upload_to_gcs
from utils.parser import ContentDistinctValueParser
from src.gcsreader.config.planet_configs import PlanetVideoUser2ItemConfig
from src.gcsreader.planet_video_user2item import PlanetVideoUser2ItemReader
from src.gcsreader import dedup_by_col
from src.negative_sampler import generate_negative_samples


def get_raw_data_from_gcs():
    raw_data = {
        'user_event': gcsreader.get_event_data(days=days),
        'item_content': gcsreader.get_content_data(),
        'user_video_category': gcsreader.get_user_category(),
        'user_video_tag': gcsreader.get_user_tag(),
        'user_video_title_embedding': gcsreader.get_user_title_embedding(),
        'user_news_category': gcsreader.get_user_category(content_type='news'),
        'user_news_title_embedding': gcsreader.get_user_title_embedding(content_type='news'),
        'user_meta': gcsreader.get_user_meta_data()
    }

    for data_name, column_to_rename_list in PlanetVideoUser2ItemConfig.COLUMN_TO_RENAME.items():
        for renamed_tuple in column_to_rename_list:
            raw_data[data_name] = raw_data[data_name].withColumnRenamed(renamed_tuple[0], renamed_tuple[1])

    return raw_data


def join_event_with_user_profile(df_left, df_profile, how='left'):
    condition = [
        df_left.openid == df_profile.openid,
        df_left.date == df_profile.date
    ]
    df = df_left.join(df_profile, on=condition, how=how) \
        .drop(df_profile.openid) \
        .drop(df_profile.date)

    return df


def parse_planet_category(df):
    def __split_cat(x):
        pos0 = ast.literal_eval(x)[0] if len(ast.literal_eval(x)) > 0 else None
        pos1 = ast.literal_eval(x)[1] if len(ast.literal_eval(x)) > 1 else None
        return pos0, pos1

    df['cat0_0'],  df['cat0_1'] = zip(*df['cat0'].apply(__split_cat))
    df['cat1_0'],  df['cat1_1'] = zip(*df['cat1'].apply(__split_cat))

    return df


if __name__ == '__main__':
    opt = BaseOptions().parse()
    project_id = opt.project_id
    run_time = opt.run_time
    content_type = opt.content_type
    content_property = opt.content_property
    days = int(opt.days)
    negative_sample_size = int(opt.negative_sample_size)
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    USER_PROFILE_LIST = ['user_video_category', 'user_video_tag', 'user_video_title_embedding', 'user_news_category', 'user_news_tag', 'user_news_title_embedding']

    # Spark initialize
    sql_context, spark_session = initial_spark()
    gcsreader = PlanetVideoUser2ItemReader(project_id, sql_context, content_type, content_property, run_time, days=days)
    BASE_FEATURE = [
        'openid', 'uuid', 'gender', 'age', 'user_video_title_embedding', 'user_news_title_embedding','item_title_embedding',
        'user_video_category', 'user_news_category', 'user_video_tag',
        'publish_time', 'site_name', 'cat0_0', 'cat0_1', 'cat1_0', 'cat1_1', 'tags', 'content_ner', 'y', 'title'
        ]

    # Dataset generation scenario
    #   1. 從 GCS 獲取需要使用到的原始資料，包含 event, content, user profile
    #       1-1. 注意：user profile 需要從 daily diff parquet 來 join event，避免偷看問題
    #   2. 建立 positive dataset
    #       2-1. 從用戶行為 (event) 獲取資料作為 positive sample pair
    #       2-2. positive sample pair join user profile (tag, category, title, meta ... etc)
    #       2-3. positive sample pair join content data
    #   3. 建立 negative dataset
    #       3-1. 根據 (2.) 建立的 positive dataset，每筆生成 k 筆 negative sample (k=opt.negative_sample_size)
    #       3-2. 同 (2-2~2-3) , 令 negative data join 對應的 user profile 與 content
    #   4. Integration
    #       4-1. Setup lable `y`
    #       4-2. dataset = positive dataset + negative dataset)

    # 1. get gcs data
    raw_data = get_raw_data_from_gcs()

    # 2. generate positive dataset
    df_positive = raw_data['user_event']
    df_profile_all = None
    for key, df_profile in raw_data.items():
        if key in USER_PROFILE_LIST:
            if not df_profile_all:
                df_profile_all = df_profile
            else:
                df_profile_all = join_event_with_user_profile(df_profile_all, df_profile, how='outer')
    df_positive = join_event_with_user_profile(df_positive, df_profile_all)

    user_profile_col = df_positive.columns

    df_positive = df_positive.join(
        raw_data['item_content'],
        on=[df_positive.uuid == raw_data['item_content'].content_id],
        how='inner')
    df_positive = df_positive.join(
        raw_data['user_meta'],
        on=[df_positive.openid == raw_data['user_meta'].openid],
        how='left').drop(raw_data['user_meta'].openid)

    # 3. generate negative dataset
    df_dedup_positive = dedup_by_col(df_positive, unique_col_base=['openid'], time_col='timestamp')
    df_positive_pandas = df_positive.toPandas()
    df_neg_pair = generate_negative_samples(df_positive_pandas, sample_size=negative_sample_size)

    df_negative = spark_session.createDataFrame(df_neg_pair)

    condition = [df_negative.openid == df_dedup_positive.openid]
    df_negative = df_negative.join(df_dedup_positive[user_profile_col], on=condition, how='left') \
        .drop(df_dedup_positive.openid) \
        .drop(df_dedup_positive.uuid)

    df_negative = df_negative.join(
        raw_data['item_content'],
        on=[df_negative.uuid == raw_data['item_content'].content_id],
        how='left')

    df_negative = df_negative.join(
        raw_data['user_meta'],
        on=[df_negative.openid == raw_data['user_meta'].openid],
        how='left').drop(raw_data['user_meta'].openid)

    # 4. Integration
    df_positive = df_positive.withColumn('y', lit(1))
    df_negative = df_negative.withColumn('y', lit(0))
    dataset = df_positive.union(df_negative)

    # final process
    dataset = dataset.toPandas()
    dataset = parse_planet_category(dataset)
    dataset = dataset[BASE_FEATURE]

    print(f"Len of positive sample: {len(dataset[dataset['y']==1])}")
    print(f"Len of negative sample: {len(dataset[dataset['y']==0])}")
    print(f"Len of dataset: {len(dataset)}")

    # collect distinct label
    distinct_content_parser = ContentDistinctValueParser()
    col2label = distinct_content_parser.parse(raw_data['item_content'],
                                              PlanetVideoUser2ItemConfig.COLS_TO_ENCODE['content'],
                                              add_ner=False)

    if opt.save:
        dump_pickle(f'dataset', dataset, base_path=BASE_PATH)
        dump_pickle(f'col2label', col2label, base_path=BASE_PATH)

    # Upload checkpoint, dataset to GCS
    if opt.upload_gcs:
        upload_to_gcs(
            BASE_PATH,
            f'machine-learning-models-{project_id}',
            f'dataset/planet_{content_type}/user2item/{run_time}'
        )
