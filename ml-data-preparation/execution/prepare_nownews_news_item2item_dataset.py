# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'datapreparation.zip')

from src.options.nownews_news_option import NowNewsNewsPreparationOptions
from pyspark.sql.functions import lit

from utils import initial_spark, dump_pickle, upload_to_gcs
from utils.parser import ContentDistinctValueParser
from src.gcsreader.config.nownews_config import NownewsNewsItem2ItemConfig
from src.gcsreader.nownews_news_item2item import NownewsNewsItem2ItemReader
from src.negative_sampler import generate_negative_samples


def get_raw_data_from_gcs():

    df_event = gcsreader.get_event_data(userid='trackid')
    df_content = gcsreader.get_content_data()
    df_view_also_view = gcsreader.get_view_also_view_data()

    condition = [df_content.content_id == df_view_also_view.content_uuid]

    df_content = df_content.join(
        df_view_also_view.select(['content_uuid', 'view_also_view_json']),
        on=condition, how='left'
    ).drop(df_view_also_view.content_uuid)

    raw_data = {
        'user_event': df_event,
        'item_content': df_content
    }

    for data_name, column_to_rename_list in NownewsNewsItem2ItemConfig.COLUMN_TO_RENAME.items():
        for renamed_tuple in column_to_rename_list:
            raw_data[data_name] = raw_data[data_name].withColumnRenamed(renamed_tuple[0], renamed_tuple[1])

    return raw_data


def join_content_data(df_positive, df_content):
    # TODO: Integrate similar function after planet_news_item2item finished.
    base_content_columns = df_content.columns

    # join page content
    page_content = df_content
    for col_name in base_content_columns:
        page_content = page_content.withColumnRenamed(col_name, f'page_{col_name}')

    condition = [df_positive.page_uuid == page_content.page_content_id]
    df_positive = df_positive.join(page_content, on=condition, how='left').drop('page_content_id')

    # join click content
    click_content = df_content
    for col_name in base_content_columns:
        click_content = click_content.withColumnRenamed(col_name, f'click_{col_name}')

    condition = [df_positive.click_uuid == click_content.click_content_id]
    df_positive = df_positive.join(click_content, on=condition, how='left').drop('click_content_id')
    return df_positive


if __name__ == '__main__':
    opt = NowNewsNewsPreparationOptions().parse()
    project_id = opt.project_id
    run_time = opt.run_time
    days = int(opt.days)
    negative_sample_size = int(opt.negative_sample_size)
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'

    ENABLE_POSITIVE_SAMPLING = opt.enable_positive_sampling
    DAILY_POSITIVE_SAMPLE_SIZE = opt.daily_positive_sample_size
    DAILY_SAMPLE_SEED = opt.daily_sample_seed

    # spark initialize
    sql_context, spark_session = initial_spark()
    gcsreader = NownewsNewsItem2ItemReader(project_id, sql_context, run_time, days=days)

    # 1. get gcs data
    raw_data = get_raw_data_from_gcs()
    if ENABLE_POSITIVE_SAMPLING:
        df_pd = raw_data['user_event'].toPandas()
        df_pd = df_pd.groupby('date').sample(n=DAILY_POSITIVE_SAMPLE_SIZE, random_state=DAILY_SAMPLE_SEED)
        raw_data['user_event'] = spark_session.createDataFrame(df_pd)

    # 2. generate positive dataset
    df_positive = join_content_data(
        raw_data['user_event'].select('page_uuid', 'click_uuid'),
        raw_data['item_content'])

    # 3. generate negative dataset
    df_positive_pandas = df_positive.select('page_uuid', 'click_uuid', 'page_publish_time') \
        .toPandas().dropna(subset=['page_publish_time'])

    df_neg_pair = generate_negative_samples(
        df_positive_pandas,
        sample_size=negative_sample_size,
        major_col='page_uuid', candidate_col='click_uuid', time_col='page_publish_time',
        enable_distinct_major=False
    )

    df_negative = spark_session.createDataFrame(df_neg_pair)
    df_negative = join_content_data(df_negative, raw_data['item_content'])

    # 4. integration
    df_positive = df_positive.withColumn('y', lit(1))
    df_negative = df_negative.withColumn('y', lit(0))
    dataset = df_positive.unionByName(df_negative, allowMissingColumns=False)

    # final process
    dataset = dataset.toPandas()

    print(f"Len of positive sample: {len(dataset[dataset['y']==1])}")
    print(f"Len of negative sample: {len(dataset[dataset['y']==0])}")
    print(f"Len of dataset: {len(dataset)}")

    # collect distinct label
    distinct_content_parser = ContentDistinctValueParser()
    col2label = distinct_content_parser.parse(raw_data['item_content'],
                                              NownewsNewsItem2ItemConfig.COLS_TO_ENCODE['content'],
                                              add_ner=False)

    dump_pickle(f'dataset', dataset, base_path=BASE_PATH)
    dump_pickle(f'col2label', col2label, base_path=BASE_PATH)

    # Upload checkpoint, dataset to GCS
    if opt.upload_gcs:
        upload_to_gcs(
            BASE_PATH,
            f'machine-learning-models-{project_id}',
            f'dataset/nownews_news/item2item/{run_time}'
        )
