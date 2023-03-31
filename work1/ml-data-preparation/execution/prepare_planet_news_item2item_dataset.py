# -*- coding: utf-8 -*-
### Mark: there might be better way than this ###
import sys
sys.path.insert(0, 'datapreparation.zip')

import pyspark.sql.functions as f
from src.gcsreader.planet_news_item2item import PlanetNewsItem2ItemReader
from src.gcsreader.config.planet_configs import PlanetNewsItem2ItemConfig
from src.gcsreader import dedup_by_col
from src.options.planet_news_options import NewsItem2ItemOptions
from utils import initial_spark
from utils import dump_pickle, upload_to_gcs, join_df, parse_view_also_view_score_udf
from utils.parser import ContentDistinctValueParser
from utils.logger import Logger


def pairs_join_and_rename(df_target, df_source, cond_1, drop_1, cond_2, drop_2, config, config_key='', how='left'):

    # join 1
    df_target = join_df(df_target, df_source, on=cond_1, drop=drop_1, how=how)

    # rename
    renamed_cols = config.RENAMED_COLS.get(config_key, {})
    for old_col, new_col in renamed_cols.items():
        df_target = df_target.withColumnRenamed(old_col, new_col)

    # join 2
    df_target = join_df(df_target, df_source, on=cond_2, drop=drop_2, how=how)

    return df_target


def get_raw_data(reader):

    # event data
    df_event = reader.get_event_data()

    # content data
    df_content = reader.get_content_data(config_key='content_pool')

    # view also view
    df_view_also_view = reader.get_view_also_view_data(config_key='view_also_view')

    raw_data = {
        'event': df_event,
        'content': df_content,
        'view_also_view': df_view_also_view
    }
    return raw_data


def get_filtered_data(df_pool, reader_func, config_key='', type_convert_map={}, extra_params={}):

    df = reader_func(df_pool, config_key=config_key, **extra_params)

    for target_col, type in type_convert_map.items():
        df = df.withColumn(target_col, f.col(target_col).cast(type))

    return df


if __name__ == '__main__':

    # define arguments
    opt = NewsItem2ItemOptions().parse()
    project_id = opt.project_id
    run_time = opt.run_time
    content_type = opt.content_type
    content_property = opt.content_property
    days = int(opt.days)
    negative_sample_size = int(opt.negative_sample_size)
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'

    # define logger
    LOGGER = Logger(logger_name=opt.logger_name, dev_mode=True)

    # define reader
    sql_context, spark_session = initial_spark(memory='20G', overhead='15G')
    sql_context.sql("SET spark.sql.autoBroadcastJoinThreshold = -1")  # disable broadcast join
    log4jLogger = sql_context._jvm.org.apache.log4j

    reader = PlanetNewsItem2ItemReader(project_id, sql_context, spark_session, content_type, content_property, run_time, days, LOGGER)

    ### get raw data ###
    raw_data = get_raw_data(reader)

    ### get filtered data: candidates, positive (negative data needs extra handling) ###
    df_candidates = get_filtered_data(raw_data['event'], reader.get_candidates, 'candidate_pool', {'date': 'string'})
    df_positive = get_filtered_data(raw_data['event'], reader.get_positive_data, 'positive_pool', {'date': 'string'})

    ### get negative samples ###
    df_negative = reader.get_negative_data(df_candidates_pool=df_candidates, df_positive_pool=df_positive,
                                           neg_sample_size=negative_sample_size, major_col='uuid',
                                           candidate_col='uuid', time_col='timestamp', prefix='click_')

    # join `name`, `click_name`, `date` and `timestamp` for integrate with positive and debug usage
    LOGGER.info(f'[Data Preparation][Planet News Item2Item] Append debug info to negative data')

    df_candidates_dedup = dedup_by_col(df_candidates, unique_col_base=['uuid'], time_col='timestamp')
    df_negative = pairs_join_and_rename(df_negative, df_candidates_dedup,
                                        cond_1=[df_negative.click_uuid == df_candidates_dedup.uuid], drop_1=[df_candidates_dedup.uuid, df_candidates_dedup.date, df_candidates_dedup.timestamp],
                                        cond_2=[df_negative.uuid == df_candidates_dedup.uuid], drop_2=[df_candidates_dedup.uuid],
                                        config=PlanetNewsItem2ItemConfig, config_key='negative_pool',
                                        how='inner')
    LOGGER.info(f'[Data Preparation][Planet News Item2Item] Negative samples: {df_negative.count()}')
    df_negative.limit(5).show()

    # integrate positive and negative
    LOGGER.info(f'[Data Preparation][Planet News Item2Item] Integrate positive and negative data')

    cols = PlanetNewsItem2ItemConfig.TARGET_COLS.get('all_pool', [])

    df_positive = df_positive.withColumn('y', f.lit(1)).select(cols)
    df_negative = df_negative.withColumn('y', f.lit(0)).select(cols)
    df = df_positive.unionByName(df_negative)

    LOGGER.info(f'[Data Preparation][Planet News Item2Item] Total number of data: {df.count()}')
    df.limit(5).show()

    # join view-also-view score
    LOGGER.info(f'[Data Preparation][Planet News Item2Item] Join view also view scores')
    df = join_df(df, raw_data['view_also_view'], on=['uuid'], how='left')
    df = df.withColumn('view_also_view_score', parse_view_also_view_score_udf()(f.struct(f.col('uuid'),
                                                                                f.col('click_uuid'),
                                                                                f.col('view_also_view_json'),
                                                                                f.col('date'))))

    # join dataset with df_content to get `site_name`, `cat0`, `cat1`, `cat2`, `title_embedding`, `content_ner`, `publish_time` for item1 and item2 respectively
    LOGGER.info(f'[Data Preparation][Planet News Item2Item] Join dataset `click` and `input` data with content data')

    df = pairs_join_and_rename(df, raw_data['content'],
                               cond_1=[df.click_uuid == raw_data['content'].content_id], drop_1=[raw_data['content'].content_id],
                               cond_2=[df.uuid == raw_data['content'].content_id], drop_2=[raw_data['content'].content_id],
                               config=PlanetNewsItem2ItemConfig, config_key='all_pool',
                               how='inner')

    LOGGER.info(f'[Data Preparation][Planet News Item2Item] All dataset: {df.count()}')
    df.limit(5).show()

    ### join all data with view-also-view score ###
    cols = PlanetNewsItem2ItemConfig.TARGET_COLS.get('final_pool', [])
    df = df.select(cols)
    df.limit(5).show()

    ### transform ###
    df_pd = df.toPandas()

    LOGGER.info(f'[Data Preparation][Planet News Item2Item] Positive Data: {len(df_pd.loc[df_pd["y"] == 1])}')
    LOGGER.info(f'[Data Preparation][Planet News Item2Item] Negative Data: {len(df_pd.loc[df_pd["y"] == 0])}')
    LOGGER.info(f'[Data Preparation][Planet News Item2Item] Total Data: {len(df_pd)}')

    ### collect all distinct cat0, cat1, cat2 and site_name ###
    distinct_content_parser = ContentDistinctValueParser()
    col2label = distinct_content_parser.parse(raw_data['content'], reader.config.COLS_TO_ENCODE['content_pool'], add_ner=True)

    ### dump data ###
    if opt.save:
        dump_pickle(f'dataset', df_pd, base_path=BASE_PATH)
        dump_pickle(f'col2label', col2label, base_path=BASE_PATH)

    if opt.upload_gcs:
        upload_to_gcs(
            BASE_PATH,
            f'machine-learning-models-{project_id}',
            f'dataset/{content_type}/item2item/{run_time}'
        )
