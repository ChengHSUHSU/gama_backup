import json
import random
from datetime import datetime, timedelta

import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, FloatType, StringType

from src.gcsreader import dedup_by_col
from src.gcsreader.udf import udf_extract_from_path
from utils import DAY_STRING_FORMAT, get_partition_path_with_blobs_check, calculate_similarity_score


def get_value_from_json_col(df, key='uuid', alias_key='uuid', column_name='page_info'):

    df = df.withColumn(alias_key, f.get_json_object(f.col(column_name), f'$.{key}'))

    return df


def filter_dataframe(df_source, config_key, config):

    events = getattr(config, 'TARGET_EVENT', {}).get(config_key, [])
    infos = getattr(config, 'TARGET_INFO', {}).get(config_key, {})
    cols = getattr(config, 'TARGET_COLS', {}).get(config_key, [])
    rules = getattr(config, 'FILTER_CONDITIONS', {}).get(config_key, {})

    # get target event
    if len(events) > 0:
        df = df_source.filter(f.col('event').isin(events))
    else:
        df = df_source

    # parse information from target info (in json format)
    for info, key_pairs in infos.items():
        for (key, alias_key) in key_pairs:
            df = get_value_from_json_col(df, key=key, alias_key=alias_key, column_name=info)

    # filtering
    for col, values in rules.items():
        df = df.filter(f.col(col).isin(values))

    if len(cols) > 0:
        df = df.select(cols)

    return df


def extend_generated_time(df: DataFrame, prefix_name: str = 'category', time_format: str = 'date') -> DataFrame:
    """Function to extend generated time from file path in gcs

    Args:
        df (DataFrame)
        prefix_name (str, optional): target path prefix name. Defaults to 'category'.
        time_format (str, optional): support `date` (ex. 20220101) and `hour` format (ex. 20220101/01). Defaults to 'date'.

    Returns:
        DataFrame
    """

    prefix = f'{prefix_name}/'
    date_len = len('yyyymmdd')
    hour_len = len('hh')
    hour_offset = date_len + 1  # +1 is for / symbol after yyyymmdd

    df = df.withColumn('input_path', f.input_file_name())

    if time_format == 'date':
        df = df.withColumn('date', udf_extract_from_path(prefix, date_len)(f.col('input_path')))

    if time_format == 'hour':
        df = df.withColumn('date', udf_extract_from_path(prefix, date_len)(f.col('input_path')))
        df = df.withColumn('hour', udf_extract_from_path(prefix, hour_len, hour_offset)(f.col('input_path')))

    return df


def filter_candidate(df: DataFrame, top_k:int, sort_col: str, dedup_unique_col: list, select_col: list) -> DataFrame:
    """Function to Filter and sort candidates

    Args:
        df (DataFrame)
        top_k (int): get top k candidates
        sort_col (str): sort field
        dedup_unique_col (list): dedup unique col
        select_col (list): fields to display

    Returns:
        DataFrame
    """
    df = dedup_by_col(df, unique_col_base=dedup_unique_col, time_col=sort_col)
    df = df.orderBy([f.col(sort_col)], ascending=False).limit(top_k)
    df = df.select(select_col)

    return df


def get_existed_blobs(run_time: str, days: int, input_path: str, prefix: str, bucket_idx: str = 2, date_column: int = 'INPUT_DATE') -> list:
    """Function to get exists blobs in gcs

    Args:
        run_time (str): start time for parsing daily data (ex: 20221019)
        days (int): data range would be run_time ~ run_time - days
        input_path (str): full path of target file in gcs
        prefix (str): prefix of blob path, for example the event path prefix would be `event_daily/date=yyyymmdd/property=PROPERTY_NAME`
        bucket_idx (str, optional): index to indicate bucket name location while splitting path by separator. Defaults to 2.
        date_column (int, optional): Defaults to 'INPUT_DATE'.

    Returns:
        list
    """

    all_path_list = []

    for d in range(int(days)):
        current_date = (datetime.strptime(run_time, DAY_STRING_FORMAT) - timedelta(days=d)).strftime(DAY_STRING_FORMAT)
        cur_path_list, _ = get_partition_path_with_blobs_check(input_path.replace(date_column, current_date),
                                                               prefix.replace(date_column, current_date),
                                                               bucket_idx=bucket_idx)
        all_path_list.extend(cur_path_list)

    return all_path_list


def rename_cols(df: DataFrame, rename_cols: list) -> DataFrame:
    """Function to rename dataframe columns

    Args:
        df (DataFrame)
        rename_cols (list): column list to rename

    Returns:
        DataFrame
    """

    for old_col, new_col in rename_cols:
        df = df.withColumnRenamed(old_col, new_col)

    return df


def extend_similar_score_field(df: DataFrame, with_col_name: str, cal_embedding_col_names: list) -> DataFrame:
    """Function to extend similar score field

    Args:
        df (DataFrame)
        with_col_name (str): extend similar score field name
        cal_embedding_col_names (list): The name of the field that needs to use embedding

    Returns:
        DataFrame
    """
    df = df.withColumn(with_col_name,
                       f.udf(lambda x, y: calculate_similarity_score(x, y), FloatType())
                       (f.col(cal_embedding_col_names[0]), f.col(cal_embedding_col_names[1])))

    return df
