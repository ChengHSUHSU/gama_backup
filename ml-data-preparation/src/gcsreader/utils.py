import json
import random
import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, ArrayType
from datetime import datetime, timedelta
from utils import DAY_STRING_FORMAT, HOUR_STRING_FORMAT, HOUR_ONLY_STRING_FORMAT, get_partition_path_with_blobs_check


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


def info_json_map_transform(info, filter_info, content_id_key, filter_key=None, filter_value_list=None):
    content_id = ''
    if info:
        if content_id_key in json.loads(info):
            if not filter_key:  # no filter
                content_id = json.loads(info)[content_id_key]
            else:  # need to filter by field value
                if filter_key in json.loads(filter_info):
                    filter_content = json.loads(filter_info).get(filter_key)
                    # filter_content might be list (ex. ['時事星']) or value (ex. 'pop')
                    if isinstance(filter_content, list):
                        for event_filter_value in filter_content:
                            if event_filter_value in filter_value_list:
                                content_id = json.loads(info)[content_id_key]
                                break
                    else:
                        if filter_content in filter_value_list:
                            content_id = json.loads(info)[content_id_key]
    return content_id


def info_array_of_json_map_transform(info, filter_info, content_id_key, filter_key=None, filter_value_list=None):
    content_id = []
    if info:
        for json_obj in json.loads(info):
            # prevent 'uuid' not in {'name': '', 'pos': 18, 'sec': 'news', 'type': 'video'}
            if not filter_key and content_id_key in json_obj:  # no filter
                content_id.append(json_obj[content_id_key])
            elif content_id_key in json_obj:  # need to filter by field value
                if filter_info == info:
                    # if filter_info is impression_info
                    if filter_key in json_obj:
                        # prevent 'uuid' not in {'name': '', 'pos': 18, 'sec': 'news', 'type': 'video'}
                        if json_obj[filter_key] in filter_value_list and content_id_key in json_obj:
                            content_id.append(json_obj[content_id_key])
                else:
                    # filter_info is click_info or page_info
                    if filter_key in json.loads(filter_info):
                        filter_content = json.loads(filter_info).get(filter_key)
                        # filter_content might be list (ex. ['時事星']) or value (ex. 'pop')
                        if isinstance(filter_content, list):
                            for event_filter_value in filter_content:
                                if event_filter_value in filter_value_list:
                                    content_id.append(json_obj[content_id_key])
                                    break
                        else:
                            if filter_content in filter_value_list:
                                content_id.append(json_obj[content_id_key])
    return content_id


def udf_info_json_map_transaction(content_id_key, filter_key, filter_value_list):
    return f.udf(lambda col, filter_col: info_json_map_transform(col, filter_col, content_id_key, filter_key, filter_value_list), StringType())


def udf_info_array_of_json_map_transaction(content_id_key, filter_key, filter_value_list):
    return f.udf(lambda col, filter_col: info_array_of_json_map_transform(col, filter_col, content_id_key, filter_key, filter_value_list), ArrayType(StringType()))


def udf_get_candidate_arms(negative_size, neg_candidate_col='uuids', pos_id_col='uuid'):

    def _get_candidate_arms(struct):

        positive_id = struct[pos_id_col]
        negative_candidates = struct[neg_candidate_col]
        negative_candidates.remove(positive_id)
        output = [positive_id]

        if len(negative_candidates) < negative_size:
            output.extend(negative_candidates)
        else:
            output.extend(random.sample(negative_candidates, k=negative_size))

        return output

    return f.udf(_get_candidate_arms, ArrayType(StringType()))


def udf_extract_from_path(prefix: str, target_len: int, offset: int = 0) -> StringType:
    """pyspark user defined function to extrac information from given path

    Args:
        prefix (str): target path prefix
        target_len (int): target information length
        offset (int, optional): offset shift. Defaults to 0.

    Returns:
        StringType
    """

    prefix_offset = len(prefix)

    def _extract_from_path(path):

        prefix_pos = path.find(prefix)

        start_pos = prefix_pos + prefix_offset + offset
        end_pos = prefix_pos + prefix_offset + target_len + offset

        return path[start_pos:end_pos]

    return f.udf(_extract_from_path, StringType())


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


def parse_impression_info_array(df: DataFrame) -> DataFrame:
    """ Parse impression info (array in json string),
        then explode it to multiple rows with dict in json string "{key: value}"
        Args:
            df: filtered event_daily / event_hourly dataframe
            output_cols: column names selected to output

        Returns: Dataframe
            columns [`output_col`]
     """

    df = df.withColumn('impression_info_array', f.from_json(f.col('impression_info'), ArrayType(StringType())))
    df = df.withColumn('impression_info_array', f.explode_outer(f.col('impression_info_array')))
    return df


def get_hourly_event_paths(run_time: str, input_path: str, time_range: int=7) -> list:
    """ Function to get hourly event paths.
    for example:
        run_time: 2022032012 (HOUR_STRING_FORMAT)
        input_path: 'gs://event-event-bf-data-uat-001/event_hourly/date=INPUT_DATE/hour=INPUT_HOUR/
                     property=beanfun/is_page_view=*/event=*/*.parquet',
        time_range: 24 (hours)

    Args:
        run_time: target base date for parsing event data
        input_path: target event folder on gcs. only support `event_hourly`
        time_range: target time range

    Returns: file_locations
    """

    # convert into datatime-format
    cur_time = datetime.strptime(run_time, HOUR_STRING_FORMAT)

    file_locations = []
    for _ in range(int(time_range)):
        # duplicate
        input_path_ = input_path
        cur_date = cur_time.strftime(DAY_STRING_FORMAT)
        cur_hour = cur_time.strftime(HOUR_ONLY_STRING_FORMAT)
        prefix = f'event_hourly/date={cur_date}/hour={cur_hour}'
        file_loc = input_path_.replace('INPUT_DATE', cur_date).replace('INPUT_HOUR', cur_hour)
        file_loc, _ = get_partition_path_with_blobs_check(file_loc, prefix, bucket_idx=2)

        file_locations.extend(file_loc)
        cur_time -= timedelta(hours=1)
    return file_locations
