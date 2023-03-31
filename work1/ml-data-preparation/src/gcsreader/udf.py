import json
import random
import numpy as np
from datetime import datetime, timedelta

import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, ArrayType

from utils import DAY_STRING_FORMAT


def udf_get_date_from_timestamp(offset: int = 1) -> StringType:
    """pyspark user defined function to set user profile join date

    Args:
        offset (int, optional): Offset for date. Defaults to 1.

    Returns:
        StringType
    """

    def _get_date_from_timestamp(timestamp):

        date = (datetime.fromtimestamp(int(timestamp)) - timedelta(days=offset)).strftime(DAY_STRING_FORMAT)

        return date

    return f.udf(_get_date_from_timestamp, StringType())


def udf_set_user_profile_join_date(day_ranges: int = 3) -> StringType:
    """pyspark user defined function to set user profile join date

    Args:
        day_ranges (int, optional): Latest user profile date to join event. Defaults to 3.

    Returns:
        StringType
    """

    def set_date(date):
        return (datetime.strptime(date, '%Y%m%d') + timedelta(days=day_ranges)).strftime('%Y%m%d')

    return f.udf(set_date, StringType())


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


def udf_info_array_of_json_map_transaction(content_id_key, filter_key, filter_value_list):
    return f.udf(lambda col, filter_col: info_array_of_json_map_transform(col, filter_col, content_id_key, filter_key, filter_value_list), ArrayType(StringType()))


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


def udf_info_json_map_transaction(content_id_key, filter_key, filter_value_list):
    return f.udf(lambda col, filter_col: info_json_map_transform(col, filter_col, content_id_key, filter_key, filter_value_list), StringType())
