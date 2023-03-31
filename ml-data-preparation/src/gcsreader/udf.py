import json
import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, ArrayType
from datetime import datetime, timedelta


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


def udf_info_array_of_json_map_transaction(content_id_key: str, filter_key: str, filter_value_list: list) -> DataFrame:
    return f.udf(lambda x, y: info_array_of_json_map_transform(x, y, content_id_key, filter_key, filter_value_list), ArrayType(StringType()))


def info_array_of_json_map_transform(info, filter_info, content_id_key, filter_key=None, filter_value_list=None):
    content_id = []
    if info:
        for json_obj in json.loads(info):
            # prevent 'uuid' not in {'name': '', 'pos': 18, 'sec': 'news', 'type': 'video'}
            if not filter_key and content_id_key in json_obj:  # no filter
                content_id.append(json_obj[content_id_key])
            else:  # need to filter by field value
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
                        else:
                            if filter_content in filter_value_list:
                                content_id.append(json_obj[content_id_key])
    return content_id
