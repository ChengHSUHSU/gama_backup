

import json
import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, ArrayType
from datetime import datetime, timedelta
import os
import sys
sys.path.insert(0, 'datapreparation.zip')


def initial_spark(app_name='metrics-calculator', cores='4', memory='20G', overhead='4G', pivot_max='10000',
                  shuffle_partitions='200', dynamic_allocate='true', num_executors='2'):
    """ Initial spark setting """
    from pyspark.sql import SparkSession
    from pyspark.sql import SQLContext
    spark_session = SparkSession.builder \
        .master('yarn') \
        .appName(app_name) \
        .config('spark.executor.cores', cores) \
        .config('spark.executor.memory', memory) \
        .config('spark.executor.memoryOverhead', overhead) \
        .config('spark.sql.pivotMaxValues', pivot_max) \
        .config('spark.sql.shuffle.partitions', shuffle_partitions) \
        .config('spark.dynamicAllocation.enabled', dynamic_allocate) \
        .config('spark.executor.instances', num_executors) \
        .getOrCreate()

    sc = spark_session.sparkContext
    sc._jsc.hadoopConfiguration() \
        .set('mapreduce.fileoutputcommitter.marksuccessfuljobs', 'false')
    sql_context = SQLContext(sc)
    return sql_context, spark_session



import json
import pyspark.sql.functions as f

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


def test_udf_info_array_of_json_map_transaction():
    import json
    import pyspark.sql.functions as f
    from pyspark.sql.types import StringType, ArrayType, StructType, StructField
    sql_context, spark_session = initial_spark()
    impression_info_data = [{'cat1': ['a'], 'name': ['b'], 'tag': ['c', 'd'], 'uuid': '1623583360515313664'}, 
                            {'cat1': ['e'], 'name': ['f'], 'tag': ['g', 'h'], 'uuid': '6456456546546'}]
    impression_info_data = json.dumps(impression_info_data)
    page_info_data = json.dumps({"cat": "\u63a8\u85a6", "page": "planet", "tab": "\u6642\u4e8b\u661f"})
    schema = StructType([StructField('impression_info', StringType()),
                         StructField('page_info', StringType())])
    df = spark_session.createDataFrame([(impression_info_data, page_info_data)],
                                        schema=schema)
    df.show(10, False)

    df_result = df.withColumn('uuids',
                             udf_info_array_of_json_map_transaction('uuid', None, [])(f.col('impression_info'), 
                                                                                      f.col('page_info')))
    df_result.show(10, False)


    uuids_data = ['1623583360515313664', '6456456546546']
    schema = StructType([StructField('impression_info', StringType()),
                         StructField('page_info', StringType()),
                         StructField('uuids', ArrayType(StringType()))])

    df_expect = spark_session.createDataFrame([(impression_info_data, page_info_data, uuids_data)],
                                        schema=schema)
    
    df_expect.show(10, False)

    dat = df_result.collect()
    print(dat)
    dat = df_expect.collect()
    print(dat)
    assert df_result.collect() == df_expect.collect()



def test_info_array_of_json_map_transaction():
    impression_info_data = [{'cat1': ['a'], 'name': ['b'], 'tag': ['c', 'd'], 'uuid': '1623583360515313664'}, 
                            {'cat1': ['e'], 'name': ['f'], 'tag': ['g', 'h'], 'uuid': '6456456546546'}]
    impression_info_data = json.dumps(impression_info_data)
    page_info_data = json.dumps({"cat": "\u63a8\u85a6", "page": "planet", "tab": "\u6642\u4e8b\u661f"})

    content_ids = info_array_of_json_map_transform(impression_info_data, page_info_data, 'uuid', None, [])
    content_ids_expect = ['1623583360515313664', '6456456546546']
    assert content_ids == content_ids_expect


### TEST
'''

impression_info->
[{"cat1": ["\u653f\u6cbb\u8a71\u984c"], 
"name": "\u540d\u5bb6\u8ad6\u58c7\u300b\u66fe\u5efa\u5143\uff0f\u9999\u6e2f\u570b\u5b89\u98a8\u66b4\u4e0b\u81fa\u6e2f\u95dc\u4fc2\u7684\u65b0\u958b\u59cb", 
"pos": 1,
 "provider": "nownews", 
 "sec": "news", 
 "tag": ["\u540d\u5bb6\u8ad6\u58c7", "\u66fe\u5efa\u5143", "\u9999\u6e2f", "\u53f0\u7063", "\u6e2f\u53f0", "l-\u9999\u6e2f", "l-\u53f0\u7063"],
"type": "news", "uuid": "1623583360515313664"},
.....]

page_info
{"cat": "\u63a8\u85a6", "page": "planet", "tab": "\u6642\u4e8b\u661f"}

uuids:
[52395, ...]
'''


test_info_array_of_json_map_transaction()