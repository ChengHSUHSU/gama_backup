from google.cloud import storage
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import StringType, FloatType, MapType
from pyspark.sql.window import Window
from datetime import datetime, timedelta
from typing import Union, Tuple
import numpy as np
import pyspark.sql.functions as f
import json
import ast
import pickle
import os

DAY_STRING_FORMAT = '%Y%m%d'
HOUR_STRING_FORMAT = '%Y%m%d%H'
HOUR_ONLY_STRING_FORMAT = '%H'
SPARK_APP_NAME = 'data-preparation'


def initial_spark(app_name=SPARK_APP_NAME, cores='4', memory='12G', overhead='10G', shuffle_partitions='200', num_executors='3', add_project_path_to_executor=True):
    """ Initial spark setting """
    spark_session = SparkSession.builder \
        .master('yarn') \
        .appName(app_name) \
        .config('spark.executor.cores', cores) \
        .config('spark.executor.memory', memory) \
        .config('spark.executor.memoryOverhead', overhead) \
        .config('spark.executor.instances', num_executors) \
        .config('spark.sql.autoBroadcastJoinThreshold', '-1') \
        .config('spark.sql.shuffle.partitions', shuffle_partitions) \
        .getOrCreate()

    if add_project_path_to_executor:
        spark_session.sparkContext.addPyFile('datapreparation.zip')

    sc = spark_session.sparkContext
    sc._jsc.hadoopConfiguration() \
        .set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    sql_context = SQLContext(sc)
    return sql_context, spark_session


def start_to_end_date(start_date_string, end_date_string):
    # if start_date is empty string, set as yesterday
    if start_date_string == '':
        start_date = (datetime.utcnow() - timedelta(days=1)) \
            .strftime(DAY_STRING_FORMAT)
    else:
        datetimeObj = datetime.strptime(start_date_string, DAY_STRING_FORMAT)
        start_date = datetimeObj.strftime(DAY_STRING_FORMAT)

    # if end_date is empty string, set as start_date
    if end_date_string == '':
        end_date = start_date
    else:
        datetimeObj = datetime.strptime(end_date_string, DAY_STRING_FORMAT)
        end_date = datetimeObj.strftime(DAY_STRING_FORMAT)

    start_ts = datetime.strptime(start_date, DAY_STRING_FORMAT)
    end_ts = datetime.strptime(end_date, DAY_STRING_FORMAT)
    date_list = []
    for n in range(int((end_ts - start_ts).days) + 1):
        date_list.append((start_ts + timedelta(n)).strftime(DAY_STRING_FORMAT))
    return date_list


def check_path_exists(path):
    sql_context, spark_session = initial_spark()
    sc = spark_session.sparkContext
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(
        sc._jvm.java.net.URI.create(path),
        sc._jsc.hadoopConfiguration(),
    )
    return fs.exists(sc._jvm.org.apache.hadoop.fs.Path(path))


def get_partition_path_with_check(input_path, run_time: str, days: str):
    path_list = []
    path_list_not_exist = []
    for i in range(int(days)):
        current_date = (datetime.strptime(
            run_time, DAY_STRING_FORMAT) - timedelta(days=i)).\
            strftime(DAY_STRING_FORMAT)
        path = input_path.replace('INPUT_DATE', current_date)
        if check_path_exists(path):
            path_list.append(path)
        else:
            path_list_not_exist.append(path)
    return path_list, path_list_not_exist


def get_partition_path_with_blobs_check(path, prefix, bucket_idx=2):
    """Function to check if give path exists in gcs

    :param path: full path of target file in gcs
    :type path: str
    :param prefix: prefix to be used for list_blobs
    :type prefix: str
    :param bucket_idx: index to indicate bucket name location while splitting path by separator
    :type buckect_idx: int
    :param separator: separator to be used to split path to get bucket name
    :type separator: str
    """
    path_list = []
    path_list_not_exist = []
    bucket_name = path.split('/')[bucket_idx]
    storage_client = storage.Client()

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    if len(list(blobs)) > 0:
        path_list.append(path)
    else:
        path_list_not_exist.append(path)

    return path_list, path_list_not_exist


def read_pickle(fn, base_path="./datasets"):
    with open(f'{base_path}/{fn}', 'rb') as handle:
        try:
            return pickle.load(handle)
        except Exception as err:
            print('fail to load pickle file.')
            return None


def dump_pickle(file_name, data, base_path=''):
    mkdir(base_path)
    with open(f'{base_path}/{file_name}.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def _upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def upload_to_gcs(
        base_path,
        bucket_name='machine-learning-models-bf-data-uat-001',
        destination_path='r12n/planet_news'
):

    BASE_PATH = base_path
    experiment_files = os.listdir(BASE_PATH)
    for FILE_NAME in experiment_files:
        try:
            source_file_name = f'{BASE_PATH}/{FILE_NAME}'
            destination_blob_name = f'{destination_path}/{source_file_name}'
            _upload_blob(bucket_name, source_file_name, destination_blob_name)
            print(f'File {source_file_name} uploaded to {destination_blob_name} success.')
        except Exception as e:
            print(f'File {source_file_name} uploaded fail.')
            raise e


def join_df(df_left, df_right, on=[], drop=[], how='inner'):
    """Function to join to pyspark DataFrame

    :param df_left: DataFrame 1
    :type df_left: pyspark DataFrame
    :param df_right: DataFrame 2
    :type df_right: pyspark DataFrame
    :param on: columns to be used for join
    :type on: list of string or pyspark column
    :param drop: columns to be dropped after join
    :type on: list of pyspark column
    """

    if not isinstance(on, list):
        raise TypeError(f'on must be a list, but get {type(on)}')

    if not isinstance(drop, list):
        raise TypeError(f'drop must be a list, but get {type(drop)}')

    df = df_left.join(df_right, on=on, how=how)

    for col in drop:
        df = df.drop(col)

    return df


def convert_type(x):
    """Function to convert col values by ast.literal_eval or json.loads (ex: embedding: '[1,2,....,0]' -> [1,2,...,0])
    """
    def _ast_convert_type(x):
        if x is not None:
            return ast.literal_eval(x)
        return x

    def _json_convert_type(x):
        # json.loads is faster for embedding
        if x is not None:
            return json.loads(x)
        return x

    # try json.loads first cause is faster for embedding
    try:
        return _json_convert_type(x)
    except:
        return _ast_convert_type(x)


def parse_view_also_view_score_udf(uuid_col='uuid', click_uuid_col='click_uuid', view_also_view_col='view_also_view_json', date_col='date'):

    def parse_view_also_view_score(struct_data):

        uuid = struct_data[uuid_col]
        click_uuid = struct_data[click_uuid_col]
        view_also_view_json = struct_data[view_also_view_col]
        date = struct_data[date_col]

        if view_also_view_json:
            view_also_view_json = view_also_view_json.get(uuid, {})

            for vv_date, vv_json in view_also_view_json.items():
                if int(vv_date) <= int(date) and vv_json.get(click_uuid, 0.0):
                    return vv_json.get(click_uuid)

        return 0.0

    return f.udf(parse_view_also_view_score, FloatType())


def aggregate_view_also_view_udf(uuid_col='uuid', date_col='date', view_also_view_col='view_also_view_json'):

    def aggregate_view_also_view(struct_data):

        uuid = struct_data[uuid_col]
        view_also_view_list = struct_data[view_also_view_col]

        output = {uuid: {}}

        for vv in view_also_view_list:

            d = vv[date_col]
            vv_json = vv[view_also_view_col]

            output[uuid][d] = vv_json

        return output

    return f.udf(aggregate_view_also_view, MapType(StringType(), MapType(StringType(), MapType(StringType(), FloatType()))))


def sample_positive_data(df, daily_positive_sample_size, random_seed):

    if bool(daily_positive_sample_size):
        df = df.withColumn('rand_val', f.rand(seed=random_seed))
        window_spec = Window.partitionBy('date').orderBy('rand_val')
        df = df.withColumn('rand_val', f.row_number().over(window_spec)).filter(f.col('rand_val') <= daily_positive_sample_size).drop('rand_val')
    return df


def filter_by_interaction_count(df, primary_col='userid', requisite_sequence_length=0):

    if bool(requisite_sequence_length):

        behavior_count = df.groupBy(primary_col).agg(f.count(primary_col).alias('count'))
        behavior_count = behavior_count.filter(f.col('count') > requisite_sequence_length)
        df = df.join(behavior_count, on=[primary_col], how='inner')

    return df


def prune_preference_user_profile(data: str, tag_type: Union[str, list], condition: Tuple[str, float]) -> str:
    """Function to pop out certain tag/cat preference in the given profile if 
    the score is lower than threshold defined in condition.

    Args:
        data (str): preference user profile to be pruned in json string. (ex: category profile, tag profile, etc)
        tag_type (Union[str, list]): first level key of the given user profile. (ex: 'editor' of nownewsNewsTagProfile)
        condition (Tuple[str, float]): filtering conditions in the format of (key, threshold)

    Returns:
        json str: pruned preference profile
    """

    profile = {}
    if isinstance(tag_type, str):
        tag_type = [tag_type]

    for type_key in tag_type:
        tag_data = json.loads(data).get(type_key, {})

        if bool(tag_data):
            for tagging_label, statistics_data in tag_data.items():
                for tag_name, tag_statistics in statistics_data.copy().items():
                    if tag_statistics[condition[0]] < condition[1]:
                        tag_data[tagging_label].pop(tag_name, None)

        profile[type_key] = tag_data

    return json.dumps(profile, ensure_ascii=False)


def calculate_similarity_score(vec_a: str, vec_b: str) -> float:
    """function to calculate similarity of vector a and b

    Args:
        vec_a (str)
        vec_b (str)

    Returns:
        float: dot-product similarity score
    """

    if vec_a is None or vec_b is None:
        return 0.0

    if isinstance(vec_a, str):
        vec_a = json.loads(vec_a)
    if isinstance(vec_b, str):
        vec_b = json.loads(vec_b)

    return float(np.dot(vec_a, vec_b))
