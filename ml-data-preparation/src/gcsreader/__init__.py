from abc import *
from pyspark.sql.functions import *
from pyspark.sql.types import StringType
from pyspark.sql import Window
from datetime import datetime, timedelta
from utils import DAY_STRING_FORMAT, get_partition_path_with_blobs_check


class GcsReader(metaclass=ABCMeta):

    def __init__(self, project_id, sql_context, content_type, property_name, run_time, days):
        self.project_id = project_id
        self.sql_context = sql_context
        self.content_type = content_type
        self.property_name = property_name
        self.days = days
        self.run_time = run_time

    @abstractmethod
    def get_event_data(self):
        pass

    @abstractmethod
    def get_content_data(self):
        pass

    def _extract_date(self, prefix, data):
        """extract date information for user profile data"""
        offset = len(prefix)
        date_offset = len('yyyymmdd')
        return udf(lambda x: x[x.find(f'{prefix}')+offset:x.find(f'{prefix}')+offset+date_offset], StringType())(data)

    def _user_profile_handler(self, input_path, profile_type='category', basepath_checkpoint=''):
        base_path = input_path[:input_path.find(basepath_checkpoint)]
        df = self.sql_context.read.option('basePath', base_path).parquet(*[input_path])
        df = df.withColumn('date', input_file_name())
        df = df.withColumn('date', self._extract_date(f'{profile_type}/', df.date))
        return df

    def _add_userid(self, df, main_col='openid', sub_col='trackid'):
        condition_col = ((col(main_col).isNull()))
        df = df.withColumn('userid', when(condition_col, col(sub_col)).otherwise(col(main_col)))
        df = df.filter(df.userid.isNotNull())
        return df

    def _get_existed_blobs(self, input_path, prefix, bucket_idx=2, days=None):
        if not days:
            days = self.days
        all_path_list = []

        for d in range(int(days)):
            current_date = (datetime.strptime(self.run_time, DAY_STRING_FORMAT) - timedelta(days=d)).strftime(DAY_STRING_FORMAT)
            cur_path_list, _ = get_partition_path_with_blobs_check(
                input_path.replace('INPUT_DATE', current_date),
                prefix.replace('INPUT_DATE', current_date),
                bucket_idx=bucket_idx
            )
            all_path_list.extend(cur_path_list)

        return all_path_list


def dedup_by_col(df, unique_col_base=['content_id'], time_col='update_time'):
    w = Window.partitionBy([col(c) for c in unique_col_base]).orderBy(col(time_col).desc())
    # filter latest log
    df = df.withColumn("rank", row_number().over(w)).filter(col("rank") == 1).drop("rank")
    df = df.dropDuplicates(subset=unique_col_base)
    return df


def join_event_with_user_profile(df_left, df_profile, cond=[], how='left'):

    if cond == []:
        cond = [df_left.userid == df_profile.userid, df_left.date == df_profile.date]

    df = df_left.join(df_profile, on=cond, how=how) \
        .drop(df_profile.userid) \
        .drop(df_profile.date)
    return df
