from pyspark.sql.types import StringType, ArrayType
from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as f
import random


def negative_sampler_udf(sample_size: int, content_data: dict, random_sample_retries: int = 0) -> ArrayType(StringType()):
    """UDF for negative sampling

    Args:
        sample_size (int): negative samples size
        content_data (dict): content data to publish time mapping
        random_sample_retries (int, optional): number of retries for one negative sampling iteration. Defaults to 0.
    """
    def negative_sampler(positive_id_list, positive_publish_time_list):

        negative_samples = []
        content_pool = content_data.items()
        maximum_timestamp = sorted(positive_publish_time_list)[-1]
        num_positive = len(positive_id_list)

        samples = []
        for _ in range(num_positive):

            for _ in range(sample_size):
                retries = 0

                picked_item, picked_item_timestamp = random.sample(content_pool, 1)[0]

                condition = (picked_item in positive_id_list or picked_item in samples or picked_item_timestamp > maximum_timestamp)

                while condition:

                    if retries > random_sample_retries:
                        break
                    else:
                        retries += 1

                    picked_item, picked_item_timestamp = random.sample(content_pool, 1)[0]
                    condition = (picked_item in positive_id_list or picked_item in samples or picked_item_timestamp > maximum_timestamp)

                if not condition:
                    samples.append(picked_item)

        negative_samples += samples

        return negative_samples

    return f.udf(negative_sampler, returnType=ArrayType(StringType()))


def generate_negative_samples(df_event: SparkDataFrame, df_content: SparkDataFrame, major_col: str = 'openid',
                              candidate_col: str = 'uuid', time_col: str = 'publish_time', sample_size: int = 20,
                              retries: int = 5) -> SparkDataFrame:
    """Do negative sampling based on given event pool and content pool

    Args:
        df_event (SparkDataFrame): event pool
        df_content (SparkDataFrame): content pool
        major_col (str, optional): primary column for negative sampling. Defaults to 'openid'.
        candidate_col (str, optional): secondary/candidate column for negative sampling. Defaults to 'uuid'.
        time_col (str, optional): time column. Defaults to 'publish_time'.
        sample_size (int, optional): negative samples size. Defaults to 20.
        retries (int, optional): number of retries for one negative sampling iteration. Defaults to 5.

    Returns:
        SparkDataFrame: pysaprk dataframe with the format of (positvie, negative) pairs
    """
    result_col = 'result'
    df = df_event.select(major_col, candidate_col, time_col)
    df = df.groupBy(major_col).agg(f.collect_list(f.col(candidate_col)).alias(candidate_col), f.collect_list(f.col(time_col)).alias(time_col))

    content_data = {row[candidate_col]: row[time_col] for row in df_content.collect()}  # content id to publish time mapping

    df = df.withColumn(result_col, f.explode(negative_sampler_udf(sample_size, content_data, retries)(f.col(candidate_col), f.col(time_col))))
    df = df.select(major_col, result_col).withColumnRenamed(result_col, candidate_col)

    return df
