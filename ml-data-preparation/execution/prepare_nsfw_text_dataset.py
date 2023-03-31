# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'datapreparation.zip')

from src.options.nsfw_options import NSFWTextOptions
from src.gcsreader.nsfw_text_collection import NSFWTextReader
from utils import initial_spark, dump_pickle, upload_to_gcs
from utils.logger import logger


if __name__ == '__main__':

    opt = NSFWTextOptions().parse()

    PROJECT_ID = opt.project_id
    CONTENT_TYPE = opt.content_type
    RUN_TIME = opt.run_time
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'

    sql_context, spark_session = initial_spark(cores='5', memory='27G', overhead='5G', shuffle_partitions='1000', num_executors='3')
    sql_context.sql('SET spark.sql.autoBroadcastJoinThreshold = -1')  # disable broadcast join
    sql_context.sql('SET spark.sql.broadcastTimeout = 600000ms')
    sql_context.sql('SET spark.sql.shuffle.partitions = 1000')

    gcsreader = NSFWTextReader(PROJECT_ID, sql_context, CONTENT_TYPE, logger=logger)
    dataset = gcsreader.get_content_data()

    logger.info('[Data Preparation][NSFW text] Convert pyspark dataframe to pandas dataframe')
    dataset = dataset.toPandas()

    logger.info(f'Len of positive sample: {len(dataset[dataset["is_adult"]=="true"])}')
    logger.info(f'Len of negative sample: {len(dataset[dataset["is_adult"]=="false"])}')
    logger.info(f'Len of dataset: {len(dataset)}')

    dump_pickle(f'dataset', dataset, base_path=BASE_PATH)

    # Upload checkpoint, dataset to GCS
    if opt.upload_gcs:
        logger.info('[Data Preparation][NSFW text] Upload dataset to GCS')
        upload_to_gcs(BASE_PATH, f'machine-learning-models-{PROJECT_ID}', f'dataset/nsfw_text/{RUN_TIME}')
