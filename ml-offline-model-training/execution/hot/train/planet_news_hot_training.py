# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'offline-model.zip')

import numpy as np
import pandas as pd

from utils.logger import Logger
from utils import download_blob, read_pickle
from src.preprocess.utils import calculate_min_max
from src.options.train_options import TrainLinUCBOptions
from src_v2.factory.planet_news.planet_news_hot import PlanetNewsHotLinUCBFactory
from utils import download_blob, calculate_dynamic_score, save_data_to_gs
from src.model_registry import MlflowHelper


RAW_SCORE_COL = 'raw_score'
NORM_RAW_SCORE_COL = 'norm_raw_score'
DYNAMIC_SCORE_COL = 'dynamic'
FINAL_SCORE_COL = 'final_score'
ARM_COL = 'content_id'
DYNAMIC_WEIGHT = 0.01


def postprocess(factory: PlanetNewsHotLinUCBFactory, pop_score: np.array=None) -> pd.DataFrame:
    # dynamic
    dynamic_score = calculate_dynamic_score(len(factory.model_handler.arms), weight=DYNAMIC_WEIGHT)
    id2pop = list(zip(list(factory.model_handler.model.index2arm.values()), pop_score))
    df_id2pop = pd.DataFrame(id2pop, columns=[ARM_COL, RAW_SCORE_COL])
    # normalize
    min_score = df_id2pop[RAW_SCORE_COL].min()
    max_score = df_id2pop[RAW_SCORE_COL].max()
    df_id2pop[NORM_RAW_SCORE_COL] = df_id2pop[RAW_SCORE_COL].apply(lambda x: calculate_min_max(x, max_score, min_score))
    df_id2pop[DYNAMIC_SCORE_COL] = dynamic_score
    df_id2pop[FINAL_SCORE_COL] = df_id2pop[NORM_RAW_SCORE_COL] + df_id2pop[DYNAMIC_SCORE_COL]
    # trim
    df_id2pop[FINAL_SCORE_COL] = df_id2pop[FINAL_SCORE_COL].apply(lambda x: 1 if x > 1 else x)
    df_id2pop.sort_values(by=[FINAL_SCORE_COL], ascending=False, inplace=True)
    return df_id2pop


if __name__ == '__main__':
    # option
    opt = TrainLinUCBOptions().parse()
    # logging
    logger = Logger(logger_name=opt.logger_name, dev_mode=True)

    # build mlflow helper
    mlflow_helper = MlflowHelper(opt, logger_name='Planet News Hot Items', logger=logger)
    mlflow_helper.connection()

    # basic option
    base_path = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    bucket_storage = f'machine-learning-models-{opt.project_id}'
    output_bucket_storage = f'pipeline-{opt.project_id}'
    run_date, run_hour = opt.run_date[:-2], opt.run_date[-2:]
    output_blob_latest = opt.output_blob.replace('RUN_DATE/RUN_HOUR', 'latest')
    output_blob_run_time = opt.output_blob.replace('RUN_DATE', run_date).replace('RUN_HOUR', run_hour)
    logger.info(f'[Model Training][Planet News Hot Item] base_path: {base_path}, experiment_name: {opt.experiment_name}')

    # add parameters to mlflow
    logger.info('[Model Training][Planet News Hot Item][MLFlow] Logging pipeline option')
    mlflow_helper.log_data(data=vars(opt), log_type='params')

    # initialize factory and training parameters
    planet_news_hot_linucb_factory = PlanetNewsHotLinUCBFactory(mode='train', **vars(opt))

    # download and read dataset
    if opt.download_dataset:
        file_to_download = [opt.dataset]
        # download train data 
        for file in file_to_download:
            logger.info(f'[Model Training][Planet News Hot Item] Download {file} from the bucket')
            download_blob(bucket_name=bucket_storage,
                          source_blob_name=f'{opt.dataset_blob_path}/{base_path}/{file}',
                          destination_file_name=f'{opt.dataroot}/{file}')

    # load dataset, all_cats, user_event (grund truth)
    dataset = read_pickle(file_name='dataset.pickle', base_path=opt.dataroot)
    popularity_news = pd.read_csv(opt.dataroot + '/popularity_news.csv')

    # add dataset information to mlflow
    logger.info('[Model Training][Planet News Hot Item][MLFlow] Logging dataset information')
    mlflow_helper.log_data(data={'Dataset Size': dataset.shape[0]}, log_type='metrics')

    # train
    planet_news_hot_linucb_factory.train(dataset, **vars(opt))

    # inference
    pop_score = planet_news_hot_linucb_factory.predict()

    # postprocess
    df_result = postprocess(factory=planet_news_hot_linucb_factory, pop_score=pop_score)

    # dump
    if len(output_blob_run_time) > 0:
        logger.info(f'[Model Training][Planet News Hot Item] dump predict results to {output_blob_run_time}')
        save_data_to_gs(df_result, output_bucket_storage, output_blob_run_time)
    if len(output_blob_latest) > 0:
        logger.info(f'[Model Training][Planet News Hot Item] dump predict results  to {output_blob_latest}')
        save_data_to_gs(df_result, output_bucket_storage, output_blob_latest)
