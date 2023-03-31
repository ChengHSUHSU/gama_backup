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
    # eval parameter
    popu_date = '20230310'
    popu_hour = '04'
    gt_date = '20230310'
    gt_hour = '05'
    hot_start_date = '20230307'
    popularity_bucket_name = 'pipeline-bf-data-prod-001'

    # option
    opt = TrainLinUCBOptions().parse()

    # logging
    logger = Logger(logger_name=opt.logger_name, dev_mode=True)

    # build mlflow helper
    opt.mlflow_host = 'http://10.32.192.39:5000' # it will be dropped
    opt.mlflow_experiment_run_name = popu_date + popu_hour
    opt.mlflow_experiment_id = '26'
    mlflow_helper = MlflowHelper(opt, logger_name='Planet News Hot Items', logger=logger)
    mlflow_helper.connection()

    # basic option
    opt.experiment_name = 'test' # it will be dropped
    base_path = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    bucket_storage = f'machine-learning-models-{opt.project_id}'
    output_bucket_storage = f'pipeline-{opt.project_id}'
    run_date, run_hour = opt.run_date[:-2], opt.run_date[-2:]
    output_blob_latest = opt.output_blob.replace('RUN_DATE/RUN_HOUR', 'latest')
    output_blob_run_time = opt.output_blob.replace('RUN_DATE', run_date).replace('RUN_HOUR', run_hour)
    logger.info(f'[Model Training][Planet News Hot Item] base_path: {base_path}, experiment_name: {opt.experiment_name}')

    # add prameter to mlflow
    logger.info('[Model Training][Planet News Hot Item][MLFlow] Logging pipeline option')
    mlflow_helper.log_data(data=vars(opt), log_type='params')

    # initialize factory and training parameters
    planet_news_hot_linucb_factory = PlanetNewsHotLinUCBFactory(mode='train', use_cuda=True, **vars(opt))

    # download and read dataset
    if opt.download_dataset:
        file_to_download = [opt.dataset, planet_news_hot_linucb_factory.config.COL2CATS_NAMES]
        # download train data 
        for file in file_to_download:
            logger.info(f'[Model Training][Planet News Hot Item] Download {file} from the bucket')
            download_blob(bucket_name=bucket_storage,
                          source_blob_name=f'{opt.dataset_blob_path}/{base_path}/{file}',
                          destination_file_name=f'{opt.dataroot}/{file}')

        # download popularity (it will be removed ??)
        logger.info(f'[Model Training][Planet News Hot Item] Download popularity_news.csv from the bucket')
        download_blob(bucket_name=popularity_bucket_name,
                      source_blob_name=f'metrics/popularity/news/{popu_date}/{popu_hour}/popularity_news.csv',
                      destination_file_name=f'{opt.dataroot}/popularity_news.csv')

        # download user_event (ground truth) (it will be removed ??)
        download_blob(bucket_name=bucket_storage,
                      source_blob_name=f'dataset/planet_news/hot/{gt_date}{gt_hour}/checkpoints/test/user_event.pickle',
                      destination_file_name=f'{opt.dataroot}/user_event_t.pickle')

    # load dataset, all_cats, user_event (grund truth) and popularity
    dataset = read_pickle(file_name='dataset.pickle', base_path=opt.dataroot)
    all_cats = read_pickle(file_name=planet_news_hot_linucb_factory.config.COL2CATS_NAMES, base_path=opt.dataroot)
    user_event = read_pickle(file_name='user_event_t.pickle', base_path=opt.dataroot)
    popularity_news = pd.read_csv(opt.dataroot+'/popularity_news.csv')

    # add dataset information to mlflow
    logger.info('[Model Training][Planet News Hot Item][MLFlow] Logging dataset information')
    mlflow_helper.log_data(data={'Dataset Size': dataset.shape[0]}, log_type='metrics')

    # TEST DATA
    dataset = dataset[dataset['date']>=hot_start_date]
    user_event = user_event[(user_event['date']==gt_date)&(user_event['hour']==str(int(gt_hour)))]

    # train
    planet_news_hot_linucb_factory.train(dataset, **vars(opt))

    # inference
    pop_score = planet_news_hot_linucb_factory.predict()

    # postprocess
    df_result = postprocess(factory=planet_news_hot_linucb_factory, pop_score=pop_score)

    df_result_ = df_result[['content_id', 'final_score']]
    print('df_result_ --------->')
    print(df_result_.head(15))

    # evaluation
    evaluation(hot_dat=df_result, popu_dat=popularity_news, gt_dat=user_event)

    # popularity_news
    popularity_news = popularity_news.sort_values(by=['final_score'], ascending=False)[['uuid', 'final_score']]
    print('popularity_news --------->')
    print(popularity_news.head(15))

    # dump
    if len(output_blob_run_time) > 0:
        logger.info(f'[Model Training][Planet News Hot Item] dump predict results to {output_blob_run_time}')
        save_data_to_gs(df_result, output_bucket_storage, output_blob_run_time)
    if len(output_blob_latest) > 0:
        logger.info(f'[Model Training][Planet News Hot Item] dump predict results  to {output_blob_latest}')
        save_data_to_gs(df_result, output_bucket_storage, output_blob_latest)
