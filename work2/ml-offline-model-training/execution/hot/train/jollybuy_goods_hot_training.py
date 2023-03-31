# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'offline-model.zip')

from bdds_recommendation.src.configs.jollybuy_hot import JollybuyGoodsContextualBanditConfig
from bdds_recommendation.src.hot.linucb_handler import NumbaLinUCBHandler
from bdds_recommendation.src.preprocess.utils import calculate_min_max
from bdds_recommendation.src.preprocess.prepare_jollybuy_hot_feature import JollybuyGoodsHotDataPreprocesser
from bdds_recommendation.src.model_registry import MlflowHelper
from bdds_recommendation.utils import download_blob, calculate_dynamic_score, save_data_to_gs, \
    HOUR_STRING_FORMAT, DAY_STRING_FORMAT, HOUR_ONLY_STRING_FORMAT
from bdds_recommendation.utils.logger import Logger
from bdds_recommendation.src.options.train_options import TrainContextualBandit
from datetime import datetime, timedelta
import pandas as pd


def postprocess(model_handler, pop_score, configs, raw_data):

    raw_col = getattr(configs, 'RAW_SCORE_COL', 'raw_score')
    norm_raw_score_col = getattr(configs, 'NORM_RAW_SCORE_COL', 'norm_raw_score')
    dynamic_score_col = getattr(configs, 'DYNAMIC_SCORE_COL', 'dynamic')
    final_col = getattr(configs, 'FINAL_SCORE_COL', 'final_score')

    id_col = getattr(configs, 'ARM_COL', 'uuid')
    total_view_cnt_col = getattr(configs, 'TOTAL_VIEW_CNT_COL', 'total_view_count')
    total_click_cnt_col = getattr(configs, 'TOTAL_CLICK_CNT_COL', 'total_click_count')

    # dynamic
    dynamic_score = calculate_dynamic_score(len(model_handler.arms), weight=configs.DYNAMIC_WEIGHT)
    id2pop = list(zip(list(model_handler.model.index2arm.values()), pop_score))
    df_id2pop = pd.DataFrame(id2pop, columns=[configs.ARM_COL, raw_col])

    # normalize
    min_score = df_id2pop[raw_col].min()
    max_score = df_id2pop[raw_col].max()

    df_id2pop[norm_raw_score_col] = df_id2pop[raw_col].apply(lambda x: calculate_min_max(x, max_score, min_score))
    df_id2pop[dynamic_score_col] = dynamic_score
    df_id2pop[final_col] = df_id2pop[norm_raw_score_col] + df_id2pop[dynamic_score_col]

    # trim
    df_id2pop[final_col] = df_id2pop[final_col].apply(lambda x: 1 if x > 1 else x)
    df_id2pop.sort_values(by=[final_col], ascending=False, inplace=True)

    # get total_view_cnt and total_click_cnt
    df_view_cnts = raw_data.groupby(id_col)[configs.VIEW_CNT_COLS].sum()
    df_click_cnts = raw_data.groupby(id_col)[configs.CLICK_CNT_COLS].sum()

    df_view_cnts[total_view_cnt_col] = df_view_cnts.sum(axis=1)
    df_click_cnts[total_click_cnt_col] = df_click_cnts.sum(axis=1)

    df_cnts = pd.DataFrame(data=list(zip(df_view_cnts.index.values, df_view_cnts[total_view_cnt_col].values)),
                           columns=[id_col, total_view_cnt_col])
    df_cnts[total_click_cnt_col] = df_click_cnts[total_click_cnt_col].values

    df_id2pop = pd.merge(left=df_id2pop, right=df_cnts, on=id_col)

    return df_id2pop


if __name__ == '__main__':

    # Parse arguments & store checkpoint
    opt = TrainContextualBandit().parse()
    logger = Logger(logger_name=opt.logger_name, dev_mode=True)

    # build mlflow helper
    mlflow_helper = MlflowHelper(opt, logger_name='Jollybuy Goods Hot Items', logger=logger)
    mlflow_helper.connection()

    # basic options
    base_path = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    input_bucket_storage = f'machine-learning-models-{opt.project_id}'
    output_bucket_storage = f'pipeline-{opt.project_id}'
    dataset_name = opt.dataset

    logger.info('[Jollybuy Goods Hot Items][MLFlow] Logging pipeline option')
    mlflow_helper.log_data(data=vars(opt), log_type='params')

    # output paths
    if opt.run_date == '':
        run_time = (datetime.utcnow() - timedelta(hours=1)).strftime(HOUR_STRING_FORMAT)
    else:
        run_time = opt.run_date

    logger.info(f'[Jollybuy Goods Hot Items] run_date input={opt.run_date}, run_time={run_time}')
    run_date = datetime.strptime(run_time, HOUR_STRING_FORMAT).strftime(DAY_STRING_FORMAT)
    run_hour = datetime.strptime(run_time, HOUR_STRING_FORMAT).strftime(HOUR_ONLY_STRING_FORMAT)

    output_blob_run_time = opt.output_blob.replace('RUN_DATE', run_date).replace('RUN_HOUR', run_hour)
    output_blob_latest = opt.output_blob.replace('RUN_DATE/RUN_HOUR', 'latest')

    configs = JollybuyGoodsContextualBanditConfig()
    file_to_download = [opt.dataset, configs.COL2CATS_NAMES]

    # download input dataset
    if opt.download_dataset:
        for file in file_to_download:
            logger.info(f'[Jollybuy Goods Hot Items][Data Preprocessing] Download {file} from the bucket')
            download_blob(bucket_name=input_bucket_storage,
                          source_blob_name=f'{opt.dataset_blob_path}/{base_path}/{file}',
                          destination_file_name=f'{opt.dataroot}/{file}')

    # preprocess dataset
    preprocesser = JollybuyGoodsHotDataPreprocesser(opt=opt,
                                                    configs=configs,
                                                    all_cats_file=configs.COL2CATS_NAMES,
                                                    is_train=opt.is_train,
                                                    col2label2idx=None,
                                                    logger=logger)
    numba_lincub_handler = NumbaLinUCBHandler(opt=opt, configs=configs,
                                              preprocesser=preprocesser, is_train=opt.is_train,
                                              logger=logger)
    numba_lincub_handler.preprocess(requisite_cols=configs.REQUISITE_COLS, dummy_encode=True)

    logger.info('[Jollybuy Goods Hot Items][Data Preprocessing] Logging dataset information')
    mlflow_helper.log_data(data={'Dataset Size': len(numba_lincub_handler.dataset)},
                           log_type='metrics')

    # preparing model, training, making inference
    numba_lincub_handler.init_model()
    numba_lincub_handler.train()
    pop_score = numba_lincub_handler.predict()

    # do post processing
    df = postprocess(model_handler=numba_lincub_handler, pop_score=pop_score,
                     configs=configs, raw_data=preprocesser.raw_dataset)

    # dump
    logger.info(f'[Jollybuy Goods Hot Items][Data Preprocessing] dump popularity score to {output_blob_run_time}')
    save_data_to_gs(df, output_bucket_storage, output_blob_run_time)

    logger.info(f'[Jollybuy Goods Hot Items][Data Preprocessing] dump popularity score to {output_blob_latest}')
    save_data_to_gs(df, output_bucket_storage, output_blob_latest)
