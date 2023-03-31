# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'offline-model.zip')

from utils.logger import logger
from utils import download_blob, dump_pickle
from src.user2item.din_handler import DINHandler
from src.preprocess.prepare_planet_novel_book_user2item_din_feature import PlanetNovelBookUser2ItemDataPreprocessor
from src.options.train_options import TrainDINOptions
from src.model_registry import MlflowHelper
from src.configs.planet_user2item import PlanetNovelBookUser2ItemDINConfig


MODEL_VERSION = '1.0.0'
configs = PlanetNovelBookUser2ItemDINConfig()

if __name__ == '__main__':

    # parse arguments & store checkpoint
    opt = TrainDINOptions().parse()

    # build mlflow helper
    mlflow_helper = MlflowHelper(opt, logger_name='Planet Novel Book User2Item', logger=logger)
    mlflow_helper.connection()
    mlflow_helper.set_model_version(model_version=MODEL_VERSION)

    # basic option
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    DATASET_FILE_NAME = opt.dataset
    BUCKET_STORAGE = f'machine-learning-models-{opt.project_id}'
    DESTINATION_PATH = f'{opt.dataroot}/{DATASET_FILE_NAME}'
    VALIDATE_REQUISITE_COLS = configs.REQUISITE_COLS + [configs.NDCG_GROUPBY_KEY] if configs.NDCG_GROUPBY_KEY else configs.REQUISITE_COLS

    logger.info('[Planet Novel Book User2Item][MLFlow] Logging pipeline option')
    mlflow_helper.log_data(data=vars(opt), log_type='params')

    if opt.download_dataset:
        file_to_download = [opt.dataset, configs.COL2CATS_NAMES]

        for file in file_to_download:
            logger.info(f'[Planet Novel Book User2Item][Data Preprocessing] Download {file} from the bucket')
            download_blob(bucket_name=BUCKET_STORAGE,
                          source_blob_name=f'{opt.dataset_blob_path}/{BASE_PATH}/{file}',
                          destination_file_name=f'{opt.dataroot}/{file}')

    logger.info('[Planet Novel Book User2Item][Data Preprocessing] Start data preprocessing')
    preprocessor = PlanetNovelBookUser2ItemDataPreprocessor(opt, configs, encoders=None, logger=logger)

    dataset = preprocessor.process()
    dataset = dataset[VALIDATE_REQUISITE_COLS]

    logger.info('[Planet Novel Book User2Item][Data Preprocessing] Logging dataset information')
    mlflow_helper.log_data(data={'Positive Sample Size': len(dataset[dataset['y'] == 1]),
                                 'Negative Sample Size': len(dataset[dataset['y'] == 0])},
                           log_type='metrics')

    params = {'batch_size': opt.batch_size, 'epochs': opt.epochs, 'verbose': opt.verbose}

    din_handler = DINHandler(opt=opt, config=configs, encoders=preprocessor.encoder.col2DataEncoder,
                             col2label2idx=preprocessor.encoder.col2label2idx, use_cuda=True)
    din_handler.set_params(**params)

    if opt.is_train:
        logger.info('[Planet Novel Book User2Item][Model Validation] Start model validation')
        p, r, f, auc, roc_curve_info, ndcg = din_handler.cross_validation(dataset, dataset['y'])

        validation_result = {
            'val_precision': p[1],
            'val_recall': r[1],
            'val_f1': f[1],
            'val_auc': auc[0]
        }

        for key, score in ndcg.items():
            validation_result[f'val_{key}'] = score

        logger.info(f'[Planet Novel Book User2Item][Model Validation] Show validation result \n{validation_result}')
        logger.info('[Planet Novel Book User2Item][Model Validation] Logging validation result')
        mlflow_helper.log_data(data=validation_result, log_type='metrics')

        logger.info('[Planet Novel User2Item][Model Training] Start model training')
        din_handler.train(dataset, dataset['y'])

    if opt.save:

        MODEL_NAME = f'model.{mlflow_helper.model_version}.pickle'
        ENCODER_NAME = f'encoders.{mlflow_helper.model_version}.pickle'
        DATA_ENCODER_NAME = f'dataencoders.{mlflow_helper.model_version}.pickle'

        logger.info('[Planet Novel User2Item][MLFlow] Store encoders')
        col2label2idx = preprocessor.encoder.col2label2idx
        ENCODER_TO_SAVE = {ENCODER_NAME: col2label2idx,
                           DATA_ENCODER_NAME: preprocessor.encoder.col2DataEncoder}

        for file_name, data in ENCODER_TO_SAVE.items():
            dump_pickle(file_name, data, base_path=BASE_PATH)
            mlflow_helper.log_artifact(f'{BASE_PATH}/{file_name}')

        logger.info('[Planet Novel User2Item][MLFlow] Store model')
        din_handler.save_model(MODEL_NAME, base_path=BASE_PATH)
        mlflow_helper.log_artifact(f'{BASE_PATH}/{MODEL_NAME}')

        if opt.deploy:
            mlflow_helper.deploy(BASE_PATH, destination_blob_path='online_model/r12n/user2item/planet_novel_book/din')
