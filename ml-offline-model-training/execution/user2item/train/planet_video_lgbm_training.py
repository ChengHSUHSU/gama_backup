# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'offline-model.zip')

from src.preprocess.prepare_planet_video_user2item_lgbm_feature import PlanetVideosUser2ItemDataPreprocessor
from src.configs.planet_user2item import PlanetVideoUser2ItemLGBMConfig
from src.user2item.lgbm_handler import LGBMHandler
from src.options.train_options import TrainLGBMOptions
from src.model_registry import MlflowHelper
from utils import download_blob, dump_pickle
from utils.logger import logger

MODEL_VERSION = '1.0.1'
configs = PlanetVideoUser2ItemLGBMConfig()


if __name__ == '__main__':
    # Parse arguments & store checkpoint
    opt = TrainLGBMOptions().parse()

    # build mlflow helper
    mlflow_helper = MlflowHelper(opt, logger_name='Planet Video User2Item', logger=logger)
    mlflow_helper.connection()
    mlflow_helper.set_model_version(model_version=MODEL_VERSION)

    # Basic option
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    DATASET_FILE_NAME = opt.dataset
    BUCKET_STORAGE = f'machine-learning-models-{opt.project_id}'
    SOURCE_BLOB = f'{opt.dataset_blob_path}/{BASE_PATH}/{DATASET_FILE_NAME}'
    DESTINATION_PATH = f'{opt.dataroot}/{DATASET_FILE_NAME}'

    logger.info('[Planet Video User2Item][MLFlow] Logging pipeline option')
    mlflow_helper.log_data(data=vars(opt), log_type='params')

    if opt.download_dataset:
        file_to_download = [opt.dataset, configs.COL2CATS_NAMES]

        for file in file_to_download:
            logger.info(f'[Planet Video User2Item][Data Preprocessing] Download {file} from the bucket')
            download_blob(bucket_name=BUCKET_STORAGE,
                          source_blob_name=f'{opt.dataset_blob_path}/{BASE_PATH}/{file}',
                          destination_file_name=f'{opt.dataroot}/{file}')

    logger.info('[Planet Video User2Item][Data Preprocessing] Start data preprocessing')
    preprocessor = PlanetVideosUser2ItemDataPreprocessor(opt, configs, encoders=None, logger=logger)

    dataset = preprocessor.process()
    dataset = dataset[configs.REQUISITE_COLS]

    logger.info('[Planet Video User2Item][Data Preprocessing] Logging dataset information')
    mlflow_helper.log_data(data={'Positive Sample Size': len(dataset[dataset['y'] == 1]),
                                 'Negative Sample Size': len(dataset[dataset['y'] == 0])},
                           log_type='metrics')

    params = {'n_estimators': opt.n_estimators, 'learning_rate': opt.learning_rate, 'objective': opt.objective}

    lgbm_handler = LGBMHandler(opt, configs)
    lgbm_handler.set_params(**params)

    if opt.is_train:
        logger.info('[Planet Video User2Item][Model Validation] Start model validation')
        X = dataset.drop(configs.ID_COLS+['y'], axis=1)

        # for ndcg metrics
        dataset = dataset.rename(columns={'openid': 'userid'})

        Y = dataset['y']
        p, r,  f, auc, ndcg = lgbm_handler.cross_validation(dataset, Y)

        validation_result = {
            'Validation Precision': p[1],
            'Validation Recall': r[1],
            'Validation F1-Score': f[1],
            'Validation AUC': auc[0]
        }

        for key, score in ndcg.items():
            validation_result[f'Validation {key}'] = score

        logger.info(f'[Planet Video User2Item][Model Validation] Show validation result \n{validation_result}')
        logger.info('[Planet Video User2Item][Model Validation] Logging validation result')
        mlflow_helper.log_data(data=validation_result, log_type='metrics')

        logger.info('[Planet Video User2Item][Model Training] Start model training')
        lgbm_handler.train(X, Y)

    if opt.save:

        MODEL_NAME = f'model.{mlflow_helper.model_version}.pickle'
        ENCODER_NAME = f'encoders.{mlflow_helper.model_version}.pickle'
        DATAENCODER_NAME = f'dataencoders.{mlflow_helper.model_version}.pickle'

        logger.info(f'[Planet Video Item2Item][MLFlow] Store encoders')
        ENCODER_TO_SAVE = {
            ENCODER_NAME: preprocessor.encoder.col2label2idx,
            DATAENCODER_NAME: preprocessor.encoder.col2DataEncoder,
            MODEL_NAME: lgbm_handler.lgbm}

        for file_name, data in ENCODER_TO_SAVE.items():
            dump_pickle(file_name, data, base_path=BASE_PATH)
            mlflow_helper.log_artifact(f'{BASE_PATH}/{file_name}')

        if opt.deploy:
            mlflow_helper.deploy(BASE_PATH, destination_blob_path='online_model/r12n/user2item/planet_video/lgbm')
