# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'offline-model.zip')

from bdds_recommendation.src.preprocess.prepare_nownews_news_user2item_din_feature import NownewsNewsUser2ItemDataPreprocesser
from bdds_recommendation.src.configs.nownews_user2item import NownewsNewsUser2ItemDINConfig
from bdds_recommendation.src.user2item.din_handler import DINHandler
from bdds_recommendation.src.options.train_options import TrainDINOptions
from bdds_recommendation.src.model_registry import MlflowHelper
from bdds_recommendation.utils import download_blob, dump_pickle
from bdds_recommendation.utils.logger import logger

MODEL_VERSION = '1.0.1'
configs = NownewsNewsUser2ItemDINConfig()


if __name__ == '__main__':
    # parse arguments & store checkpoint
    opt = TrainDINOptions().parse()

    # build mlflow helper
    mlflow_helper = MlflowHelper(opt, logger_name='Nownews News User2Item', logger=logger)
    mlflow_helper.connection()
    mlflow_helper.set_model_version(model_version=MODEL_VERSION)

    # basic option
    base_path = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    dataset_file_name = opt.dataset
    bucket_storage = f'machine-learning-models-{opt.project_id}'
    source_blob = f'{opt.dataset_blob_path}/{base_path}/{dataset_file_name}'
    destination_path = f'{opt.dataroot}/{dataset_file_name}'
    validate_requisite_cols = configs.REQUISITE_COLS + [configs.NDCG_GROUPBY_KEY] if configs.NDCG_GROUPBY_KEY else configs.REQUISITE_COLS

    logger.info('[Nownews News User2Item][MLFlow] Logging pipeline option')
    mlflow_helper.log_data(data=vars(opt), log_type='params')

    if opt.download_dataset:
        file_to_download = [opt.dataset, configs.COL2CATS_NAMES]

        for file in file_to_download:
            logger.info(f'[Nownews News User2Item][Data Preprocessing] Download {file} from the bucket')
            download_blob(bucket_name=bucket_storage,
                          source_blob_name=f'{opt.dataset_blob_path}/{base_path}/{file}',
                          destination_file_name=f'{opt.dataroot}/{file}')

    logger.info('[Nownews News User2Item][Data Preprocessing] Start data preprocessing')
    preprocesser = NownewsNewsUser2ItemDataPreprocesser(opt, configs, encoders=None, logger=logger)

    dataset = preprocesser.process()
    dataset = dataset[validate_requisite_cols]

    logger.info('[Nownews News User2Item][Data Preprocessing] Logging dataset information')
    mlflow_helper.log_data(data={'Positive Sample Size': len(dataset[dataset['y'] == 1]),
                                 'Negative Sample Size': len(dataset[dataset['y'] == 0])},
                           log_type='metrics')

    params = {'batch_size': opt.batch_size, 'epochs': opt.epochs, 'verbose': opt.verbose}

    din_handler = DINHandler(opt=opt, config=configs, encoders=preprocesser.encoder.col2DataEncoder,
                             col2label2idx=preprocesser.encoder.col2label2idx, use_cuda=True)
    din_handler.set_params(**params)

    if opt.is_train:
        logger.info('[Nownews News User2Item][Model Validation] Start model validation')
        p, r, f, auc, roc_curve_info, ndcg = din_handler.cross_validation(dataset, dataset['y'])

        validation_result = {
            'val_precision': p[1],
            'val_recall': r[1],
            'val_f1': f[1],
            'val_auc': auc[0]
        }

        for key, score in ndcg.items():
            validation_result[f'val_{key}'] = score

        logger.info(f'[Nownews News User2Item][Model Validation] Show validation result \n{validation_result}')
        logger.info('[Nownews News User2Item][Model Validation] Logging validation result')
        mlflow_helper.log_data(data=validation_result, log_type='metrics')

        logger.info('[Nownews News User2Item][Model Training] Start model training')
        din_handler.train(dataset, dataset['y'])

    if opt.save:

        model_name = f'model.{mlflow_helper.model_version}.pickle'
        encoder_name = f'encoders.{mlflow_helper.model_version}.pickle'
        data_encoder_name = f'data_encoders.{mlflow_helper.model_version}.pickle'

        logger.info('[Nownews News User2Item][MLFlow] Store encoders')
        col2label2idx = preprocesser.encoder.col2label2idx
        encoder_to_save = {encoder_name: col2label2idx,
                           data_encoder_name: preprocesser.encoder.col2DataEncoder}

        for file_name, data in encoder_to_save.items():
            dump_pickle(file_name, data, base_path=base_path)
            mlflow_helper.log_artifact(f'{base_path}/{file_name}')

        logger.info('[Nownews News User2Item][MLFlow] Store model')
        din_handler.save_model(model_name, base_path=base_path)
        mlflow_helper.log_artifact(f'{base_path}/{model_name}')

        if opt.deploy:
            mlflow_helper.deploy(base_path, destination_blob_path='online_model/r12n/user2item/nownews_news/din')
