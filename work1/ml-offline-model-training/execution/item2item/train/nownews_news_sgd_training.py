# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'offline-model.zip')

from bdds_recommendation.src.preprocess.prepare_nownews_news_item2item_feature import NownewsNewsItem2ItemDataPreprocesser
from bdds_recommendation.src.configs.nownews_item2item import NownewsNewsItem2ItemConfig
from bdds_recommendation.src.item2item.sgd_classifier_handler import SGDClassifierHandler
from bdds_recommendation.src.options.train_options import TrainSGDOptions
from bdds_recommendation.src.model_registry import MlflowHelper
from bdds_recommendation.utils import download_blob, dump_pickle
from bdds_recommendation.utils.logger import logger

MODEL_VERSION = '1.0.1'
configs = NownewsNewsItem2ItemConfig()


if __name__ == '__main__':
    # Parse arguments & store checkpoint
    opt = TrainSGDOptions().parse()

    # build mlflow helper
    mlflow_helper = MlflowHelper(opt, logger_name='Nownews News Item2Item', logger=logger)
    mlflow_helper.connection()
    mlflow_helper.set_model_version(model_version=MODEL_VERSION)

    # Basic option
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    DATASET_FILE_NAME = opt.dataset
    BUCKET_STORAGE = f'machine-learning-models-{opt.project_id}'

    logger.info('[Nownews News Item2Item][MLFlow] Logging pipeline option')
    mlflow_helper.log_data(data=vars(opt), log_type='params')

    if opt.download_dataset:
        file_to_download = [opt.dataset, configs.COL2CATS_NAMES]

        for file in file_to_download:
            logger.info(f'[Nownews News Item2Item][Data Preprocessing] Download {file} from the bucket')
            download_blob(bucket_name=BUCKET_STORAGE,
                          source_blob_name=f'{opt.dataset_blob_path}/{BASE_PATH}/{file}',
                          destination_file_name=f'{opt.dataroot}/{file}')

    logger.info('[Nownews News Item2Item][Data Preprocessing] Start data preprocessing')
    preprocesser = NownewsNewsItem2ItemDataPreprocesser(opt, configs, encoders=None, logger=logger)

    dataset = preprocesser.process()
    dataset = dataset[configs.REQUISITE_COLS]

    logger.info('[Nownews News Item2Item][Data Preprocessing] Logging dataset information')
    mlflow_helper.log_data(data={'Positive Sample Size': len(dataset[dataset['y'] == 1]),
                                 'Negative Sample Size': len(dataset[dataset['y'] == 0])},
                           log_type='metrics')

    # picked requisit features
    X = dataset.drop(['y'], axis=1)
    Y = dataset['y']

    logistic_model = SGDClassifierHandler(opt, configs)
    params = {
        'loss': opt.loss,
        'penalty': opt.penalty,
        'alpha': opt.alpha,
        'fit_intercept': opt.fit_intercept,
        'max_iter': opt.max_iter,
        'tol': opt.tol,
        'n_iter_no_change': opt.n_iter_no_change,
        'early_stopping': opt.early_stopping,
        'shuffle': opt.shuffle,
        'learning_rate': opt.learning_rate_mode,
        'eta0': opt.learning_rate,
    }
    logistic_model.set_params(**params)

    if opt.is_train:
        logger.info('[Nownews News Item2Item][Model Validation] Start model validation')
        eval_metrics = logistic_model.cross_validation(X, Y, k=5)
        logger.info(eval_metrics)

        validation_result = {
            'Validation Precision': eval_metrics['precision'][1],
            'Validation Recall': eval_metrics['recall'][1],
            'Validation F1-Score': eval_metrics['f1'][1],
            'Validation AUC': eval_metrics['auc'][0]
        }

        logger.info(f'[Nownews News Item2Item][Model Validation] Show validation result \n{eval_metrics}')
        logger.info('[Nownews News Item2Item][Model Validation] Logging validation result')
        mlflow_helper.log_data(data=validation_result, log_type='metrics')

        logger.info('[Nownews News Item2Item][Model Training] Start model training')
        logistic_model.train(X, Y)

    if opt.save:

        model_name = f'model.{mlflow_helper.model_version}.pickle'
        encoder_name = f'encoders.{mlflow_helper.model_version}.pickle'

        file_to_save = {model_name: logistic_model.clf, encoder_name: {}}

        for file_name, data in file_to_save.items():
            dump_pickle(file_name, data, base_path=BASE_PATH)
            mlflow_helper.log_artifact(f'{BASE_PATH}/{file_name}')

        if opt.deploy:
            mlflow_helper.deploy(BASE_PATH, destination_blob_path='online_model/r12n/item2item/nownews_news/sgd')
