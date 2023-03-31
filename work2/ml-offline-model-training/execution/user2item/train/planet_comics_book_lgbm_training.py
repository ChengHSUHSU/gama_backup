# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'offline-model.zip')

from bdds_recommendation.src.preprocess.prepare_planet_comics_book_user2item_lgbm_feature import PlanetComicsBookUser2ItemDataPreprocessor
from bdds_recommendation.src.configs.planet_user2item import PlanetComicsBookUser2ItemLGBMConfig
from bdds_recommendation.src.user2item.lgbm_handler import LGBMHandler
from bdds_recommendation.src.options.train_options import TrainLGBMOptions
from bdds_recommendation.src.model_registry import MlflowHelper
from bdds_recommendation.src.monitoring import ModelMonitor, MailOperator
from bdds_recommendation.utils import download_blob, dump_pickle, read_pickle
from bdds_recommendation.utils.logger import logger

MODEL_VERSION = '1.0.1'
MODEL_TYPE = 'lgbm'
configs = PlanetComicsBookUser2ItemLGBMConfig()


if __name__ == '__main__':
    # Parse arguments & store checkpoint
    opt = TrainLGBMOptions().parse()

    # build mlflow helper
    mlflow_helper = MlflowHelper(opt, logger_name='Planet Comics Book User2Item', logger=logger)
    mlflow_helper.connection()
    mlflow_helper.set_model_version(model_version=MODEL_VERSION)

    # Basic option
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    DATASET_FILE_NAME = opt.dataset
    BUCKET_STORAGE = f'machine-learning-models-{opt.project_id}'
    SOURCE_BLOB = f'{opt.dataset_blob_path}/{BASE_PATH}/{DATASET_FILE_NAME}'
    DESTINATION_PATH = f'{opt.dataroot}/{DATASET_FILE_NAME}'

    SERVICE_TYPE = getattr(configs, 'SERVICE_TYPE', 'user2item')
    CONTENT_TYPE = getattr(configs, 'CONTENT_TYPE', 'planet_comics_book')

    logger.info('[Planet Comics Book User2Item][MLFlow] Logging pipeline option')
    mlflow_helper.log_data(data=vars(opt), log_type='params')

    if opt.download_dataset:
        file_to_download = [opt.dataset, configs.COL2CATS_NAMES]

        for file in file_to_download:
            logger.info(f'[Planet Comics Book User2Item][Data Preprocessing] Download {file} from the bucket')
            download_blob(bucket_name=BUCKET_STORAGE,
                          source_blob_name=f'{opt.dataset_blob_path}/{BASE_PATH}/{file}',
                          destination_file_name=f'{opt.dataroot}/{file}')

    logger.info('[Planet Comics Book User2Item][Data Preprocessing] Start data preprocessing')
    preprocessor = PlanetComicsBookUser2ItemDataPreprocessor(opt, configs, encoders=None, logger=logger)

    dataset = preprocessor.process()

    dataset = dataset[configs.REQUISITE_COLS]
    logger.info('[Planet Comics Book User2Item][Data Preprocessing] Logging dataset information')
    mlflow_helper.log_data(data={'Positive Sample Size': len(dataset[dataset['y'] == 1]),
                                 'Negative Sample Size': len(dataset[dataset['y'] == 0])},
                           log_type='metrics')

    params = {'n_estimators': opt.n_estimators, 'learning_rate': opt.learning_rate, 'objective': opt.objective}

    lgbm_handler = LGBMHandler(opt, configs)
    lgbm_handler.set_params(**params)

    if not opt.is_train:
        exit(0)

    logger.info('[Planet Comics Book User2Item][Model Validation] Start model validation')
    X = dataset.drop(configs.ID_COLS+['y'], axis=1)
    Y = dataset['y']
    p, r, f, auc, ndcg = lgbm_handler.cross_validation(dataset, Y)

    validation_result = {
        'val_precision': p[1],
        'val_recall': r[1],
        'val_f1': f[1],
        'val_auc': auc[0]
    }

    for key, score in ndcg.items():
        validation_result[f'val_{key}'] = score

    logger.info(f'[Planet Comics Book User2Item][Model Validation] Show validation result \n{validation_result}')
    logger.info('[Planet Comics Book User2Item][Model Validation] Logging validation result')
    mlflow_helper.log_data(data=validation_result, log_type='metrics')

    logger.info('[Planet Comics Book User2Item][Model Training] Start model training')
    lgbm_handler.train(X, Y)

    if opt.save:

        MODEL_NAME = f'model.{mlflow_helper.model_version}.pickle'
        ENCODER_NAME = f'encoders.{mlflow_helper.model_version}.pickle'
        DATAENCODER_NAME = f'dataencoders.{mlflow_helper.model_version}.pickle'

        logger.info(f'[Planet Comics Book User2Item][MLFlow] Store encoders')
        ENCODER_TO_SAVE = {
            ENCODER_NAME: preprocessor.encoder.col2label2idx,
            DATAENCODER_NAME: preprocessor.encoder.col2DataEncoder,
            MODEL_NAME: lgbm_handler.lgbm
        }
        for file_name, data in ENCODER_TO_SAVE.items():
            dump_pickle(file_name, data, base_path=BASE_PATH)
            mlflow_helper.log_artifact(f'{BASE_PATH}/{file_name}')

        if opt.deploy:
            mlflow_helper.deploy(BASE_PATH, destination_blob_path=f'online_model/r12n/{SERVICE_TYPE}/{CONTENT_TYPE}/{MODEL_TYPE}')

    if opt.monitoring:
        logger.info('[Planet Comics Book User2Item][Model Offline Monitoring] Start offline monitoring')

        monitor = ModelMonitor(mlflow_host=opt.mlflow_host,
                               experiment_id=opt.mlflow_experiment_id,
                               service_endpoint=opt.api_model_version_url,
                               service_type=SERVICE_TYPE,
                               content_type=CONTENT_TYPE,
                               model_type=MODEL_TYPE,
                               model_path=configs.BASELINE_MODEL_PATH)
        monitor.process()
        logger.info(f'[Planet Comics Book User2Item][Model Offline Monitoring] Compare with version : {monitor.model_version}')
        base_model_path = f'model.{monitor.model_version}.pickle'
        base_data_encoders_path = f'dataencoders.{monitor.model_version}.pickle'
        base_data_encoders = read_pickle(base_data_encoders_path, base_path=configs.BASELINE_MODEL_PATH)

        logger.info('[Planet Comics Book User2Item][Data Preprocessing] Start data preprocessing')
        opt.is_train = False
        preprocessor = PlanetComicsBookUser2ItemDataPreprocessor(opt, configs, encoders=base_data_encoders, logger=logger)

        dataset = preprocessor.process()
        dataset = dataset[configs.REQUISITE_COLS]
        logger.info('[Planet Comics Book User2Item][Data Preprocessing] Logging dataset information')
        mlflow_helper.log_data(data={'Positive Sample Size': len(dataset[dataset['y'] == 1]),
                                     'Negative Sample Size': len(dataset[dataset['y'] == 0])},
                               log_type='metrics')

        params = {'n_estimators': opt.n_estimators, 'learning_rate': opt.learning_rate, 'objective': opt.objective}

        baseline_handler = LGBMHandler(opt, configs)
        baseline_handler.set_params(**params)
        baseline_handler.load_model(base_path=configs.BASELINE_MODEL_PATH, postfix_tag=monitor.model_version)
        _, _, _, baseline_auc, baseline_ndcg = baseline_handler.cross_validation(dataset, dataset['y'])

        comparison_result = {'val_auc': auc[0]}
        for key, score in baseline_ndcg.items():
            comparison_result[f'val_{key}'] = score

        logger.info('[Planet Comics Book User2Item][Model Offline Monitoring] Compute performance improvement rate')

        comparison_result = monitor.compare_model_performance(metrics_current=validation_result,
                                                              metrics_compared=comparison_result,
                                                              postfix=configs.PERFORMANCE_IMPROVEMENT_RATE_POSTFIX,
                                                              requisite_metrics=configs.MONITOR_METRICS_TO_THRESHOLD_MAPPING.keys())

        mail_operator = MailOperator(setting=opt)

        alert_list = []
        for monitor_metric, threshold in configs.MONITOR_METRICS_TO_THRESHOLD_MAPPING.items():
            value = comparison_result[f'{monitor_metric}{configs.PERFORMANCE_IMPROVEMENT_RATE_POSTFIX}']
            if (value > threshold) or (value < -threshold):
                alert_list.append((monitor_metric, value))

        if len(alert_list) > 0:
            logger.info('[Planet Comics Book User2Item][Model Offline Monitoring] Send alert mail')

            message = mail_operator.get_alert_message(content_type=CONTENT_TYPE,
                                                      service_type=SERVICE_TYPE,
                                                      model_type=MODEL_TYPE,
                                                      current_model_version=mlflow_helper.model_version,
                                                      online_model_version=monitor.model_version,
                                                      performance_improvement_rate_list=alert_list)

            mail_operator.send_mail(mail_subject='[ml-offline-model-training] offline monitoring alert',
                                    mail_from=opt.mail_server_account,
                                    mail_to=configs.ALERT_MAIL_RECIPIENTS,
                                    mail_message=message)
        else:
            logger.info('[Planet Comics Book User2Item][Model Offline Monitoring] No alert mail need to be sent')

        logger.info(f'[Planet Comics Book User2Item][Model Offline Monitoring] Show validation result \n{comparison_result}')
        logger.info('[Planet Comics Book User2Item][Model Offline Monitoring] Logging validation result')
        mlflow_helper.log_data(data=comparison_result, log_type='metrics')
