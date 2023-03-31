# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'offline-model.zip')

from bdds_recommendation.src.preprocess.prepare_club_post_user2item_feature import ClubPostUser2ItemDataPreprocesser
from bdds_recommendation.src.configs.club_user2item import ClubPostUser2ItemConfig
from bdds_recommendation.src.user2item.dcn_handler import DCNHandler
from bdds_recommendation.src.options.train_options import TrainDCNOptions
from bdds_recommendation.src.model_registry import MlflowHelper
from bdds_recommendation.src.monitoring import ModelMonitor, MailOperator
from bdds_recommendation.utils import download_blob, dump_pickle, read_pickle
from bdds_recommendation.utils.logger import logger

MODEL_VERSION = '1.0.1'
MODEL_TYPE = 'dcn'

configs = ClubPostUser2ItemConfig()


if __name__ == '__main__':
    # parse arguments & store checkpoint
    opt = TrainDCNOptions().parse()

    # build mlflow helper
    mlflow_helper = MlflowHelper(opt, logger_name='Club Post User2Item', logger=logger)
    mlflow_helper.connection()
    mlflow_helper.set_model_version(model_version=MODEL_VERSION)

    # basic option
    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    DATASET_FILE_NAME = opt.dataset
    BUCKET_STORAGE = f'machine-learning-models-{opt.project_id}'
    SOURCE_BLOB = f'{opt.dataset_blob_path}/{BASE_PATH}/{DATASET_FILE_NAME}'
    DESTINATION_PATH = f'{opt.dataroot}/{DATASET_FILE_NAME}'
    VALIDATE_REQUISITE_COLS = configs.REQUISITE_COLS + [configs.NDCG_GROUPBY_KEY] if configs.NDCG_GROUPBY_KEY else configs.REQUISITE_COLS
    SERVICE_TYPE = getattr(configs, 'SERVICE_TYPE', 'user2item')
    CONTENT_TYPE = getattr(configs, 'CONTENT_TYPE', 'club_post')

    logger.info('[Club Post User2Item][MLFlow] Logging pipeline option')
    mlflow_helper.log_data(data=vars(opt), log_type='params')

    if opt.download_dataset:
        file_to_download = [opt.dataset, configs.COL2CATS_NAMES]

        for file in file_to_download:
            logger.info(f'[Club Post User2Item][Data Preprocessing] Download {file} from the bucket')
            download_blob(bucket_name=BUCKET_STORAGE,
                          source_blob_name=f'{opt.dataset_blob_path}/{BASE_PATH}/{file}',
                          destination_file_name=f'{opt.dataroot}/{file}')

    logger.info('[Club Post User2Item][Data Preprocessing] Start data preprocessing')
    preprocesser = ClubPostUser2ItemDataPreprocesser(opt, configs, encoders=None, logger=logger)

    dataset = preprocesser.process()
    dataset = dataset[VALIDATE_REQUISITE_COLS]

    logger.info('[Club Post User2Item][Data Preprocessing] Logging dataset information')
    mlflow_helper.log_data(data={'Positive Sample Size': len(dataset[dataset['y'] == 1]),
                                 'Negative Sample Size': len(dataset[dataset['y'] == 0])},
                           log_type='metrics')

    params = {'batch_size': opt.batch_size, 'epochs': opt.epochs, 'verbose': opt.verbose}

    dcn_handler = DCNHandler(opt=opt, config=configs, encoders=preprocesser.encoder.col2DataEncoder,
                             col2label2idx=preprocesser.encoder.col2label2idx, use_cuda=True)
    dcn_handler.set_params(**params)

    if opt.is_train:
        logger.info('[Club Post User2Item][Model Validation] Start model validation')
        p, r, f, auc, roc_curve_info, ndcg = dcn_handler.cross_validation(dataset, dataset['y'], random_state=opt.seed)

        validation_result = {
            'val_precision': p[1],
            'val_recall': r[1],
            'val_f1': f[1],
            'val_auc': auc[0]
        }

        for key, score in ndcg.items():
            validation_result[f'val_{key}'] = score

        logger.info(f'[Club Post User2Item][Model Validation] Show validation result \n{validation_result}')
        logger.info('[Club Post User2Item][Model Validation] Logging validation result')
        mlflow_helper.log_data(data=validation_result, log_type='metrics')

        logger.info('[Club Post User2Item][Model Training] Start model training')
        dcn_handler.train(dataset, dataset['y'])

        if opt.save:

            model_name = f'model.{mlflow_helper.model_version}.pickle'
            encoder_name = f'encoders.{mlflow_helper.model_version}.pickle'
            data_encoder_name = f'data_encoders.{mlflow_helper.model_version}.pickle'

            logger.info('[Club Post User2Item][MLFlow] Store encoders')
            col2label2idx = preprocesser.encoder.col2label2idx
            encoder_to_save = {encoder_name: col2label2idx,
                               data_encoder_name: preprocesser.encoder.col2DataEncoder}

            for file_name, data in encoder_to_save.items():
                dump_pickle(file_name, data, base_path=BASE_PATH)
                mlflow_helper.log_artifact(f'{BASE_PATH}/{file_name}')

            logger.info('[Club Post User2Item][MLFlow] Store model')
            dcn_handler.save_model(model_name, base_path=BASE_PATH)
            mlflow_helper.log_artifact(f'{BASE_PATH}/{model_name}')

            if opt.deploy:
                mlflow_helper.deploy(BASE_PATH, destination_blob_path=f'online_model/r12n/{SERVICE_TYPE}/{CONTENT_TYPE}/{MODEL_TYPE}')

        if opt.monitoring:

            logger.info('[Club Post User2Item][Model Offline Monitoring] Start offline monitoring')

            monitor = ModelMonitor(mlflow_host=opt.mlflow_host,
                                   experiment_id=opt.mlflow_experiment_id,
                                   service_endpoint=opt.api_model_version_url,
                                   service_type=SERVICE_TYPE,
                                   content_type=CONTENT_TYPE,
                                   model_type=MODEL_TYPE,
                                   model_path=configs.BASELINE_MODEL_PATH)
            monitor.process()

            logger.info(f'[Club Post User2Item][Model Offline Monitoring] Compare with version : {monitor.model_version}')

            base_model_path = f'model.{monitor.model_version}.pickle'
            base_data_encoders_path = f'data_encoders.{monitor.model_version}.pickle'
            base_data_encoders = read_pickle(base_data_encoders_path, base_path=configs.BASELINE_MODEL_PATH)

            opt.is_train = False
            baseline_preprocesser = ClubPostUser2ItemDataPreprocesser(opt, configs, encoders=base_data_encoders, logger=logger)
            dataset = baseline_preprocesser.process()
            baseline_handler = DCNHandler(opt, configs, baseline_preprocesser.encoders)
            baseline_handler.set_params(**params)
            baseline_handler.load_model(model_path=base_model_path, base_path=configs.BASELINE_MODEL_PATH)
            _, _, _, baseline_auc, _, baseline_ndcg = baseline_handler.cross_validation(dataset, dataset['y'], random_state=opt.seed)

            comparison_result = {'val_auc': auc[0]}
            for key, score in baseline_ndcg.items():
                comparison_result[f'val_{key}'] = score

            logger.info('[Club Post User2Item][Model Offline Monitoring] Compute performance improvement rate')

            comparison_result = monitor.compare_model_performance(metrics_current=validation_result,
                                                                  metrics_compared=comparison_result,
                                                                  postfix=configs.PERFORMANCE_IMPROVEMENT_RATE_POSTFIX,
                                                                  requisite_metrics=configs.MONITOR_METRICS_TO_THRESHOLD_MAPPING.keys())

            logger.info('[Club Post User2Item][Model Offline Monitoring] Send alert mail')
            mail_operator = MailOperator(setting=opt)

            alert_list = []
            for monitor_metric, threshold in configs.MONITOR_METRICS_TO_THRESHOLD_MAPPING.items():
                value = comparison_result[f'{monitor_metric}{configs.PERFORMANCE_IMPROVEMENT_RATE_POSTFIX}']
                if (value > threshold) or (value < -threshold):
                    alert_list.append((monitor_metric, value))

            if len(alert_list) > 0:

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

            logger.info(f'[Club Post User2Item][Model Offline Monitoring] Show validation result \n{comparison_result}')
            logger.info('[Club Post User2Item][Model Offline Monitoring] Logging validation result')
            mlflow_helper.log_data(data=comparison_result, log_type='metrics')
