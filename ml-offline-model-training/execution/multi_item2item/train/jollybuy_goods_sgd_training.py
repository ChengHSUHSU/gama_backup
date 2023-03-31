# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'offline-model.zip')

from src.configs.jollybuy_item2item import JollybuyGoodsMultiItem2ItemConfig
from src.item2item.sgd_classifier_handler import SGDClassifierHandler
from src.model_registry import MlflowHelper
from src.monitoring import ModelMonitor, MailOperator
from src.options.train_options import TrainSGDOptions
from src.preprocess.prepare_jollybuy_item2item_xgb_feature import JollybuyGoodsItem2ItemDataPreprocesser
from utils import download_blob, dump_pickle, read_pickle
from utils.logger import Logger

MODEL_VERSION = '1.0.0'
MODEL_TYPE = 'sgd'

configs = JollybuyGoodsMultiItem2ItemConfig()

if __name__ == '__main__':

    # Parse arguments & store checkpoint
    opt = TrainSGDOptions().parse()
    logger = Logger(logger_name=opt.logger_name, dev_mode=True)

    # build mlflow helper
    mlflow_helper = MlflowHelper(opt, logger_name='Jollybuy Goods Multi Item2Item', logger=logger)
    mlflow_helper.connection()
    mlflow_helper.set_model_version(model_version=MODEL_VERSION)

    BASE_PATH = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    BUCKET_STORAGE = f'machine-learning-models-{opt.project_id}'
    SERVICE_TYPE = getattr(configs, 'SERVICE_TYPE', 'item2item')
    CONTENT_TYPE = getattr(configs, 'CONTENT_TYPE', 'jollybuy_goods')

    logger.info('[Jollybuy Goods Multi Item2Item][MLFlow] Logging pipeline option')
    mlflow_helper.log_data(data=vars(opt), log_type='params')

    file_to_download = [opt.dataset, configs.COL2CATS_NAMES]

    if opt.download_dataset:

        for file in file_to_download:
            logger.info(f'Downloads {file} from the GCS.')
            download_blob(bucket_name=BUCKET_STORAGE,
                          source_blob_name=f'{opt.dataset_blob_path}/{BASE_PATH}/{file}',
                          destination_file_name=f'{opt.dataroot}/{file}')

    # feature preprocess
    logger.info('Start Feature Preprocessing\n')
    preprocesser = JollybuyGoodsItem2ItemDataPreprocesser(opt=opt, configs=configs,
                                                          all_cats_file=configs.COL2CATS_NAMES, logger=logger)
    dataset = preprocesser.process()

    # picked requisite features
    dataset_all = dataset[configs.REQUISITE_COLS]

    logger.info('[Jollybuy Goods Multi Item2Item][Data Preprocessing] Logging dataset information')
    mlflow_helper.log_data(data={'Positive Sample Size': len(dataset[dataset['y'] == 1]),
                                 'Negative Sample Size': len(dataset[dataset['y'] == 0])},
                           log_type='metrics')

    X = dataset_all.drop(['y'], axis=1)
    Y = dataset_all['y']

    # training
    logistic_model = SGDClassifierHandler(opt, JollybuyGoodsMultiItem2ItemConfig)
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
        logger.info('Start cross validation')
        eval_metrics = logistic_model.cross_validation(X, Y, k=5, random_state=opt.seed)
        logger.info(eval_metrics)
        ndcg_scores = logistic_model.get_ndcg_score(X, Y, dataset, k=[5, 10, 20])
        logger.info(ndcg_scores)

        validation_result = {
            'val_precision': eval_metrics['precision'][1],
            'val_recall': eval_metrics['recall'][1],
            'val_f1': eval_metrics['f1'][1],
            'val_auc': eval_metrics['auc'][0],
            'val_ndcg5': ndcg_scores['ndcg5'],
            'val_ndcg10': ndcg_scores['ndcg10'],
            'val_ndcg20': ndcg_scores['ndcg20']
        }

        logger.info(f'[Jollybuy Goods Multi Item2Item][Model Validation] Show validation result \n{eval_metrics}')
        logger.info('[Jollybuy Goods Multi Item2Item][Model Validation] Logging validation result')
        mlflow_helper.log_data(data=validation_result, log_type='metrics')

        logger.info('Start training on all data')
        logistic_model.train(X, Y)

        if opt.save:
            model_name = f'model.{mlflow_helper.model_version}.pickle'
            encoder_name = f'encoders.{mlflow_helper.model_version}.pickle'

            dump_pickle(model_name, logistic_model.clf, base_path=BASE_PATH)
            dump_pickle(encoder_name, preprocesser.col2label2idx, base_path=BASE_PATH)
            mlflow_helper.log_artifact(f'{BASE_PATH}/{model_name}')
            mlflow_helper.log_artifact(f'{BASE_PATH}/{encoder_name}')

            if opt.deploy:
                mlflow_helper.deploy(BASE_PATH, destination_blob_path='online_model/r12n/multi_item2item/jollybuy_goods/sgd')

        if opt.monitoring:
            logger.info('[Jollybuy Goods Multi Item2Item][Model Offline Monitoring] Start offline monitoring')

            monitor = ModelMonitor(mlflow_host=opt.mlflow_host,
                                   experiment_id=opt.mlflow_experiment_id,
                                   service_endpoint=opt.api_model_version_url,
                                   service_type=SERVICE_TYPE,
                                   content_type=CONTENT_TYPE,
                                   model_type=MODEL_TYPE,
                                   model_path=configs.BASELINE_MODEL_PATH)
            monitor.process()

            logger.info(f'[Jollybuy Goods Multi Item2Item][Model Offline Monitoring] Compare with version : {monitor.model_version}')
            base_model_path = f'model.{monitor.model_version}.pickle'
            base_data_encoders_path = f'encoders.{monitor.model_version}.pickle'
            base_data_encoders = read_pickle(base_data_encoders_path, base_path=configs.BASELINE_MODEL_PATH)

            opt.is_train = False
            baseline_preprocesser = JollybuyGoodsItem2ItemDataPreprocesser(opt=opt, configs=configs,
                                                                           all_cats_file=configs.COL2CATS_NAMES,
                                                                           is_train=opt.is_train,
                                                                           col2label2idx=base_data_encoders,
                                                                           logger=logger)
            dataset = baseline_preprocesser.process()
            baseline_handler = SGDClassifierHandler(opt, JollybuyGoodsMultiItem2ItemConfig)
            baseline_handler.set_params(**params)
            baseline_handler.load_model(model_path=base_model_path, base_path=configs.BASELINE_MODEL_PATH)
            baseline_metrics = baseline_handler.cross_validation(X, Y, k=5, random_state=opt.seed)
            baseline_ndcg = baseline_handler.get_ndcg_score(X, Y, dataset, k=[5, 10, 20])

            comparison_result = {'val_auc': baseline_metrics['auc'][0]}
            for key, score in baseline_ndcg.items():
                comparison_result[f'val_{key}'] = score

            logger.info('[Jollybuy Goods Multi Item2Item][Model Offline Monitoring] Compute performance improvement rate')

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
                logger.info('[Jollybuy Goods Multi Item2Item][Model Offline Monitoring] Send alert mail')

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
                logger.info('[Jollybuy Goods Multi Item2Item][Model Offline Monitoring] No alert mail need to be sent')

            logger.info(f'[Jollybuy Goods Multi Item2Item][Model Offline Monitoring] Show validation result \n{comparison_result}')
            logger.info('[Jollybuy Goods Multi Item2Item][Model Offline Monitoring] Logging validation result')
            mlflow_helper.log_data(data=comparison_result, log_type='metrics')

