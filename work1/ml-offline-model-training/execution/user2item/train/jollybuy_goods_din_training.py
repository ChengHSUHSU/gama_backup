# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, 'offline-model.zip')

from bdds_recommendation.utils.logger import logger
from bdds_recommendation.utils import download_blob, dump_pickle, read_pickle
from bdds_recommendation.src_v2.factory.jollybuy.jollybuy_goods_user2item import JollybuyGoodsUser2ItemLearn2RankDINFactory
from bdds_recommendation.src.options.train_options import TrainDINOptions
from bdds_recommendation.src.monitoring import MailOperator, ModelMonitor
from bdds_recommendation.src.model_registry import MlflowHelper
from bdds_recommendation.src.validation.cross_validation import cross_validation


MODEL_VERSION = '2.0.1'
MODEL_TYPE = 'din'


if __name__ == '__main__':
    # Parse arguments & store checkpoint
    opt = TrainDINOptions().parse()

    # build mlflow helper
    mlflow_helper = MlflowHelper(opt, logger_name='Jollybuy Goods User2Item', logger=logger)
    mlflow_helper.connection()
    mlflow_helper.set_model_version(model_version=MODEL_VERSION)

    # basic option
    base_path = f'{opt.checkpoints_dir}/{opt.experiment_name}'
    bucket_storage = f'machine-learning-models-{opt.project_id}'

    # initialize factory and training parameters
    jb_goods_u2i_din_factory = JollybuyGoodsUser2ItemLearn2RankDINFactory(mode='train', use_cuda=True, **vars(opt))

    # download and read dataset
    if opt.download_dataset:
        file_to_download = [opt.dataset, jb_goods_u2i_din_factory.config.COL2CATS_NAMES]

        for file in file_to_download:
            logger.info(f'[Jollybuy Goods User2Item][Data Preprocessing] Download {file} from the bucket')
            download_blob(bucket_name=bucket_storage,
                          source_blob_name=f'{opt.dataset_blob_path}/{base_path}/{file}',
                          destination_file_name=f'{opt.dataroot}/{file}')

    dataset = read_pickle(file_name='dataset.pickle', base_path=opt.dataroot)
    all_cats = read_pickle(file_name=jb_goods_u2i_din_factory.config.COL2CATS_NAMES, base_path=opt.dataroot)

    logger.info('[Jollybuy Goods User2Item][Data Preprocessing] Logging dataset information')
    mlflow_helper.log_data(data={'Positive Sample Size': len(dataset[dataset['y'] == 1]),
                                 'Negative Sample Size': len(dataset[dataset['y'] == 0])},
                           log_type='metrics')

    # start training
    logger.info('[Jollybuy Goods User2Item][Model Validation] Start model validation')
    logger.info(f'[Jollybuy Goods User2Item][Model Validation] opt: {vars(opt)}')
    train_params = {'batch_size': opt.batch_size, 'epochs': opt.epochs, 'verbose': opt.verbose}
    p, r, f, auc, roc_curve_info, ndcg = cross_validation(jb_goods_u2i_din_factory, dataset,
                                                          dataset['y'], train_params=train_params,
                                                          groupby_key=jb_goods_u2i_din_factory.config.NDCG_GROUPBY_KEY,
                                                          random_state=opt.seed, **vars(opt))

    validation_result = {
        'val_precision': p[1],
        'val_recall': r[1],
        'val_f1': f[1],
        'val_auc': auc[0]
    }

    for key, score in ndcg.items():
        validation_result[f'val_{key}'] = score

    logger.info(f'[Jollybuy Goods User2Item][Model Validation] Show validation result \n{validation_result}')
    logger.info('[Jollybuy Goods User2Item][Model Validation] Logging validation result')
    mlflow_helper.log_data(data=validation_result, log_type='metrics')

    if opt.save:
        train_factory = JollybuyGoodsUser2ItemLearn2RankDINFactory(mode='train', use_cuda=True, **vars(opt))
        train_factory.train(dataset=dataset, y_true=dataset['y'], train_params=train_params, all_cats=all_cats)

        model_name = f'model.{mlflow_helper.model_version}.pickle'
        encoder_name = f'encoders.{mlflow_helper.model_version}.pickle'
        processed_data_name = f'dataset.{mlflow_helper.model_version}.pickle'

        assets_to_log = {
            encoder_name: train_factory.preprocessor.col2label2idx,
            processed_data_name: train_factory.processed_dataset,
            model_name: train_factory
        }

        for file_name, asset in assets_to_log.items():
            logger.info(f'[Jollybuy Goods User2Item][MLFlow] Store {file_name}')

            if file_name == model_name:
                asset.model_handler.save_model(output_path=os.path.join(base_path, file_name))
            else:
                dump_pickle(file_name, asset, base_path=base_path)

            mlflow_helper.log_artifact(os.path.join(base_path, file_name))

        if opt.deploy:
            mlflow_helper.deploy(base_path, destination_blob_path='online_model/r12n/user2item/jollybuy_goods/din')

    if opt.monitoring:
        logger.info('[Jollybuy Goods User2Item][Model Offline Monitoring] Start offline monitoring')

        monitor = ModelMonitor(mlflow_host=opt.mlflow_host,
                               experiment_id=opt.mlflow_experiment_id,
                               service_endpoint=opt.api_model_version_url,
                               service_type=jb_goods_u2i_din_factory.config.SERVICE_TYPE,
                               content_type=jb_goods_u2i_din_factory.config.CONTENT_TYPE,
                               model_type=MODEL_TYPE,
                               model_path=jb_goods_u2i_din_factory.config.BASELINE_MODEL_PATH)
        monitor.process()

        logger.info(f'[Jollybuy Goods User2Item][Model Offline Monitoring] Compare with version : {monitor.model_version}')
        base_model_path = f'model.{monitor.model_version}.pickle'
        base_col2label2idx_path = f'encoders.{monitor.model_version}.pickle'
        base_col2label2idx = read_pickle(base_col2label2idx_path, base_path=jb_goods_u2i_din_factory.config.BASELINE_MODEL_PATH)

        baseline_factory = JollybuyGoodsUser2ItemLearn2RankDINFactory(mode='validation',
                                                                      col2label2idx=base_col2label2idx,
                                                                      use_cuda=True,
                                                                      model_path=os.path.join(jb_goods_u2i_din_factory.config.BASELINE_MODEL_PATH,
                                                                                              base_model_path),
                                                                      **vars(opt))

        _, _, _, baseline_auc, _, baseline_ndcg = cross_validation(baseline_factory, dataset,
                                                                   dataset['y'], train_params=train_params,
                                                                   groupby_key=baseline_factory.config.NDCG_GROUPBY_KEY)

        comparison_result = {'val_auc': baseline_auc[0]}
        for key, score in baseline_ndcg.items():
            comparison_result[f'val_{key}'] = score

        logger.info(f'[Jollybuy Goods User2Item][Model Validation] Baseline result \n{comparison_result}')
        logger.info('[Jollybuy Goods User2Item][Model Offline Monitoring] Compute performance improvement rate')

        comparison_result = monitor.compare_model_performance(metrics_current=validation_result,
                                                              metrics_compared=comparison_result,
                                                              postfix=jb_goods_u2i_din_factory.config.PERFORMANCE_IMPROVEMENT_RATE_POSTFIX,
                                                              requisite_metrics=jb_goods_u2i_din_factory.config.MONITOR_METRICS_TO_THRESHOLD_MAPPING.keys())

        logger.info(f'[Jollybuy Goods User2Item][Model Validation] Improvement Rate \n{comparison_result}')

        mail_operator = MailOperator(setting=opt)

        alert_list = []
        for monitor_metric, threshold in jb_goods_u2i_din_factory.config.MONITOR_METRICS_TO_THRESHOLD_MAPPING.items():
            value = comparison_result[f'{monitor_metric}{jb_goods_u2i_din_factory.config.PERFORMANCE_IMPROVEMENT_RATE_POSTFIX}']
            if (value > threshold) or (value < -threshold):
                alert_list.append((monitor_metric, value))

        if len(alert_list) > 0:
            logger.info('[Jollybuy Goods User2Item][Model Offline Monitoring] Send alert mail')

            message = mail_operator.get_alert_message(content_type=jb_goods_u2i_din_factory.config.CONTENT_TYPE,
                                                      service_type=jb_goods_u2i_din_factory.config.SERVICE_TYPE,
                                                      model_type=MODEL_TYPE,
                                                      current_model_version=mlflow_helper.model_version,
                                                      online_model_version=monitor.model_version,
                                                      performance_improvement_rate_list=alert_list)

            mail_operator.send_mail(mail_subject='[ml-offline-model-training] offline monitoring alert',
                                    mail_from=opt.mail_server_account,
                                    mail_to=jb_goods_u2i_din_factory.config.ALERT_MAIL_RECIPIENTS,
                                    mail_message=message)
        else:
            logger.info('[Jollybuy Goods User2Item][Model Offline Monitoring] No alert mail need to be sent')

        logger.info('[Jollybuy Goods User2Item][Model Offline Monitoring] Logging validation result')
        mlflow_helper.log_data(data=comparison_result, log_type='metrics')
