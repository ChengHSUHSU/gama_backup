import mlflow

from utils.logger import logger
from utils import upload_blob
from datetime import datetime

HOUR_STRING_FORMAT = '%Y%m%d%H'
MLFLOW_RUN_ID = 'mlflow_run_id'
MODEL_VERSION = 'model_version'
MODEL_RELEASE_DATE = 'model_release_date'
MODEL_RELEASE_TIMESTAMP = 'model_release_timestamp'


class MlflowHelper():

    ENCODERS = f'encoders.model_version.pickle'
    MODEL = f'model.model_version.pickle'

    def __init__(self, setting, logger_name='', logger=logger):

        self.setting = setting
        self.logger = logger
        self.logger_name = f'[{logger_name}][MLFlow]'
        self.bucket_name = f'machine-learning-models-{self.setting.project_id}'

    def connection(self):

        self.MLFLOW_TRACKING_URI = self.setting.mlflow_host
        self.MLFLOW_EXPERIMENT_ID = self.setting.mlflow_experiment_id
        self.MLFLOW_EXPERIMENT_RUN_NAME = self.setting.mlflow_experiment_run_name

        self.logger.info(f'{self.logger_name} connect to mlflow server : {self.MLFLOW_TRACKING_URI}')
        self.logger.info(f'{self.logger_name} experiment id : {self.MLFLOW_EXPERIMENT_ID}')
        self.logger.info(f'{self.logger_name} experiment run name : {self.MLFLOW_EXPERIMENT_RUN_NAME}')

        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_id=self.MLFLOW_EXPERIMENT_ID)
        mlflow.start_run(run_name=self.MLFLOW_EXPERIMENT_RUN_NAME)

        self.MLFLOW_EXPERIMENT_RUN_ID = mlflow.active_run().info.run_id

        self.log_data(data={MLFLOW_RUN_ID: self.MLFLOW_EXPERIMENT_RUN_ID}, log_type='tags')
        self.logger.info(f'{self.logger_name} experiment run_id : {self.MLFLOW_EXPERIMENT_RUN_ID}')

    def close(self):
        self.logger.info(f'{self.logger_name} close mlflow connection')
        mlflow.end_run()

    def log_data(self, data: dict, log_type: str = 'params'):

        if log_type == 'params':
            mlflow.log_params(data)

        elif log_type == 'metrics':
            mlflow.log_metrics(data)

        elif log_type == 'tags':
            mlflow.set_tags(data)

    def log_artifact(self, file_path: str):

        mlflow.log_artifact(file_path)

    def set_model_version(self, model_version: str):

        current_time = datetime.utcnow()
        current_date = current_time.strftime(HOUR_STRING_FORMAT)
        current_timestamp = int(datetime.timestamp(current_time))
        self.model_version = f'{model_version}-{current_date}-{self.MLFLOW_EXPERIMENT_RUN_ID}'

        self.log_data(data={MODEL_VERSION: self.model_version,
                            MODEL_RELEASE_DATE: current_date,
                            MODEL_RELEASE_TIMESTAMP: current_timestamp},
                      log_type='tags')

    def deploy(self, source_path, destination_blob_path):

        self.model_name = self.MODEL.replace(MODEL_VERSION, self.model_version)
        self.encoders_name = self.ENCODERS.replace(MODEL_VERSION, self.model_version)

        model_to_upload = [self.model_name, self.encoders_name]

        for name in model_to_upload:
            upload_blob(bucket_name=self.bucket_name,
                        source_file_name=f'{source_path}/{name}',
                        destination_blob_name=f'{destination_blob_path}/{name}')
            self.logger.info(f'{self.logger_name} upload model {name} success')
