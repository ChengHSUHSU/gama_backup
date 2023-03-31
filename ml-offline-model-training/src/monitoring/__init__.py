import mlflow
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from mlflow.tracking import MlflowClient
from src.monitoring.utils import get_model_description
from utils import mkdir


class ModelMonitor():

    def __init__(self, mlflow_host, experiment_id, service_endpoint, service_type, content_type, model_type, model_path='./ml-model'):

        self.mlflow_host = mlflow_host
        self.experiment_id = str(experiment_id)
        self.service_endpoint = service_endpoint
        self.service_type = service_type
        self.content_type = content_type
        self.model_type = model_type
        self.model_path = model_path

    def process(self):

        # get service model version online
        self.get_model_version()

        # set connection of model registry
        self._set_mlflow_connect()

        # download service model from model registry
        #   for example `model_version`: 0.0.1-2022111111-06757104292346789dbf149cb68b049f (semantic_version-datetime-experiment_id)
        #   we need to split `model_version` by '-' to get mlflow experiment_id for downloading
        self._download_experiment_data(run_id=self.model_version.split('-')[-1])

    def get_model_version(self):

        model_descript = get_model_description(self.service_endpoint)[0]
        self.model_version = model_descript.get(self.service_type, {}).get(self.content_type, {}).get('default', {}).get('version', None)

        return self.model_version

    def _download_experiment_data(self, run_id=None, source_path=''):

        mkdir(self.model_path)
        this_run_id = self.model_version if run_id is None else run_id
        MlflowClient().download_artifacts(this_run_id, source_path, self.model_path)

    def _set_mlflow_connect(self):

        mlflow.set_tracking_uri(self.mlflow_host)
        mlflow.set_experiment(experiment_id=self.experiment_id)

    def compare_model_performance(self, metrics_current: dict, metrics_compared: dict, metric_prefix='val_', postfix='_pir', requisite_metrics=['auc', 'ndcg5', 'ndcg10', 'ndcg20']):
        """method for computing performance improvement rate (PIR) between `metrics_current` and `metrics_compared`

        Args:
            metrics_current (dict): validation results of current model
            metrics_compared (dict): validation results of compared model
            metric_prefix (str, optional): prefix of dictionary key . Defaults to 'val_'.
            postfix (str, optional): postfix of dictionary key. Defaults to '_pir', means `Performance Improvement Rate`.
            requisite_metrics (list, optional): metrics list to compute PIR. Defaults to ['auc', 'ndcg5', 'ndcg10', 'ndcg20'].

        Returns:
            result (dics): compared result
        """

        result = {}

        for metric_type in requisite_metrics:
            metric_cur = metrics_current[f'{metric_prefix}{metric_type}']
            metric_com = metrics_compared[f'{metric_prefix}{metric_type}']
            performance_improvement_rate = (metric_cur-metric_com)/metric_com
            performance_improvement_rate = round(performance_improvement_rate, 5) * 100
            result[f'{metric_type}{postfix}'] = performance_improvement_rate

        return result


class MailOperator():

    def __init__(self, setting):

        self.setting = setting

    def get_alert_message(self, content_type, service_type, model_type, current_model_version, online_model_version, performance_improvement_rate_list: list):

        message_template = f"""
                            <p><span style="font-family: 'Lucida Sans Unicode', 'Lucida Grande', sans-serif;">Hi, all</span></p>
                            <p><span style="font-family: 'Lucida Sans Unicode', 'Lucida Grande', sans-serif;">We might want to take a look at following model:</span></p>
                            <p><br></p>
                            <p><span style="font-family: 'Lucida Sans Unicode', 'Lucida Grande', sans-serif;">  - content_type: CONTENT_TYPE</span></p>
                            <p><span style="font-family: 'Lucida Sans Unicode', 'Lucida Grande', sans-serif;">  - service_type: SERVICE_TYPE</span></p>
                            <p><span style="font-family: 'Lucida Sans Unicode', 'Lucida Grande', sans-serif;">  - model_type: MODEL_TYPE</span></p>
                            <p><span style="font-family: 'Lucida Sans Unicode', 'Lucida Grande', sans-serif;">  - current model_version: CURRENT_MODEL_VERSION</span></p>
                            <p><span style="font-family: 'Lucida Sans Unicode', 'Lucida Grande', sans-serif;">  - online model_version: ONLINE_MODEL_VERSION</span></p>
                            <p><br></p>"""

        mail_message = message_template. \
            replace('CONTENT_TYPE', content_type). \
            replace('SERVICE_TYPE', service_type). \
            replace('MODEL_TYPE', model_type). \
            replace('CURRENT_MODEL_VERSION', current_model_version). \
            replace('ONLINE_MODEL_VERSION', online_model_version)

        for pair_data in performance_improvement_rate_list:
            metric, performance_improvement_rate = pair_data
            mail_message += f"""<p><span style="font-family: 'Lucida Sans Unicode', 'Lucida Grande', sans-serif;">{metric} is {performance_improvement_rate} % than online model </span></p>"""
        mail_message += """<p><br></p><p><span style="font-family: 'Lucida Sans Unicode', 'Lucida Grande', sans-serif;">You can see more information about this experiment in MLFlow server : http://34.149.189.66/</span></p>"""

        return mail_message

    def send_mail(self, mail_subject, mail_from, mail_to, mail_message):
        content = MIMEMultipart()
        content['subject'] = mail_subject
        content['from'] = mail_from
        content['to'] = mail_to

        content.attach(MIMEText(mail_message, 'html'))

        with smtplib.SMTP(host=self.setting.mail_server, port=self.setting.mail_server_port) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(self.setting.mail_server_account, self.setting.mail_server_password)
            smtp.send_message(content)
