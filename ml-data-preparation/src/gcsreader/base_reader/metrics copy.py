from datetime import datetime, timedelta
from src.gcsreader.utils import rename_cols, extend_generated_time
from src.gcsreader.base_reader import GcsReader
from utils import get_partition_path_with_blobs_check, DAY_STRING_FORMAT
import logging


class MetricsReader(GcsReader):
    def __init__(self, project_id, sql_context, run_time, days=30, config=None, logger=logging):
        super().__init__(project_id, sql_context, run_time, days, config, logger)
        self.data = dict()

    def process(self, content_type, metrics_type):
        """Function to process metrics reader pipeline

        Args:
            content_type (str)
        """
        if metrics_type == 'snapshot_popularity':
            self.logger.info(f'[Data Preparation][Metrics] get {content_type} latest popularity score data')
            popularity_path = self.config.METRICS_POPULARITY_PATH.replace('PROJECT_ID', self.project_id) \
                .replace('INPUT_DATE/INPUT_HOUR', 'latest') \
                .replace('POPULARITY_FOLDER', self.config.POPULARITY_FOLDER[content_type])

            df = self.sql_context.read.options(**{'header': 'true', 'escape': '"'}).csv(popularity_path)

        elif metrics_type == 'snapshot_statistics':
            self.logger.info(f'[Data Preparation][Metrics] get {content_type} latest statistics data')
            statistics_path = self.config.STATISTICS_PATH.replace('PROJECT_ID', self.project_id) \
                .replace('INPUT_DATE/INPUT_HOUR', 'latest') \
                .replace('STATISTICS_FOLDER', self.config.STATISTICS_FOLDER[content_type])

            df = self.sql_context.read.options(**{'header': 'true', 'escape': '"'}).csv(statistics_path)
            # rename columns
            df = rename_cols(df, rename_cols=self.config.STEPS_TO_RENAMCOLS.get('metrics', []).get(metrics_type, []))

        elif metrics_type == 'popularity':
            self.logger.info(f'[Data Preparation][Metrics] get {content_type} popularity score data')
            popularity_folder = self.config.POPULARITY_FOLDER[content_type]
            popularity_path = self.config.METRICS_POPULARITY_PATH
            popularity_path = (
                popularity_path
                .replace('PROJECT_ID', self.project_id)
                .replace('POPULARITY_FOLDER', popularity_folder)
            )
            path_list = self._get_existed_metrics_path(blob_path=popularity_path, prefix='metrics/popularity', content_folder=popularity_folder)
            df = self.sql_context.read.options(**{'header': 'true', 'escape': '"'}).csv(path_list)

            self.logger.info('[Data Preparation][Metrics] extend generated time')
            df = extend_generated_time(df, prefix_name=popularity_folder, time_format='hour')
            df = df.dropDuplicates(subset=['uuid', 'date', 'hour'])

        elif metrics_type == 'statistics':
            self.logger.info(f'[Data Preparation][Metrics] get {content_type} statistics score data')
            statistic_folder = self.config.STATISTICS_FOLDER[content_type]
            statistic_path = self.config.STATISTICS_PATH
            statistic_path = (
                statistic_path
                .replace('PROJECT_ID', self.project_id)
                .replace('STATISTICS_FOLDER', statistic_folder)
            )
            path_list = self._get_existed_metrics_path(blob_path=statistic_path, prefix='statistics', content_folder=statistic_folder)

            df = self.sql_context.read.options(**{'header': 'true', 'escape': '"'}).csv(path_list)
            self.logger.info('[Data Preparation][Metrics] extend generated time')
            df = extend_generated_time(df, prefix_name=statistic_folder, time_format='hour')
            df = df.dropDuplicates(subset=['uuid', 'date', 'hour'])

        self.logger.info('[Data Preparation][Metrics] rename columns')
        df = rename_cols(df, rename_cols=self.config.STEPS_TO_RENAMCOLS.get('metrics', []).get(metrics_type, []))
        self.logger.info('[Data Preparation][Metrics] select requisite columns')
        self.data[metrics_type] = df.select(self.config.REQUISITE_COLS.get('metrics', {}).get(metrics_type, '*'))

    def _get_existed_metrics_path(self, content_type) -> list:
        """Private function to get the existing metircs path

        Args:
            content_type (str)

        Returns:
            list
        """

        all_dates = []
        all_hours = []
        all_path_list = []

        popularity_folder = self.config.POPULARITY_FOLDER[content_type]
        popularity_path = self.config \
            .METRICS_POPULARITY_PATH.replace('PROJECT_ID', self.project_id) \
            .replace('POPULARITY_FOLDER', popularity_folder)
        
        for d in range(int(self.days)):
            current_date = (datetime.strptime(self.run_time, DAY_STRING_FORMAT) - timedelta(days=d)).strftime(DAY_STRING_FORMAT)

            for h in range(24):
                current_hour = (datetime.strptime(str(h), '%H')).strftime('%H')
                prefix = f'metrics/popularity/{popularity_folder}/{current_date}/{current_hour}'
                path = popularity_path.replace('INPUT_DATE', current_date).replace('INPUT_HOUR', current_hour)
                cur_path_list, _ = get_partition_path_with_blobs_check(path, prefix, bucket_idx=2)

                if len(cur_path_list) > 0:
                    all_path_list.extend(cur_path_list)
                    all_dates.append(current_date)
                    all_hours.append(current_hour)

        return all_path_list
