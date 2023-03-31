import json
import pandas as pd
from bdds_recommendation.utils.logger import logger
from bdds_recommendation.src.preprocess.utils import process_time_period, process_age, process_tag
from bdds_recommendation.src_v2.preprocess.base import BaseUser2ItemPreprocessor


class PlanetNewsUser2ItemDINPreprocessor(BaseUser2ItemPreprocessor):

    def __init__(self, configs, col2label2idx: dict = None, logger=logger, mode='train', **kwargs):

        super(PlanetNewsUser2ItemDINPreprocessor, self).__init__(configs, col2label2idx, logger, mode)

    def process_age(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---process age feature---')

        dataset = process_age(dataset, columns='age', age_range=20)

        return dataset

    def process_tag(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---process tag feature---')

        dataset = process_tag(df=dataset, columns=['tags'])

        return dataset

    def process_category(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---process category feature---')

        for col in self.configs.CATEGORY_COLS:
            dataset[col] = dataset[col].apply(lambda x: [x[0]])

        return dataset

    def process_time_period(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---process time period feature---')

        for col in self.configs.UNIX_TIME_COLS:

            if self.verbose:
                self.logger.info(f'process : {col}')

            dataset[col] = dataset[col].apply(lambda x: int(x/1000))

        dataset = process_time_period(dataset,
                                      start_time_col='publish_time',
                                      end_time_col='timestamp',
                                      time_type=['hour'],
                                      postfix='_time_period')

        dataset['hour_time_period'] = dataset['hour_time_period'].apply(lambda x: x if x > 0 else 0)

        return dataset

    def append_metrics_data(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---process metrics feature---')

        dataset['popularity_score'] = dataset['statistics'] \
            .apply(lambda x: x.get('last_7_day', {}).get('popularity_score', 0.0) if isinstance(x, dict) else 0.0)

        dataset['total_click_count'] = dataset['cassandra_statistics'] \
            .apply(lambda x: json.loads(x).get('7_last_day_total_click_count', 0) if isinstance(x, str) else 0.0)

        dataset['total_view_count'] = dataset['cassandra_statistics'] \
            .apply(lambda x: json.loads(x).get('7_last_day_total_view_count', 0) if isinstance(x, str) else 0.0)

        return dataset
