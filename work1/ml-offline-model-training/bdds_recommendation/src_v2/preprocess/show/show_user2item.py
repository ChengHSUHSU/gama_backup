import pandas as pd
from bdds_recommendation.utils.logger import logger
from bdds_recommendation.src.preprocess.utils import process_time_period, process_age
from bdds_recommendation.src_v2.preprocess.base import BaseUser2ItemPreprocessor
from bdds_recommendation.src.preprocess.utils.encoder import UNKNOWN_LABEL


class ShowUser2ItemDINPreprocessor(BaseUser2ItemPreprocessor):

    def __init__(self, configs, col2label2idx: dict = None, logger=logger, mode='train', **kwargs):

        super(ShowUser2ItemDINPreprocessor, self).__init__(configs, col2label2idx, logger, mode)

    def process_age(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---process age feature---')

        dataset = process_age(dataset, columns='age', age_range=20)

        return dataset

    def process_category(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---process category feature---')

        for col in self.configs.CATEGORY_COLS:
            dataset[col] = dataset[col].apply(lambda x: [x[0]] if len(x) >= 1 else [UNKNOWN_LABEL])

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
