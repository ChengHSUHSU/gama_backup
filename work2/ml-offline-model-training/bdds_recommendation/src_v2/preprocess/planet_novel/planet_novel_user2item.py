import pandas as pd
from bdds_recommendation.utils.logger import logger
from bdds_recommendation.src.preprocess.utils import process_age
from bdds_recommendation.src_v2.preprocess.base import BaseUser2ItemPreprocessor
from bdds_recommendation.src.preprocess.utils.encoder import UNKNOWN_LABEL


class PlanetNovelUser2ItemDINPreprocessor(BaseUser2ItemPreprocessor):

    def __init__(self, configs, col2label2idx: dict = None, logger=logger, mode='train', **kwargs):

        super(PlanetNovelUser2ItemDINPreprocessor, self).__init__(configs, col2label2idx, logger, mode)

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

    def get_publish_time_to_now(self, dataset: pd.DataFrame, event_time_col='timestamp', publish_time_col='publish_time', time_type=['min', 'hour'], **kwargs) -> pd.DataFrame:
        """TODO: For computing time period, use `process_time_period` instead of this from now on
        """

        time_type_to_sec = {'min': 60, 'hour': 3600}

        for t in time_type:

            dataset[f'{t}_to_current'] = dataset.apply(
                lambda x: (int(x[event_time_col])-int(x[publish_time_col])) / (int(time_type_to_sec[t])*1000), axis=1)

        return dataset
