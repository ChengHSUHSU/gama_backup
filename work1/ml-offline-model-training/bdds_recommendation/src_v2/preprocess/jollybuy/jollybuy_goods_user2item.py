import pandas as pd
from bdds_recommendation.utils.logger import logger
from bdds_recommendation.src.preprocess.utils import process_age
from bdds_recommendation.src_v2.preprocess.base import BaseUser2ItemPreprocessor


class JollybuyGoodsUser2ItemDINPreprocessor(BaseUser2ItemPreprocessor):

    def __init__(self, configs, col2label2idx={}, logger=logger, mode='train', **kwargs):

        super(JollybuyGoodsUser2ItemDINPreprocessor, self).__init__(configs, col2label2idx, logger, mode)

    def process_age(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---process age feature---')

        dataset = process_age(dataset, columns='age', age_range=20)

        return dataset

    def append_metrics_data(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---process metrics feature---')

        dataset['final_score'] = dataset['statistics'] \
            .apply(lambda x: x.get('last_7_day', {}).get('popularity_score', 0.0) if isinstance(x, dict) else 0.0)

        return dataset
