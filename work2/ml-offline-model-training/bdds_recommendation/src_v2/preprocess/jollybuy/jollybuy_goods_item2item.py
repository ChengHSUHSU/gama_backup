import pandas as pd
from bdds_recommendation.utils.logger import logger
from bdds_recommendation.src.preprocess.utils import process_embedding, cal_features_sim
from bdds_recommendation.src_v2.preprocess.base import BaseItem2ItemPreprocessor


class JollybuyGoodsItem2ItemSGDPreprocessor(BaseItem2ItemPreprocessor):

    def __init__(self, configs, col2label2idx: dict = None, logger=logger, mode='train', **kwargs):

        super(JollybuyGoodsItem2ItemSGDPreprocessor, self).__init__(configs, col2label2idx, logger, mode)

    def aggregate_features(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---Aggregate features---')

        for col, pairs in self.configs.COLS_TO_AGGREGATE.items():

            if self.verbose:
                self.logger.info(f'process {col}: {pairs}')

            dataset[col] = list(map(cal_features_sim(mode='dot'), dataset[pairs[0]], dataset[pairs[1]]))

        return dataset
