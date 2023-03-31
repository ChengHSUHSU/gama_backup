import pandas as pd
from bdds_recommendation.utils.logger import logger
from bdds_recommendation.src.preprocess.utils import process_embedding, parse_content_ner, cal_ner_matching, cal_features_sim
from bdds_recommendation.src_v2.preprocess.base import BaseItem2ItemPreprocessor


class PlanetNewsItem2ItemSGDPreprocessor(BaseItem2ItemPreprocessor):

    def __init__(self, configs, col2label2idx: dict = None, logger=logger, mode='train', **kwargs):

        super(PlanetNewsItem2ItemSGDPreprocessor, self).__init__(configs, col2label2idx, logger, mode)

    def process_embedding(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---Process embedding data---')

        for process in self.configs.EMBEDDING_PROCESS:

            if self.verbose:
                self.logger.info(f'process {process["embed_col1"]}, {process["embed_col2"]} : {process["mode"]}')

            dataset = process_embedding(dataset=dataset, **process)

        return dataset

    def parse_ner(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---Parse content ner data---')

        dataset = parse_content_ner(dataset, self.configs.CONTENT_NER_COLS_MAP)

        return dataset

    def cal_ner_matching(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---Calculate NER matching scores---')

        for col, pairs in self.configs.COLS_OF_NER_MATCHING.items():

            if self.verbose:
                self.logger.info(f'process {col}: {pairs}')

            dataset[col] = list(map(cal_ner_matching, dataset[pairs[0]], dataset[pairs[1]]))

        return dataset

    def aggregate_features(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---Aggregate features---')

        for col, pairs in self.configs.COLS_TO_AGGREGATE.items():

            if self.verbose:
                self.logger.info(f'process {col}: {pairs}')

            dataset[col] = list(map(cal_features_sim(mode='dot'), dataset[pairs[0]], dataset[pairs[1]]))

        return dataset
