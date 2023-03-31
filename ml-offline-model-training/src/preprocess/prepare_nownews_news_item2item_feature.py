from src.preprocess.utils.tag_parser import ner_parser
from src.preprocess.utils import process_embedding, read_pickle
from src.preprocess.utils import convert_type, cal_features_matching
from utils.logger import logger
import pandas as pd
import json


class NownewsNewsItem2ItemDataPreprocesser():
    """The preprocesser support `xgboost` and `wdl` model for now
    """

    def __init__(self, opt, configs, encoders=None, logger=logger):

        self.opt = opt
        self.configs = configs
        self.logger = logger
        self.encoders = encoders if encoders else {}

        # For example , item-item pair column naming would be like :
        #   (page_cat0, click_cat0),
        #   (page_tags, click_tags) ... etc
        self.major_col_prefix = self.configs.MAJOR_COL_PREFIX
        self.minor_col_prefix = self.configs.MINOR_COL_PREFIX

    def process(self, dataset=None, requisite_cols=None):

        if isinstance(dataset, pd.DataFrame) and len(dataset):
            dataset = dataset
        else:
            dataset = self._read_data(dataset)

        # convert data type
        dataset = self._convert_data_type(dataset)

        # proces view also view score
        dataset = self._process_view_also_view(dataset)

        # process embedding
        dataset = self._process_embedding(dataset)

        # parse ner
        dataset = self._parse_ner(dataset)

        # aggregate features
        dataset = self._aggregate_features(dataset)

        if requisite_cols and isinstance(requisite_cols, list):
            return dataset[requisite_cols]

        return dataset

    def _read_data(self, dataset=None):
        self.logger.info(f'---read data---')

        if self.opt.is_train:
            dataset = read_pickle(self.opt.dataset, base_path=self.opt.dataroot)
            dataset = dataset.dropna(subset=self.configs.REQUISITE_INPUT_FEATURE).reset_index()

        return dataset

    def _convert_data_type(self, dataset):
        self.logger.info(f'---convert data type---')
        dataset = convert_type(dataset, self.configs.TYPE_CONVERT_MODE2COLS)

        return dataset

    def _process_view_also_view(self, dataset):
        self.logger.info(f'---process view also view score---')

        vav_input_col, vav_output_col, vav_pair_col = \
            self.configs.VIEW_ALSO_VIEW_COL['input'], \
            self.configs.VIEW_ALSO_VIEW_COL['output'], \
            self.configs.VIEW_ALSO_VIEW_COL['pair']

        dataset[vav_output_col] = dataset.apply(
            lambda x: x[vav_input_col].get(x[vav_pair_col], 0.0) if bool(x[vav_input_col]) else 0.0, axis=1)

        return dataset

    def _process_embedding(self, dataset):
        self.logger.info(f'---process embedding data---')

        major_col = f'{self.major_col_prefix}_item_title_embedding'
        minor_col = f'{self.minor_col_prefix}_item_title_embedding'

        for mode in self.configs.SEMANTICS_INTERACTION_MODE:
            self.logger.info(f'process : {mode}')
            dataset = process_embedding(dataset, major_col, minor_col, mode=mode)

        return dataset

    def _parse_ner(self, dataset):
        self.logger.info(f'---parse content ner data---')

        for col in self.configs.CONTENT_NER_PAIRS:
            self.logger.info(f'process: {col}')
            dataset[col] = dataset[col].apply(lambda x: ner_parser(json.loads(x)) if isinstance(x, str) else {})

        return dataset

    def _aggregate_features(self, dataset):
        self.logger.info(f'---Aggregate features---')

        for col, pairs in self.configs.COLS_TO_AGGREGATE.items():

            self.logger.info(f'process {col}: {pairs}')
            dataset[col] = dataset.apply(
                lambda x: cal_features_matching(x[pairs[0]], x[pairs[1]]), axis=1)

        for col in self.configs.CONTENT_NER_LABEL:

            ner_paris_0 = self.configs.CONTENT_NER_PAIRS[0]
            ner_paris_1 = self.configs.CONTENT_NER_PAIRS[1]

            self.logger.info(f'process: {col}: {(ner_paris_0, ner_paris_1)}')

            dataset[f'ner_{col}_dot'] = dataset.apply(
                lambda x: cal_features_matching(
                    x[ner_paris_0].get(col, []),
                    x[ner_paris_1].get(col, [])), axis=1)

        return dataset
