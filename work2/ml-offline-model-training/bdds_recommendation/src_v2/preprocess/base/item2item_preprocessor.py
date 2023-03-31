import json
import pandas as pd

from bdds_recommendation.utils.logger import logger
from bdds_recommendation.src.preprocess.utils import process_embedding, convert_type, read_pickle, normalize_data, \
    process_time_period, merge_data_with_source
from bdds_recommendation.src.preprocess.utils.encoder import CategoricalEncoder
from bdds_recommendation.src_v2.configs import Config


class BaseItem2ItemPreprocessor():

    MODE = ['train', 'validation', 'inference']

    def __init__(self, configs=Config({}), col2label2idx=None, logger=logger, mode='train'):

        assert mode in self.MODE, 'only support train, validation and inference mode'

        self.configs = configs
        self.col2label2idx = col2label2idx if col2label2idx else {}
        self.mode = mode
        self.logger = logger
        self.verbose = self.mode != 'inference'

    def process(self, dataset: pd.DataFrame, source_data: pd.Series = None,
                chain_configs: dict = None, requisite_cols: list = None, **kwargs):

        chain_configs = self.configs.CHAIN_CONFIGS if chain_configs is None else chain_configs
        requisite_cols = self.configs.REQUISITE_COLS if requisite_cols is None else requisite_cols

        for func_name, func_params in chain_configs.items():

            if self.mode in func_params.get('disable_mode', []):
                continue

            try:
                chain_func = getattr(self, func_name)
            except ModuleNotFoundError:
                logger.error(f'Module {func_name} not found.')

            func_kwargs = {**kwargs, **func_params, 'source_data': source_data}
            dataset = chain_func(dataset=dataset, **func_kwargs)

        return dataset[requisite_cols]

    def append_source_data(self, dataset: pd.DataFrame, source_data: pd.Series, prefix='click_', **kwargs):

        dataset = merge_data_with_source(dataset=dataset,
                                         source_data=source_data,
                                         prefix=prefix,
                                         columns=self.configs.APPEND_CANDIDATE_DATA_PROCESS)

        return dataset

    def append_view_also_view_data(self,
                                   dataset: pd.DataFrame,
                                   view_also_view_raw_col: str = 'view_also_view_json',
                                   view_also_view_score_col: str = 'view_also_view_score',
                                   click_id: str = 'click_content_id',
                                   **kwargs):

        dataset[view_also_view_score_col] = dataset.apply(
            lambda x: json.loads(x[view_also_view_raw_col]).get(x[click_id], 0.0)
            if x[view_also_view_raw_col] else 0.0, axis=1)

        return dataset

    def convert_data_type(self, dataset, **kwargs):

        if self.verbose:
            self.logger.info('---convert data type---')

        if self.mode == 'inference':
            dataset = convert_type(dataset, self.configs.TYPE_CONVERT_MODE2COLS_INFERENCE, verbose=self.verbose)
        else:
            dataset = convert_type(dataset, self.configs.TYPE_CONVERT_MODE2COLS, verbose=self.verbose)

        return dataset

    def convert_columns_name(self, dataset, **kwargs):

        if self.verbose:
            self.logger.info('---convert columns name---')

        dataset = dataset.rename(columns=self.configs.COLUMN_TO_RENAME)

        return dataset

    def handle_missing_data(self, dataset, **kwargs):

        if self.verbose:
            self.logger.info('---handle missing data---')

        for key, fillna_value in self.configs.COLUMN_TO_FILLNA.items():

            if self.verbose:
                self.logger.info(f'col: {key} - value: {fillna_value}')

            if isinstance(fillna_value, list):
                dataset[key] = dataset[key].apply(lambda x: x if x else fillna_value)
            elif callable(fillna_value):
                dataset[key] = dataset[key].apply(fillna_value)
            else:
                dataset[key] = dataset[key].fillna(fillna_value)

        return dataset

    def encode_features(self, dataset, prefix='', suffix='', **kwargs):

        if self.verbose:
            self.logger.info('---process categorical data---')

        data_encoder = CategoricalEncoder(col2label2idx=self.col2label2idx)

        if self.mode == 'train' and not self.col2label2idx:

            if kwargs.get('all_cats'):
                all_cats = kwargs['all_cats']
            else:
                all_cats_file = self.configs.COL2CATS_NAMES
                all_cats = read_pickle(all_cats_file, base_path=kwargs.get('dataroot', './dataset'))

            for feature_col, (enable_padding, enable_unknown, mode) in self.configs.CATEGORY_FEATURES_PROCESS.items():

                if self.verbose:
                    self.logger.info(f'column: {feature_col} - mode: {mode}')

                col = feature_col.replace(prefix, '').replace(suffix, '')
                dataset[feature_col] = data_encoder.encode_transform(dataset[feature_col], feature_col, all_cats[col],
                                                                     enable_padding, enable_unknown, mode)
        else:
            for feature_col, (_, _, mode) in self.configs.CATEGORY_FEATURES_PROCESS.items():
                col = feature_col.replace(prefix, '').replace(suffix, '')
                dataset[feature_col] = data_encoder.transform(dataset[feature_col], self.col2label2idx[col], mode)

        self.encoder = data_encoder
        self.col2label2idx = self.encoder.col2label2idx

        return dataset

    def process_time_feature(self, dataset, **kwargs):

        if self.verbose:
            self.logger.info(f'---process time feature---')

        for params in self.configs.PROCESS_TIME_FEATURE:
            dataset = process_time_period(dataset, **params)

        return dataset

    def normalize_data(self, dataset, **kwargs):

        if self.verbose:
            self.logger.info(f'---normalize data---')

        dataset = normalize_data(dataset, self.configs.NORMALIZE_COLS)

        return dataset

    def process_embedding(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---Process embedding data---')

        for process in self.configs.EMBEDDING_PROCESS:

            if self.verbose:
                self.logger.info(f'process {process["embed_col1"]}, {process["embed_col2"]} : {process["mode"]}')

            dataset = process_embedding(dataset=dataset, **process)

        return dataset
