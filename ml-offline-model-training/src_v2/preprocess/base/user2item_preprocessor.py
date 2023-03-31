import json
import pandas as pd

from utils.logger import logger
from src.preprocess.utils import convert_type, read_pickle, normalize_data, process_time_period, merge_data_with_user
from src.preprocess.utils.encoder import CategoricalEncoder
from src_v2.preprocess.utils.preference import Preference
from src_v2.preprocess.utils.din import get_behavior_feature
from src_v2.configs import Config


class BaseUser2ItemPreprocessor():

    MODE = ['train', 'validation', 'inference']

    def __init__(self, configs=Config({}), col2label2idx=None, logger=logger, mode='train'):

        assert mode in self.MODE, 'only support train, validation and inference mode'

        self.configs = configs
        self.col2label2idx = col2label2idx if col2label2idx else {}
        self.mode = mode
        self.logger = logger
        self.verbose = self.mode != 'inference'

    def process(self, dataset: pd.DataFrame = None, user_data: pd.DataFrame = None, realtime_user_data: pd.Series = None,
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

            func_kwargs = {**kwargs, **func_params, 'user_data': user_data, 'realtime_user_data': realtime_user_data}
            dataset = chain_func(dataset=dataset, **func_kwargs)

        return dataset[requisite_cols]

    def append_user_data(self, dataset: pd.DataFrame, user_data: pd.Series, realtime_user_data: pd.Series, **kwargs):

        # user profile
        dataset = merge_data_with_user(dataset, user_data, self.configs.APPEND_USER_DATA_PROCESS.get('user_profile', {}))

        # realtime user profile
        if not realtime_user_data.empty:
            dataset = merge_data_with_user(dataset, realtime_user_data, self.configs.APPEND_USER_DATA_PROCESS.get('realtime_user_profile', {}))

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

    def aggregate_preference(self, dataset, **kwargs):

        if self.verbose:
            self.logger.info('---aggregate preference---')

        user_pref_helper = Preference(self.configs)

        for cat_pref_score_paras in self.configs.CATEGORY_PREF_SCORE_PROCESS:
            if self.verbose:
                self.logger.info(f'category pref score: {cat_pref_score_paras}')

            dataset = user_pref_helper.extract_category_pref_score(df=dataset,
                                                                   enable_single_user=self.mode == 'inference',
                                                                   **cat_pref_score_paras)

        for tag_pref_score_paras in self.configs.TAG_PREF_SCORE_PROCESS:
            if self.verbose:
                self.logger.info(f'tag pref score: {tag_pref_score_paras}')

            dataset = user_pref_helper.extract_tag_pref_score(df=dataset,
                                                              enable_single_user=self.mode == 'inference',
                                                              **tag_pref_score_paras)

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

    def process_user_behavior_sequence(self, dataset, **kwargs):

        if self.verbose:
            self.logger.info(f'---process user behavior sequence---')

        max_seq_len = self.configs.BEHAVIOR_SEQUENCE_SIZE

        for col, list_params in self.configs.BEHAVIOR_SEQUENCE_FEATURES_PROCESS.items():
            for params in list_params:

                if self.verbose:
                    self.logger.info(f'col: {col} - params: {params}')

                encoder_key, event_col, level, hist_suffix, seq_len_suffix = params

                dataset = get_behavior_feature(dataset=dataset,
                                               encoder=self.encoder.col2label2idx[encoder_key],
                                               behavior_col=col,
                                               encoder_key=encoder_key,
                                               seq_length_col=f'{seq_len_suffix}{encoder_key}',
                                               prefix=hist_suffix,
                                               profile_key=event_col,
                                               sequence_key=level,
                                               max_sequence_length=max_seq_len)

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
