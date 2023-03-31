from src.preprocess.utils import process_embedding, read_pickle
from src.preprocess.utils.encoder import CategoricalEncoder
from src.preprocess.utils.din import get_behavior_feature
from src.preprocess.utils.preference import Preference
from src.preprocess.utils import convert_type, normalize_data, process_age
from utils.logger import logger
import pandas as pd


class JollybuyGoodsUser2ItemTrainPreprocesser():

    def __init__(self, opt, configs, col2label2idx=None, logger=logger):

        self.opt = opt
        self.configs = configs
        self.logger = logger
        self.col2label2idx = col2label2idx if col2label2idx else {}

    def process(self, dataset=None, requisite_cols=None):

        # read data
        dataset = self._read_data(dataset)

        # convert data type
        dataset = self._convert_data_type(dataset)

        # convert columns name
        dataset = self._convert_columns_name(dataset)

        # special encoding to age feature
        dataset = self._process_age(dataset)

        # handle missing data
        dataset = self._handle_missing_data(dataset)

        # aggregate preference
        dataset = self._aggregate_preference(dataset)

        # process categorical features
        dataset = self._encode_features(dataset)

        # process user behavior sequence
        dataset = self._process_user_behavior_sequence(dataset)

        # normalize data
        dataset = self._normalize_data(dataset)

        if requisite_cols and isinstance(requisite_cols, list):
            return dataset[requisite_cols]

        return dataset

    def _read_data(self, dataset=None):
        self.logger.info(f'---read data---')

        if isinstance(dataset, pd.DataFrame) and len(dataset):
            dataset = dataset
        else:
            dataset = read_pickle(self.opt.dataset, base_path=self.opt.dataroot)

        return dataset

    def _handle_missing_data(self, dataset):
        self.logger.info(f'---handle missing data---')

        for key, fillna_value in self.configs.COLUMN_TO_FILLNA.items():
            self.logger.info(key)
            if isinstance(fillna_value, list):
                dataset[key] = dataset[key].apply(lambda x: x if x else fillna_value)
            else:
                dataset[key] = dataset[key].fillna(fillna_value)

        return dataset

    def _convert_data_type(self, dataset):
        self.logger.info(f'---convert data type---')

        dataset = convert_type(dataset, self.configs.TYPE_CONVERT_MODE2COLS)

        return dataset

    def _convert_columns_name(self, dataset):
        self.logger.info(f'---convert columns name---')

        dataset = dataset.rename(columns=self.configs.COLUMN_TO_RENAME)

        return dataset

    def _process_age(self, dataset):
        self.logger.info(f'---process age feature---')

        dataset = process_age(dataset, columns='age', age_range=20)

        return dataset

    def _aggregate_preference(self, dataset):
        self.logger.info(f'---Aggregate preference---')

        user_pref_helper = Preference(self.configs)

        self.logger.info(f'cat0')
        dataset = user_pref_helper.extract_category_pref_score(df=dataset,
                                                               level=['click', 'cat0'],
                                                               cat_col=self.configs.USER_CAT_COL,
                                                               score_col=self.configs.CAT0_PREF_COL)

        self.logger.info(f'cat1')
        dataset = user_pref_helper.extract_category_pref_score(df=dataset,
                                                               level=['click', 'cat1'],
                                                               cat_col=self.configs.USER_CAT_COL,
                                                               score_col=self.configs.CAT1_PREF_COL)

        if self.opt.is_train:
            enable_single_user = False
        else:
            enable_single_user = True

        self.logger.info(f'tag')
        dataset = user_pref_helper.extract_tag_pref_score(df=dataset,
                                                          user_tag_col=self.configs.USER_TAG_COL,
                                                          item_tag_col=self.configs.ITEM_TAG_COL,
                                                          tagging_type='editor',
                                                          score_col=self.configs.TAG_PREF_COL,
                                                          enable_single_user=enable_single_user)

        return dataset

    def _encode_features(self, dataset, prefix='', suffix=''):

        self.logger.info('---process categorical data---')
        data_encoder = CategoricalEncoder(col2label2idx=self.col2label2idx if self.col2label2idx else {})

        if self.opt.is_train and not bool(self.col2label2idx):
            # encode categorical features
            self.logger.info('Load categorical feature mapping dictionary')
            all_cats_file = getattr(self.configs, 'COL2CATS_NAMES', 'col2label.pickle')
            all_cats = read_pickle(all_cats_file, base_path=self.opt.dataroot)

            self.logger.info(f'---encode transform feature---')
            for feature_col, (enable_padding, enable_unknown, mode) in self.configs.CATEGORY_FEATURES_PROCESS.items():
                self.logger.info(feature_col)
                col = feature_col.replace(prefix, '').replace(suffix, '')
                dataset[feature_col] = data_encoder.encode_transform(dataset[feature_col], feature_col, all_cats[col],
                                                                     enable_padding, enable_unknown, mode)
        else:
            self.logger.info(f'---transform feature---')
            for feature_col, (enable_padding, enable_unknown, mode) in self.configs.CATEGORY_FEATURES_PROCESS.items():
                self.logger.info(feature_col)
                dataset[feature_col] = data_encoder.transform(dataset[feature_col], data_encoder.col2label2idx, mode)

        if self.opt.is_train:
            self.encoder = data_encoder
            self.col2label2idx = data_encoder.col2label2idx

        return dataset

    def _process_user_behavior_sequence(self, dataset):
        self.logger.info(f'---process user behavior sequence---')

        hist_suffix = 'hist_'
        seq_len_suffix = 'seq_length_'

        for col, list_params in self.configs.BEHAVIOR_SEQUENCE_FEATURES_PROCESS.items():
            for params in list_params:
                self.logger.info(f'Process {col}: {params}')

                encoder_key, event_col, level = params

                dataset = get_behavior_feature(dataset=dataset,
                                               encoder=self.col2label2idx[encoder_key],
                                               behavior_col=col,
                                               encoder_key=encoder_key,
                                               seq_length_col=f'{seq_len_suffix}{encoder_key}',
                                               prefix=hist_suffix,
                                               profile_key=event_col,
                                               sequence_key=level,
                                               max_sequence_length=self.configs.BEHAVIOR_SEQUENCE_SIZE,
                                               mode='training')

        return dataset

    def _normalize_data(self, dataset):
        self.logger.info(f'---normalize data---')

        dataset = normalize_data(dataset, self.configs.NORMALIZE_COLS)

        return dataset
