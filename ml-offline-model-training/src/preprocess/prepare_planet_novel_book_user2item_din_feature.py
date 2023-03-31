import numpy as np
import pandas as pd
import random
from src.preprocess.utils import read_pickle, process_age, convert_type, \
    get_publish_time_to_now, normalize_data
from src.preprocess.utils.encoder import CategoricalEncoder, UNKNOWN_LABEL
from src.preprocess.utils.din import get_behavior_feature
from src.preprocess.utils.preference import Preference
from utils.logger import logger


class PlanetNovelBookUser2ItemDataPreprocessor:

    def __init__(self, opt, configs, encoders=None, logger=logger):

        self.opt = opt
        self.configs = configs
        self.logger = logger
        self.encoders = encoders if encoders else {}

    def process(self, dataset=None, requisite_cols=None):

        if isinstance(dataset, pd.DataFrame) and len(dataset):
            dataset = dataset
        else:
            dataset = self._read_data(dataset)

        # convert data type
        dataset = self._convert_data_type(dataset)

        # convert columns name
        dataset = self._convert_columns_name(dataset)

        # handle missing data
        dataset = self._handle_missing_data(dataset)

        # process category
        dataset = self._process_category(dataset)

        # process embedding
        dataset = self._process_embedding(dataset)

        # aggregate preference
        dataset = self._aggregate_preference(dataset)

        # process categorical features
        dataset = self._encode_features(dataset)

        # process user behavior sequence
        dataset = self._process_user_behavior_sequence(dataset)

        # process time feature
        dataset = self._process_time_feature(dataset)

        # process data normalization
        dataset = self._normalize_data(dataset)

        if requisite_cols and isinstance(requisite_cols, list):
            return dataset[requisite_cols]

        return dataset

    def _read_data(self, dataset=None):

        self.logger.info('---read data---')
        if self.opt.is_train:
            dataset = read_pickle(self.opt.dataset, base_path=self.opt.dataroot)

        return dataset

    def _convert_data_type(self, dataset):

        self.logger.info('---convert data type---')
        dataset = convert_type(dataset, self.configs.TYPE_CONVERT_MODE2COLS)
        dataset['chapter_newest_first_publish_time'] = dataset['chapter_newest_first_publish_time'].astype(int)
        dataset['timestamp'] = dataset['timestamp'].astype(int)
        dataset['display_author'] = dataset['display_author'].apply(lambda x: 1 if x is not None else 0)

        return dataset

    def _convert_columns_name(self, dataset):

        self.logger.info('---convert columns name---')
        dataset = dataset.rename(columns=self.configs.COLUMN_TO_RENAME)

        return dataset

    def _handle_missing_data(self, dataset):

        self.logger.info('---handle missing data of age, gender and series---')
        dataset = process_age(dataset, columns='age', age_range=20)

        for key, fillna_value in self.configs.COLUMN_TO_FILLNA.items():
            self.logger.info(key)
            if isinstance(fillna_value, list):
                dataset[key] = dataset[key].apply(lambda x: x if x else fillna_value)
            else:
                dataset[key] = dataset[key].fillna(fillna_value)

        return dataset

    def _process_embedding(self, dataset, dim=300, fillna=True):

        self.logger.info('---process user title, user tag, and item embedding---')
        embedding_col_list = self.configs.EMBEDDING_COL_LIST

        for embed_col in embedding_col_list:
            if fillna:
                dataset[embed_col] = dataset[embed_col].apply(lambda x: x if isinstance(x, list) and len(x) == dim else [0.0] * dim)

            dataset[embed_col] = np.array(dataset[embed_col])

        return dataset

    def _process_category(self, dataset):

        self.logger.info('---process category ---')

        dataset['cat1'] = dataset['cat1'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else UNKNOWN_LABEL)

        return dataset

    def _aggregate_preference(self, dataset):

        self.logger.info('---Aggregate preference---')
        user_pref_helper = Preference(self.configs)

        self.logger.info('process cat1')
        dataset = user_pref_helper.extract_category_pref_score(dataset, level=['click', 'cat1'])

        self.logger.info('process tags')
        dataset['tags'] = dataset['tags'].apply(lambda x: user_pref_helper._parse_tags_from_string(x) if pd.notna(x) else [])

        if self.opt.is_train:
            enable_single_user = False
        else:
            enable_single_user = True

        self.logger.info('process tags: editor')
        dataset = user_pref_helper.extract_tag_pref_score(dataset, tagging_type='editor', enable_single_user=enable_single_user)

        return dataset

    def _encode_features(self, dataset, prefix='', suffix=''):

        self.logger.info('---process categorical data---')
        data_encoder = CategoricalEncoder(col2label2idx={})

        # encode categorical features
        self.logger.info('Load categorical feature mapping dictionary')
        all_cats_file = getattr(self.configs, 'COL2CATS_NAMES', 'col2label.pickle')
        all_cats = read_pickle(all_cats_file, base_path=self.opt.dataroot)

        for feature_col, (enable_padding, enable_unknown, mode) in self.configs.CATEGORY_FEATURES_PROCESS.items():
            self.logger.info(feature_col)
            col = feature_col.replace(prefix, '').replace(suffix, '')
            dataset[feature_col] = data_encoder.encode_transform(dataset[feature_col], feature_col, all_cats[col],
                                                                 enable_padding, enable_unknown, mode)

        self.encoder = data_encoder

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
                                               encoder=self.encoder.col2label2idx[encoder_key],
                                               behavior_col=col,
                                               seq_length_col=f'{seq_len_suffix}{encoder_key}',
                                               prefix=hist_suffix,
                                               profile_key=event_col,
                                               sequence_key=level,
                                               max_sequence_length=self.configs.BEHAVIOR_SEQUENCE_SIZE,
                                               mode='training')

        return dataset

    def _process_time_feature(self, dataset):
        self.logger.info('---process time feature---')

        dataset = get_publish_time_to_now(
            dataset, event_time_col='timestamp', publish_time_col='chapter_newest_first_publish_time', time_type=['hour'])

        return dataset

    def _normalize_data(self, dataset):
        self.logger.info('---normalize data---')

        dataset = normalize_data(dataset, self.configs.NORMALIZE_COLS)

        return dataset
