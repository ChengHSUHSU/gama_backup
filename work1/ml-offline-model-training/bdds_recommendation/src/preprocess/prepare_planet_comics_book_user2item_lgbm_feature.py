from bdds_recommendation.src.preprocess.utils import process_embedding, read_pickle
from bdds_recommendation.src.preprocess.utils.encoder import CategoricalEncoder
from bdds_recommendation.src.preprocess.utils.preference import Preference
from bdds_recommendation.src.preprocess.utils import convert_type
import ast
from bdds_recommendation.utils.logger import logger
import pandas as pd
import numpy as np


class PlanetComicsBookUser2ItemDataPreprocessor():
    """The preprocessor support `lgbm` model for now"""

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
        # cat1 and embedding to array, count information from content to int
        dataset = self._convert_data_type(dataset)

        # rename columns name according to COLUMN_TO_RENAME
        dataset = self._convert_columns_name(dataset)

        # get concat of all cat1 string
        dataset = self._parse_planet_category(dataset)

        # handle age/gender missing data
        dataset = self._handle_missing_data(dataset)

        # using dot to calculate user/item embedding similarity
        dataset = self._process_embedding(dataset)

        # extract user_category / category_pref_score / tag_pref_score
        dataset = self._aggregate_preference(dataset)

        # one-hot encoding for categorical features
        dataset = self._encode_features(dataset)

        return dataset

    def _read_data(self, dataset=None):
        self.logger.info(f'---read data---')
        dataset = read_pickle(self.opt.dataset, base_path=self.opt.dataroot)

        return dataset

    def _convert_columns_name(self, dataset):
        self.logger.info(f'---convert columns name---')

        dataset = dataset.rename(columns=self.configs.COLUMN_TO_RENAME)

        return dataset

    def _handle_missing_data(self, dataset):
        self.logger.info(f'---handle missing data---')

        df_tmp = dataset[dataset['age'].notnull()]
        df_tmp['age'] = df_tmp.age.astype(int)
        median_age = int(df_tmp.age.median())

        dataset['age'] = dataset.age.fillna(median_age)
        dataset['age'] = dataset.age.apply(lambda x: median_age if int(x) < 0 else int(x))
        dataset['gender'] = dataset.gender.fillna(CategoricalEncoder.UNKNOWN_LABEL)

        return dataset

    def _convert_data_type(self, dataset):
        self.logger.info(f'---convert data type---')

        dataset = convert_type(dataset, self.configs.TYPE_CONVERT_MODE2COLS)

        dataset['read_count_repeat'] = dataset.fillna(0).read_count_repeat.astype(int)
        dataset['read_count_norepeat'] = dataset.fillna(0).read_count_norepeat.astype(int)
        dataset['like_count'] = dataset.fillna(0).like_count.astype(int)

        return dataset

    def _process_embedding(self, dataset, result_col_prefix=[]):
        self.logger.info(f'---process embedding data---')

        major_col = self.configs.SEMANTICS_FEATURE[0]
        minor_col = self.configs.SEMANTICS_FEATURE[1]

        for mode in self.configs.SEMANTICS_INTERACTION_MODE:
            self.logger.info(f'process : {mode}')
            dataset = process_embedding(dataset, major_col, minor_col, mode=mode)

        return dataset

    def _aggregate_preference(self, dataset):
        self.logger.info(f'---Aggregate preference---')

        user_pref_helper = Preference(self.configs)

        self.logger.info(f'process cat1')
        dataset = user_pref_helper.extract_category_pref_score(dataset, level=['click', 'cat1'])

        self.logger.info(f'process tags')

        # ['原創新作', '原子少年', '男團選秀節目'] -> [原創新作, 原子少年, 男團選秀節目]
        dataset['tags'] = dataset['tags'].apply(lambda x: user_pref_helper._parse_tags_from_string(x) if pd.notna(x) else [])

        if self.opt.is_train:
            enable_single_user = False
        else:
            enable_single_user = True

        self.logger.info(f'process tags: editor')
        # TODO: currently user_tag_editor_{others,person,event,organization,location} has no importance, improve it
        # dataset = user_pref_helper.extract_tag_pref_score(dataset, tagging_type='editor', enable_single_user=enable_single_user)

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

    def _parse_planet_category(self, dataset):
        dataset['cat1'] = dataset['cat1'].transform(lambda x: ','.join(map(str, x)))
        return dataset
