import os
import pickle
import numpy as np
import pandas as pd
from utils.logger import logger
from src.preprocess.utils import process_embedding, read_pickle, convert_type, cal_features_sim
from src.preprocess.utils.encoder import DataEncoder


class CategoricalEncoder():

    DEFAULT_UNKNOWN_TOKEN = 0

    def __init__(self, enable_unknown, enable_padding, col2label2idx=None):

        self.enable_unknown = enable_unknown
        self.enable_padding = enable_padding
        self.col2label2idx = col2label2idx if col2label2idx else {}

    def _build(self, col, all_cats):

        enc = DataEncoder(enable_padding=self.enable_padding, enable_unknown=self.enable_unknown)
        enc.add_data(all_cats)
        self.col2label2idx[col] = enc.data2index

        return enc

    def _encode(self, data, data2index, code, unknown_token):

        if isinstance(data, list):
            idx = [data2index.get(d, unknown_token) for d in data]
        else:
            idx = [data2index.get(data, unknown_token)]  # default = unknown_token

        if len(idx) == 0:
            features = np.zeros((1, len(data2index)))
        else:
            features = code[idx, :]

        return features

    def transform(self, dataset, col):

        encoded_features = []
        unknown_token = self.col2label2idx[col].get('UNKNOWN', 0)  # default = 0

        code = np.eye(len(self.col2label2idx[col]))

        for data in dataset[col]:
            features = np.sum(self._encode(data, self.col2label2idx[col], code, unknown_token),
                              axis=0).tolist()
            encoded_features.append(features)

        return encoded_features

    def encode_transform(self, dataset, col, all_cats):

        enc = self._build(col, all_cats)
        encoded_features = self.transform(dataset, col)

        return encoded_features


class JollybuyGoodsItem2ItemDataPreprocesser():

    def __init__(self, opt, configs, all_cats_file='col2label.pickle', is_train=True, col2label2idx=None, logger=logger):

        self.opt = opt
        self.configs = configs
        self.all_cats_file = all_cats_file
        self.is_train = is_train
        self.logger = logger
        self.col2label2idx = col2label2idx if col2label2idx else {}

    def process(self, dataset=None, requisite_cols=None):

        dataset = self._read_data(dataset)

        # convert data type
        dataset = self._convert_data_type(dataset)

        # process embedding
        dataset = self._process_embedding(dataset)

        # process categorical features
        dataset = self._process_categorical(dataset)

        # aggregate features
        dataset = self._aggregate_features(dataset)

        if requisite_cols and isinstance(requisite_cols, list):
            return dataset[requisite_cols]

        return dataset

    def _read_data(self, dataset=None):

        if isinstance(dataset, pd.DataFrame) and len(dataset):
            dataset = dataset
        else:
            dataset = read_pickle(self.opt.dataset, base_path=self.opt.dataroot)

        return dataset

    def _convert_data_type(self, dataset):
        self.logger.info('---convert data type---')

        dataset = convert_type(dataset, self.configs.TYPE_CONVERT_MODE2COLS)

        return dataset

    def _process_embedding(self, dataset):
        self.logger.info('---process embedding data---')

        dataset = process_embedding(dataset, 'title_embedding', 'click_title_embedding', mode='prod')
        dataset = process_embedding(dataset, 'title_embedding', 'click_title_embedding', mode='dot')

        return dataset

    def _process_categorical(self, dataset, prefix='click_'):

        encoder = CategoricalEncoder(enable_unknown=True, enable_padding=False, col2label2idx=self.col2label2idx)

        if self.is_train:
            # load saved categorical data
            self.logger.info('---Load categorical feature mapping dictionary---')
            with open(os.path.join(self.opt.dataroot, self.all_cats_file), 'rb') as handle:
                all_cats = pickle.load(handle)

        self.logger.info('---process categorical data---')
        for col in self.configs.COLUMNS_TO_ENCODE:
            self.logger.info(f'process: {col}')

            c = col.replace(prefix, '')

            if self.is_train:
                dataset[col] = encoder.encode_transform(dataset, col, all_cats[c])
            else:
                dataset[col] = encoder.transform(dataset, col)

            self.col2label2idx = encoder.col2label2idx

        return dataset

    def _aggregate_features(self, dataset):
        # aggregate features
        self.logger.info('---Aggregate features---')
        for col, pairs in self.configs.COLS_TO_AGGREGATE.items():

            self.logger.info(f'process {col}: {pairs}')
            dataset[col] = list(map(cal_features_sim(mode='dot'), dataset[pairs[0]], dataset[pairs[1]]))

        return dataset
