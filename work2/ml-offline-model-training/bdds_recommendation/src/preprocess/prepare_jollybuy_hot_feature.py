import os
import pickle
import numpy as np
import pandas as pd
from bdds_recommendation.utils.logger import logger
from bdds_recommendation.src.preprocess.utils import read_pickle, convert_type, calculate_min_max
from bdds_recommendation.src.preprocess.utils.encoder import DataEncoder, UNKNOWN_LABEL


class CategoricalEncoder():

    DEFAULT_UNKNOWN_TOKEN = 0

    def __init__(self, enable_unknown, enable_padding, col2label2idx=None):

        self.enable_unknown = enable_unknown
        self.enable_padding = enable_padding
        self.col2label2idx = col2label2idx if col2label2idx else {}

    def transform(self, dataset, col):

        encoded_features = []
        unknown_token = self.col2label2idx[col].get(UNKNOWN_LABEL, 0)

        code = np.eye(len(self.col2label2idx[col]))

        for data in dataset[col]:

            features = np.sum(self._encode(data, self.col2label2idx[col], code, unknown_token), axis=0).tolist()
            encoded_features.append(features)

        return encoded_features

    def encode_transform(self, dataset, col, all_cats):

        enc = self._build(col, all_cats)
        encoded_features = self.transform(dataset, col)

        return encoded_features

    def _build(self, col, all_cats):

        enc = DataEncoder(enable_padding=self.enable_padding, enable_unknown=self.enable_unknown)
        enc.add_data(all_cats)
        self.col2label2idx[col] = enc.data2index

        return enc

    def _encode(self, data, data2index, code, unknown_token):

        if isinstance(data, list):
            idx = [data2index.get(d, unknown_token) for d in data]
        else:
            idx = [data2index.get(data, unknown_token)]

        if len(idx) == 0:
            features = np.zeros((1, len(data2index)))
        else:
            features = code[idx]

        return features


class JollybuyGoodsHotDataPreprocesser():

    def __init__(self, opt, configs, all_cats_file='col2label.pickle', is_train=True, col2label2idx=None, logger=logger):

        self.opt = opt
        self.configs = configs
        self.all_cats_file = all_cats_file
        self.is_train = is_train
        self.logger = logger
        self.col2label2idx = col2label2idx if col2label2idx else {}

    def process(self, requisite_cols: list = None, dummy_encode: bool = False):
        ''' Method to process dataset step by step
        :param requisite_cols: list of columns to get from processed dataset
        :type requisite_cols: List[str]
        :param dummy_encode: whether to do dummy encoding to categorical features
        :type dummy_encode: bool
        :return: processed dataset
        :rtype: pd.DataFrame
        '''

        dataset = self.read_data()

        # convert data type
        dataset = self.convert_data_type(dataset)

        # normalize data
        dataset = self.minmax_norm(dataset)

        # normalize reward
        dataset = self.step_norm(dataset, getattr(self.configs, 'REWARD_NORM_STEPS', []))

        # process categorical features
        dataset = self.process_categorical(dataset, get_dummy=True)

        if requisite_cols and isinstance(requisite_cols, list):
            if dummy_encode:
                # if we apply dummy encoding, we need to expand final columns with all categorical columns
                # ex: cat1 = ['電玩遊戲', '文創', '寵物'] -> requisite_cols.extend(['UNKNOWN', '電玩遊戲', '文創', '寵物'])
                for col in self.configs.COLUMNS_TO_ENCODE:
                    requisite_cols.extend(list(self.col2label2idx[col].keys()))

            dataset = dataset[requisite_cols]

        return dataset

    def read_data(self, dataset=None):
        '''Method to read pickled dataset
        '''

        if self.is_train:
            dataset = read_pickle(self.opt.dataset, base_path=self.opt.dataroot)

        self.raw_dataset = dataset.copy()

        return dataset

    def convert_data_type(self, dataset: pd.DataFrame):
        '''Method to transform data type for columns specified in config "TYPE_CONVERT_MODE2COLS"
        :param dataset: input dataset
        :type dataset: pd.DataFrame
        :return: dataset
        :rtype: pd.DataFrame
        '''
        self.logger.info(f'---convert data type---')
        dataset = convert_type(dataset, self.configs.TYPE_CONVERT_MODE2COLS)

        return dataset

    def process_categorical(self, dataset: pd.DataFrame, prefix: str = 'click_',
                            suffix: str = '_list', get_dummy: bool = False):
        '''Method to encode categorical data into index

        :param dataset: input dataset
        :type dataset: pd.DataFrame
        :param prefix: original prefix to be removed
        :type prefix: string
        :param suffix: original suffix to be removed
        :type suffix: string
        :param get_dummy: whether to transform categorical feature into one-hot/multi-hot features
        :type get_dummy: bool
        :return: dataset
        :rtype: pd.DataFrame
        '''
        encoder = CategoricalEncoder(enable_unknown=True, enable_padding=False, col2label2idx=self.col2label2idx)

        if self.is_train:
            # load saved categorical data
            self.logger.info(f'---Load categorical feature mapping dictionary---')
            with open(os.path.join(self.opt.dataroot, self.all_cats_file), 'rb') as handle:
                all_cats = pickle.load(handle)

        self.logger.info(f'---process categorical data---')
        for col in self.configs.COLUMNS_TO_ENCODE:
            self.logger.info(f'process: {col}')

            c = col.replace(prefix, '').replace(suffix, '')

            if self.is_train:
                dataset[col] = encoder.encode_transform(dataset, col, all_cats[c])
            else:
                dataset[col] = encoder.transform(dataset, col)

            if get_dummy:
                data_to_encode = dataset[col].to_list()
                dummy_cols = list(encoder.col2label2idx[col].keys())

                tmp_dataset = pd.DataFrame(data=data_to_encode, columns=dummy_cols)
                dataset[dummy_cols] = tmp_dataset.values

        self.col2label2idx = encoder.col2label2idx

        return dataset

    def minmax_norm(self, dataset: pd.DataFrame):
        '''Method to do min-max normalization to current dataset on the columns specified in config "COLUMNS_TO_MINMAX" field

        :param dataset: input dataset
        :type dataset: pd.DataFrame
        :return: normalized dataset
        :rtype: pd.DataFrame
        '''
        self.logger.info(f'---process min-max normalization---')
        for col in self.configs.COLUMNS_TO_MINMAX:
            self.logger.info(f'process: {col}')

            min_score = dataset[col].min()
            max_score = dataset[col].max()
            dataset[col] = dataset[col].apply(lambda x: calculate_min_max(x, max_score, min_score))

        return dataset

    def step_norm(self, dataset: pd.DataFrame, steps: list, target_col: str = 'reward'):
        '''Method to do step normalization to current dataset on the "target_col"
        EX:
        steps = [1,2,3] ; target_col = reward 
        -> reward = 3 if reward == 3;
           reward = 2 if (reward < 3 and reward >=2);
           reward = 1 if (reward < 2 and reward >=1)

        :param dataset: input dataset
        :type dataset: pd.DataFrame
        :param steps: steps for normalization (ex: [1, 2, 3])
        :type steps: list
        :param target_col: target column to do step normalization
        :type target_col: string
        :return: normalized dataset
        :rtype: pd.DataFrame
        '''

        self.logger.info(f'---step norm reward col: {target_col}---')
        steps = sorted(steps, reverse=True)

        for i, step in enumerate(steps):

            if i != 0:
                dataset.loc[(dataset[target_col] < pre_step) & (dataset[target_col] >= step), target_col] = step
            else:
                dataset.loc[dataset[target_col] > step, target_col] = step

            pre_step = step

        return dataset
