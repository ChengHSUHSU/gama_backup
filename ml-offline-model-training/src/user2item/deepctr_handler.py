# -*- coding: utf-8 -*-
from src.user2item.base_handler import ModelHandler
from src.module.deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names
from src.preprocess.utils.encoder import DataEncoder
from sklearn.model_selection import KFold
from typing import Dict
import torch
import numpy as np


class DeepCTRHandler(ModelHandler):

    def __init__(self, opt: dict, config: dict, encoders: Dict[str, DataEncoder],
                 col2label2idx: dict = None, use_cuda: bool = True):

        self.device = 'cuda:0' if(use_cuda and torch.cuda.is_available()) else 'cpu'
        self.opt = opt
        self.config = config

        # TODO: Remove self.encoders
        # In order to be consistent with serving code, we'll need to use col2label2idx instead of encoders (DataEncoder)
        # Currently, only Jollybuy user2item is consistent with serving code
        self.encoders = encoders
        self.col2label2idx = col2label2idx

    def code(cls):
        return 'DeepCTR'

    def _get_sparse_feature_columns(self):
        """privete method for getting sparse feature"""

        sparse_feature_columns = []

        for feature_column_name in self.config.ONE_HOT_FEATURE:

            # TODO: Remove self.encoders[feature_column_name].num_of_data
            # In order to be consistent with serving code, we'll need to use col2label2idx instead of encoders (DataEncoder)
            # Currently, only Jollybuy user2item is consistent with serving code
            feature_length = len(self.col2label2idx[feature_column_name]) if self.col2label2idx else self.encoders[feature_column_name].num_of_data

            embedding_size = self.config.FEATURE_EMBEDDING_SIZE[feature_column_name]
            sparse_feature_columns.append(
                SparseFeat(
                    feature_column_name,
                    feature_length,
                    embedding_dim=embedding_size
                ))
        return sparse_feature_columns

    def _get_dense_feature_columns(self):
        """privete method for getting dense feature"""

        dence_feature_columns = []

        for feature_column_name in self.config.DENSE_FEATURE:
            dence_feature_columns += [DenseFeat(feature_column_name, 1)]

        return dence_feature_columns

    def _get_semantics_feature_columns(self):
        """privete method for getting semantics feature"""

        semantics_feature_columns = []

        for feature_column_name in self.config.SEMANTICS_INPUT:
            semantics_feature_columns += [DenseFeat(feature_column_name, self.config.SEMANTICS_EMBEDDING_SIZE)]

        return semantics_feature_columns

    def set_input(self, df, column_need_padding=[]):

        feature_dict = {}

        for column_name in df:
            feature_dict[column_name] = np.array(df[column_name].to_list())

        input_features = {name: feature_dict[name] for name in get_feature_names(self.feature_columns)}
        return input_features

    def train(self, X_train, y_train):
        self.initialize()
        X_train = self.set_input(X_train)
        y_train = np.array(y_train)
        result = self.model.fit(X_train, y_train, **self.params)
        return result

    def predict(self, X_test):
        X_test = self.set_input(X_test)
        return self.model.predict(X_test)

    def cross_validation(self, X_train, y_train, k=5, random_state=None):
        """Kfold cross validation"""

        NDCG_GROUPBY_KEY = self.config.NDCG_GROUPBY_KEY     # NDCG metrics is requirement
        NDCG_AT_K = self.opt.ndcg
        NDCG_AT_K_LIST = NDCG_AT_K.split(',') if isinstance(NDCG_AT_K, str) else NDCG_AT_K

        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

        metrics = {
            'precision': np.empty((kf.n_splits, 2)),
            'recall': np.empty((kf.n_splits, 2)),
            'f1': np.empty((kf.n_splits, 2)),
            'auc': np.empty((kf.n_splits, 1)),
            'roc_curve': [],
            'ndcg': {}
        }

        for i, index in enumerate(kf.split(X_train)):
            train_index, test_index = index

            kf_X_train = X_train.iloc[train_index]
            kf_y_train = y_train.iloc[train_index]
            kf_X_test = X_train.iloc[test_index]
            kf_y_test = y_train.iloc[test_index]

            if self.opt.is_train:
                self.initialize()
                kf_X_train = self.set_input(kf_X_train)
                kf_y_train = np.array(kf_y_train)
                result = self.model.fit(kf_X_train, kf_y_train, **self.params)

            kf_X_test = self.set_input(kf_X_test)
            kf_y_test = np.array(kf_y_test)

            y_pred = self.model.predict(kf_X_test).reshape(-1)
            p, r, f, fpr, tpr, auc = self.metrics(kf_y_test, y_pred, option=['roc', 'auc'])

            metrics['precision'][i] = p
            metrics['recall'][i] = r
            metrics['f1'][i] = f
            metrics['auc'][i] = auc
            metrics['roc_curve'].append({'fpr': fpr, 'tpr': tpr})

            if NDCG_GROUPBY_KEY:
                for ndcgk in NDCG_AT_K_LIST:
                    metrics['ndcg'].setdefault(f'ndcg{ndcgk}', [])
                    ndcg_score = self.metric_ndcg(X_train.iloc[test_index], y_pred,
                                                  int(ndcgk), groupby_key=NDCG_GROUPBY_KEY)
                    metrics['ndcg'][f'ndcg{ndcgk}'].append(ndcg_score)

                    if i == (k-1):
                        metrics['ndcg'][f'ndcg{ndcgk}'] = np.array(metrics['ndcg'][f'ndcg{ndcgk}']).mean(axis=0)

        result = (
            metrics['precision'].mean(axis=0),
            metrics['recall'].mean(axis=0),
            metrics['f1'].mean(axis=0),
            metrics['auc'].mean(axis=0),
            metrics['roc_curve'],
            metrics['ndcg'])

        return result

    def save_model(self, model_path, base_path='./'):
        torch.save(self.model.state_dict(), f'{base_path}/{model_path}')

    def load_model(self, model_path, base_path='./'):
        self.initialize()
        self.model.load_state_dict(torch.load(f'{base_path}/{model_path}'))
