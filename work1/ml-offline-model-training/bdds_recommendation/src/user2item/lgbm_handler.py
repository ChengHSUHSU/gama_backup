# -*- coding: utf-8 -*-
from bdds_recommendation.src.user2item.base_handler import ModelHandler
from bdds_recommendation.utils import read_pickle, dump_pickle

from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
import numpy as np

class LGBMHandler(ModelHandler):

    def __init__(self, opt, config):
        self.opt = opt
        self.config = config
        self.lgbm = LGBMClassifier(n_estimators=self.opt.n_estimators,
                                   learning_rate=self.opt.learning_rate,
                                   objective=self.opt.objective)

    def code(cls):
        return 'LGBM'

    def set_input(self, dataset):
        X = dataset[self.config.REQUISITE_COLS].drop(self.config.ID_COLS + ['y'])
        Y = dataset['y']
        return X, Y

    def train(self, X_train, Y_train):
        self.lgbm.fit(X_train, Y_train)

    def predict(self, X_pred):
        return self.lgbm.predict(X_pred)

    def cross_validation(self, X_all, y_train, k=5):
        """Kfold cross validation"""

        NDCG_GROUPBY_KEY = self.config.NDCG_GROUPBY_KEY     # NDCG metrics is requirement
        NDCG_AT_K = self.opt.ndcg
        NDCG_AT_K_LIST = NDCG_AT_K.split(',') if isinstance(NDCG_AT_K, str) else NDCG_AT_K

        kf = KFold(n_splits=k, shuffle=True)
        X_train = X_all.drop(['userid', 'uuid', 'y'], axis=1)

        metrics = {
            'precision': np.empty((kf.n_splits, 2)),
            'recall': np.empty((kf.n_splits, 2)),
            'f1': np.empty((kf.n_splits, 2)),
            'auc': np.empty((kf.n_splits, 1)),
            'ndcg': {}
        }

        for i, index in enumerate(kf.split(X_train)):
            train_index, test_index = index

            kf_X_train = X_train.iloc[train_index]
            kf_y_train = y_train.iloc[train_index]
            kf_X_test = X_train.iloc[test_index]
            kf_y_test = y_train.iloc[test_index]

            self.lgbm.fit(kf_X_train, kf_y_train)

            y_pred = self.lgbm.predict(kf_X_test)
            p, r, f, auc = self.metrics(kf_y_test, y_pred, option=['auc'])

            metrics['precision'][i] = p
            metrics['recall'][i] = r
            metrics['f1'][i] = f
            metrics['auc'][i] = auc

            if NDCG_GROUPBY_KEY:
                for ndcgk in NDCG_AT_K_LIST:
                    metrics['ndcg'].setdefault(f'ndcg{ndcgk}', [])
                    ndcg_score = self.metric_ndcg(X_all.iloc[test_index], y_pred, int(ndcgk), groupby_key=NDCG_GROUPBY_KEY)
                    metrics['ndcg'][f'ndcg{ndcgk}'].append(ndcg_score)
                    if i == (k-1):
                        metrics['ndcg'][f'ndcg{ndcgk}'] = np.array(metrics['ndcg'][f'ndcg{ndcgk}']).mean(axis=0)
        result = (
            metrics['precision'].mean(axis=0),
            metrics['recall'].mean(axis=0),
            metrics['f1'].mean(axis=0),
            metrics['auc'].mean(axis=0),
            metrics['ndcg'])
        return result

    def save_model(self, base_path='./', model_name='model', postfix_tag='latest'):
        dump_pickle(f'{model_name}.{postfix_tag}', self.lgbm, base_path)

    def load_model(self, base_path='./', model_name='model', postfix_tag='latest'):
        model_name = f'{model_name}.{postfix_tag}.pickle'
        self.lgbm = read_pickle(model_name, base_path)
