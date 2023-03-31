# -*- coding: utf-8 -*-
from bdds_recommendation.src.item2item.base_handler import ModelHandler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from bdds_recommendation.utils import read_pickle, dump_pickle
from bdds_recommendation.utils.advanced_metrics import get_average_metrics
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class XGBHandler(ModelHandler):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, opt):
        self.opt = opt
        self.xgb = xgb

    @classmethod
    def code(cls):
        return 'xgboost'

    def train(self, X_train, y_train):
        dtrain = self.set_input(X_train, y_train)
        eval_list = [(dtrain, 'train')]
        self.clf = self.xgb.train(
            self.params,
            dtrain,
            self.opt.num_boost_round,
            eval_list
        )

    def cross_validation(self, X_train, y_train, k=5):
        """Kfold cross validation"""
        kf = KFold(n_splits=k, shuffle=True)

        metrics = {
            'precision': np.empty((kf.n_splits, 2)),
            'recall': np.empty((kf.n_splits, 2)),
            'f1': np.empty((kf.n_splits, 2)),
            'auc': np.empty((kf.n_splits, 1))
        }

        for i, index in enumerate(kf.split(X_train)):
            train_index, test_index = index

            kf_X_train = X_train.iloc[train_index]
            kf_y_train = y_train.iloc[train_index]
            kf_X_test = X_train.iloc[test_index]
            kf_y_test = y_train.iloc[test_index]

            dtrain = self.set_input(kf_X_train, kf_y_train)
            dtest = self.set_input(kf_X_test, kf_y_test)

            clf = self.xgb.train(
                self.params,
                dtrain,
                self.opt.num_boost_round
            )

            y_pred = clf.predict(dtest)
            all_metrics = self.metrics(kf_y_test, y_pred, option=['auc'])

            for k, v in all_metrics.items():
                metrics[k][i] = v

        return {k: v.mean(axis=0) for (k, v) in metrics.items()}

    def get_ndcg_score(self, x, y, source_raw_data, valid_size=0.2, k=[5], id_key='uuid'):

        if valid_size == 0:
            x_train, y_train = x, y
            x_test, y_test = x, y
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=valid_size)

        self.train(x_train, y_train)
        y_pred = self.predict(x_test)

        # prepare data for calculating ndcg score
        target_idx = y_test.index
        df_ndcg = pd.DataFrame(source_raw_data.iloc[target_idx][id_key].values, columns=[id_key], index=target_idx)
        df_ndcg['y_true'] = y_test.values
        df_ndcg['y_pred'] = y_pred

        # get score
        ndcg_scores = {}

        if isinstance(k, int):
            ndcg_scores[f'NDCG@{k}'] = get_average_metrics(df_ndcg, k=k, metric_type='ndcg', groupby_key=id_key)
        elif isinstance(k, list):
            for k_value in k:
                ndcg_scores[f'NDCG@{k_value}'] = get_average_metrics(df_ndcg, k=k_value, metric_type='ndcg', groupby_key=id_key)

        return ndcg_scores

    def predict(self, X_test):
        dtest = self.xgb.DMatrix(X_test)

        return self.clf.predict(dtest)

    def feature_importances(self, importance_type='gain', visualize=False):
        score = self.clf.get_score(importance_type=importance_type)
        dict(sorted(score.items(), key=lambda x: x[1], reverse=True))

        if visualize:
            self._visualize_feature_importances(score)

        return score

    def set_input(self, X, y=None):
        return self.xgb.DMatrix(X, label=y)

    def load_model(self, base_path='./', model_name='model', postfix_tag='latest'):
        model_name = f'{model_name}.{postfix_tag}.pickle'
        self.clf = read_pickle(model_name, base_path)

    def save_model(self, base_path='./', model_name='model', postfix_tag='latest'):
        dump_pickle(f'{model_name}.{postfix_tag}', self.clf, base_path)

    def _visualize_feature_importances(self, importances):

        keys = list(importances.keys())
        values = list(importances.values())

        plt.style.use('ggplot')
        plt.barh(keys, values)
        plt.title('Feature Importances')
        plt.ylabel('Features')
        plt.xlabel('Gain')
        plt.xticks(range(0, round(max(values)) + 10, 10))

        for i, v in enumerate(values):
            plt.text(v + 3, i, str(round(v, ndigits=2)), color='blue', fontweight='bold')
        plt.show()
