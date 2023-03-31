# -*- coding: utf-8 -*-
from .base_handler import ModelHandler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from utils import read_pickle, dump_pickle
from utils.advanced_metrics import get_average_metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SGDClassifierHandler(ModelHandler):

    def __init__(self, opt, configs):
        self.opt = opt
        self.configs = configs
        self.sgd = SGDClassifier

    @classmethod
    def code(cls):
        return 'SGDClassifier'

    def _setup_pipeline(self):

        pre_pipelines = getattr(self.configs, 'PRE_PIPELINE', None)
        post_pipelines = getattr(self.configs, 'POST_PIPELINE', None)
        pipeline = []

        if pre_pipelines:
            pipeline += pre_pipelines

        pipeline += [self.sgd(**self.params)]

        if post_pipelines:
            pipeline += post_pipelines

        pipeline = make_pipeline(*pipeline)

        return pipeline

    def train(self, X_train, y_train):

        self.clf = self._setup_pipeline()
        self.clf.fit(X_train, y_train)

    def cross_validation(self, X_train, y_train, k=5, random_state=None):
        """Kfold cross validation"""
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

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

            model = self._setup_pipeline()
            model = model.fit(kf_X_train, kf_y_train)
            y_pred = model.predict(kf_X_test)

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
            ndcg_scores[f'ndcg{k}'] = get_average_metrics(df_ndcg, k=k, metric_type='ndcg', groupby_key=id_key)
        elif isinstance(k, list):
            for k_value in k:
                ndcg_scores[f'ndcg{k_value}'] = get_average_metrics(df_ndcg, k=k_value, metric_type='ndcg', groupby_key=id_key)

        return ndcg_scores

    def predict(self, X_test):
        y_pred_prob = self.clf.predict_proba(X_test)[:, 1]

        return y_pred_prob

    def feature_importances(self, features, visualize=False):
        weights = self.clf['sgdclassifier'].coef_[0]

        score = {k: v for (k, v) in zip(features, weights)}

        if visualize:
            self._visualize_feature_importances(score)

        return score

    def set_input(self, x, y):

        if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
            return x.values, y.values

        return X, y

    def load_model(self, model_path=None, base_path='./'):
        self.clf = read_pickle(model_path, base_path)

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