# -*- coding: utf-8 -*-
from .base_handler import ModelHandler
from sklearn.model_selection import KFold
from utils import read_pickle, dump_pickle
from datetime import date
import xgboost as xgb
import numpy as np


class XGBHandler(ModelHandler):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, opt, col2lbe=None):
        self.opt = opt
        self.col2lbe = col2lbe
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
            p, r, f, auc = self.metrics(kf_y_test, y_pred, option=['auc'])

            metrics['precision'][i] = p
            metrics['recall'][i] = r
            metrics['f1'][i] = f
            metrics['auc'][i] = auc

        return (metrics['precision'].mean(axis=0),
                metrics['recall'].mean(axis=0),
                metrics['f1'].mean(axis=0),
                metrics['auc'].mean(axis=0))

    def predict(self, X_test):
        dtest = self.xgb.DMatrix(X_test)
        return self.clf.predict(dtest)

    def feature_importances(self, importance_type='gain'):
        score = self.clf.get_score(importance_type=importance_type)
        dict(sorted(score.items(), key=lambda x: x[1], reverse=True))
        return score

    def set_input(self, X, y=None):
        return self.xgb.DMatrix(X, label=y)

    def load_model(self, base_path='./', model_name='model', postfix_tag='latest'):
        model_name = f'{model_name}.{postfix_tag}.pickle'
        self.clf = read_pickle(model_name, base_path)

    def save_model(self, base_path='./', model_name='model', postfix_tag='latest'):
        if postfix_tag:
            dump_pickle(f'{model_name}.{postfix_tag}', self.clf, base_path)
