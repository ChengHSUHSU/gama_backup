# -*- coding: utf-8 -*-
from sklearn.metrics import *
from abc import *
from utils.advanced_metrics import get_average_metrics
import pandas as pd


class ModelHandler(metaclass=ABCMeta):
    """Model Handler for machine learning module"""
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def set_params(self, **params):
        self.params = params

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def set_input(self):
        pass

    @abstractmethod
    def cross_validation(self):
        pass

    def metrics(self, y_true, y_pred, option=[]):
        y_binary = [round(value) for value in y_pred]

        r = recall_score(y_true, y_binary, average=None)
        p = precision_score(y_true, y_binary, average=None)
        f1 = f1_score(y_true, y_binary, average=None)

        fpr, tpr, _ = roc_curve(y_true, y_pred)

        result = (p, r, f1,)
        if 'roc' in option:
            result = result + (fpr, tpr,)
        if 'auc' in option:
            result = result + (auc(fpr, tpr),)
        return result

    def metric_ndcg(self, df, y_pred, n=5, groupby_key='userid'):
        """
        Arguments:

        df (pd.dataframe): raw dataset, which must include ['openid', 'y']
        y_pred (nd.array): model predict result
        n (int): top-n ranking of ndcg
        groupby_key (str): ndcg primary key
        """

        df.rename(columns={'y': 'y_true'}, inplace=True)
        df['y_pred'] = y_pred
        ndcg_scores = get_average_metrics(df, k=n, metric_type='ndcg', groupby_key=groupby_key)

        return ndcg_scores

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass
