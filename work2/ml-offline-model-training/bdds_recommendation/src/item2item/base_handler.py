# -*- coding: utf-8 -*-
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc
from abc import ABCMeta, abstractmethod


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

        fpr, tpr, threshold = roc_curve(y_true, y_pred)

        result = {'recall': r, 'precision': p, 'f1': f1}

        if 'roc' in option:
            result['fpr'] = fpr
            result['tpr'] = tpr
        if 'auc' in option:
            result['auc'] = auc(fpr, tpr)
        return result

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass
