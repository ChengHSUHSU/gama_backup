# -*- coding: utf-8 -*-
from typing import List
from abc import ABCMeta, abstractmethod


class BanditHotModelHandler(metaclass=ABCMeta):
    """Model Handler for bandit hot model"""

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass
