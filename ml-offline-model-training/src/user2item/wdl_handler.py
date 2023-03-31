# -*- coding: utf-8 -*-
from src.user2item.deepctr_handler import DeepCTRHandler
from src.module.deepctr_torch.models.wdl import WDL
import torch


class WDLHandler(DeepCTRHandler):

    def __init__(self, opt, config, encoders, use_cuda=True):
        self.device = 'cuda:0' if(use_cuda and torch.cuda.is_available()) else 'cpu'
        self.opt = opt
        self.config = config
        self.encoders = encoders

    def code(cls):
        return 'WDL'

    def initialize(self):

        self.set_feature_columns()

        if self.opt.is_train:
            self.model = WDL(self.linear_feature_columns, self.dnn_feature_columns, task=self.opt.task, device=self.device)
            self.model.compile(self.opt.optimizer, self.opt.objective_function, metrics=self.opt.metrics)
        else:
            self.model = WDL(self.linear_feature_columns, self.dnn_feature_columns)

    def set_feature_columns(self):
        self.dnn_feature_columns = self._get_sparse_feature_columns() + self._get_dense_feature_columns() + self._get_semantics_feature_columns()
        self.linear_feature_columns = self._get_sparse_feature_columns() + self._get_dense_feature_columns()
        self.feature_columns = self.dnn_feature_columns + self.linear_feature_columns
