# -*- coding: utf-8 -*-
from bdds_recommendation.src.user2item.deepctr_handler import DeepCTRHandler
from bdds_recommendation.src.module.deepctr_torch.models.dcn import DCN
import torch


class DCNHandler(DeepCTRHandler):

    def __init__(self, opt, config, encoders, col2label2idx=None, use_cuda=True):
        super().__init__(opt, config, encoders, col2label2idx, use_cuda)
        self.device = 'cuda:0' if(use_cuda and torch.cuda.is_available()) else 'cpu'
        self.opt = opt
        self.config = config
        self.encoders = encoders
        self.col2label2id = col2label2idx

    def code(cls):
        return 'DCN'

    def initialize(self):

        self.set_feature_columns()

        if self.opt.is_train:
            self.model = DCN(self.linear_feature_columns, self.dnn_feature_columns, task=self.opt.task, device=self.device)
            self.model.compile(self.opt.optimizer, self.opt.objective_function, metrics=self.opt.metrics)
        else:
            self.model = DCN(self.linear_feature_columns, self.dnn_feature_columns)

    def set_feature_columns(self):
        self.dnn_feature_columns = self._get_sparse_feature_columns() + self._get_dense_feature_columns() + self._get_semantics_feature_columns()
        self.linear_feature_columns = self._get_sparse_feature_columns() + self._get_dense_feature_columns()
        self.feature_columns = self.dnn_feature_columns + self.linear_feature_columns
