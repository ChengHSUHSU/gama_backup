# -*- coding: utf-8 -*-
from bdds_recommendation.src.user2item.deepctr_handler import DeepCTRHandler
from bdds_recommendation.src.module.deepctr_torch.inputs import (SparseFeat, VarLenSparseFeat)
from bdds_recommendation.src.module.deepctr_torch.models.din import DIN
from bdds_recommendation.src.preprocess.utils.encoder import DataEncoder
from typing import Dict
import torch


class DINHandler(DeepCTRHandler):
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
        return "DIN"

    def initialize(self):

        self.model = DIN(self.get_feature_columns(),
                         self.config.BEHAVIOR_FEATURE,
                         att_activation=self.opt.att_activation,
                         att_weight_normalization=self.opt.att_weight_normalization,
                         dnn_activation=self.opt.dnn_activation,
                         dnn_dropout=self.opt.dnn_dropout,
                         dnn_use_bn=self.opt.dnn_use_bn,
                         l2_reg_dnn=self.opt.l2_reg_dnn,
                         l2_reg_embedding=self.opt.l2_reg_embedding,
                         init_std=self.opt.init_std,
                         seed=self.opt.seed,
                         task=self.opt.task,
                         device=self.device)

        if self.opt.is_train:
            self.model.compile(self.opt.optimizer, self.opt.objective_function, metrics=self.opt.metrics)

    def _get_behavior_feature_columns(self):
        """privete method for getting behavior feature"""

        behavior_feature_cols = getattr(self.config, 'BEHAVIOR_FEATURE', [])
        behavior_seq_length_cols = getattr(self.config, 'BEHAVIOR_FEATURE_SEQ_LENGTH', [])

        behavior_feature_columns = []

        for (feature_column_name, feature_seq_len_name) in zip(behavior_feature_cols, behavior_seq_length_cols):
            embedding_size = self.config.FEATURE_EMBEDDING_SIZE[feature_column_name]

            # TODO: Remove self.encoders
            # In order to be consistent with serving code, we'll need to use col2label2idx instead of encoders (DataEncoder)
            # Currently, only Jollybuy user2item is consistent with serving code
            feature_length = len(self.col2label2idx[feature_column_name]) if self.col2label2idx else self.encoders[feature_column_name].num_of_data

            behavior_feature_columns += [
                VarLenSparseFeat(
                    SparseFeat(
                        f'hist_{feature_column_name}',
                        feature_length + 1,  # TODO: remove +1 along with serving code
                        embedding_dim=embedding_size),
                    self.config.BEHAVIOR_SEQUENCE_SIZE,
                    length_name=feature_seq_len_name
                )
            ]

        return behavior_feature_columns

    def get_feature_columns(self):
        """public method for getting behavior feature"""

        self.feature_columns = self._get_sparse_feature_columns() \
            + self._get_dense_feature_columns() \
            + self._get_semantics_feature_columns() \
            + self._get_behavior_feature_columns()

        return self.feature_columns
