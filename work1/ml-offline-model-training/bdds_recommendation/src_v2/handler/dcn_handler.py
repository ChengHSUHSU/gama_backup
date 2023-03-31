import torch
import pandas as pd
import numpy as np
from typing import Union

from bdds_recommendation.src_v2.handler import BaseModelHandler
from bdds_recommendation.src_v2.configs import Config
from bdds_recommendation.src.module.deepctr_torch.utils import get_sparse_feature_columns, get_dense_feature_columns, \
    get_behavior_feature_columns
from bdds_recommendation.src.module.deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names
from bdds_recommendation.src.module.deepctr_torch.models.dcn import DCN


class DCNHandler(BaseModelHandler):

    def __init__(self, col2label2idx: dict, config: Config, mode: str = 'train', use_cuda: bool = False, **kwargs):

        super(DCNHandler, self).__init__(mode)

        self.device = 'cuda:0' if (use_cuda and torch.cuda.is_available()) else 'cpu'
        self.config = config
        self.col2label2idx = col2label2idx
        self.mode = mode
        # self.linear_feature_columns = self._get_sparse_feature_columns() + self._get_dense_feature_columns()
        # #self.dnn_feature_columns = self._get_sparse_feature_columns() + self._get_dense_feature_columns() + self._get_semantics_feature_columns()
        # self.dnn_feature_columns = self.get_feature_columns()
        # self.feature_columns = self.dnn_feature_columns + self.linear_feature_columns




        kwargs['device'] = self.device
        self.initialize(**kwargs)

    def code(cls):
        return 'DCN'

    def initialize(self, **kwargs):

        self.set_feature_columns()

        self.model = DCN(dnn_feature_columns=self.dnn_feature_columns,
                         linear_feature_columns=self.linear_feature_columns,
                         **kwargs)
 
        if self.mode == 'train':
            self.model.compile(optimizer=kwargs.get('optimizer', 'adagrad'),
                               loss=kwargs.get('objective_function', 'binary_crossentropy'),
                               metrics=kwargs.get('metrics', ['binary_crossentropy']))

    def train(self, x_train: pd.DataFrame, y_train: Union[pd.Series, np.array, list], train_params: dict):

        x_train = self.set_input(x_train)
        
        #print('x_train : ', x_train.keys())
 
        y_train = np.array(y_train)

        result = self.model.fit(x_train, y_train, **train_params)
        return result

    def predict(self, df: pd.DataFrame):
        df = self.set_input(df)

        return self.model.predict(df)

    def load_model(self, model_path: str):

        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

    def save_model(self, output_path: str):

        torch.save(self.model.state_dict(), output_path)

    def set_input(self, df: pd.DataFrame):
        feature_dict = {}
        print('11 df : ', df.head(1))

        for column_name in df:
            feature_dict[column_name] = np.array(df[column_name].to_list()).astype(np.float)

        input_features = {name: feature_dict[name] for name in get_feature_names(self.feature_columns)}
        return  input_features

    def get_feature_columns(self) -> list:
        """Public method to get feature columns

        Returns:
            list: list of feature columns
        """
        sparse_feat_cols = get_sparse_feature_columns(sparse_feature_cols=self.config.ONE_HOT_FEATURE,
                                                      feature_embed_size_map=self.config.FEATURE_EMBEDDING_SIZE,
                                                      col2label2idx=self.col2label2idx)

        dense_feat_cols = get_dense_feature_columns(dense_feature_cols=self.config.DENSE_FEATURE,
                                                    dense_embed_size_map=self.config.DENSE_FEATURE_SIZE)

        behavior_feat_cols = get_behavior_feature_columns(behavior_feature_cols=self.config.BEHAVIOR_FEATURE, behavior_feature_len_cols=self.config.BEHAVIOR_FEATURE_SEQ_LENGTH,
                                                          feature_embed_size_map=self.config.FEATURE_EMBEDDING_SIZE, col2label2idx=self.col2label2idx,
                                                          behavior_seq_size=self.config.BEHAVIOR_SEQUENCE_SIZE)

        self.feature_columns = sparse_feat_cols + dense_feat_cols + behavior_feat_cols

        return self.feature_columns

    def _get_sparse_feature_columns(self):
        """privete method for getting sparse feature"""

        sparse_feature_columns = []

        for feature_column_name in self.config.ONE_HOT_FEATURE:

            # TODO: Remove self.encoders[feature_column_name].num_of_data
            # In order to be consistent with serving code, we'll need to use col2label2idx instead of encoders (DataEncoder)
            # Currently, only Jollybuy user2item is consistent with serving code
            feature_length = len(self.col2label2idx[feature_column_name]) if self.col2label2idx else self.encoders[feature_column_name].num_of_data

            embedding_size = self.config.FEATURE_EMBEDDING_SIZE[feature_column_name]
            sparse_feature_columns.append(
                SparseFeat(
                    feature_column_name,
                    feature_length,
                    embedding_dim=embedding_size
                ))
        return sparse_feature_columns

    def _get_dense_feature_columns(self):
        """privete method for getting dense feature"""

        dence_feature_columns = []

        for feature_column_name in self.config.DENSE_FEATURE:
            dence_feature_columns += [DenseFeat(feature_column_name, 1)]

        return dence_feature_columns

    def _get_semantics_feature_columns(self):
        """privete method for getting semantics feature"""

        semantics_feature_columns = []

        for feature_column_name in self.config.SEMANTICS_INPUT:
            semantics_feature_columns += [DenseFeat(feature_column_name, self.config.SEMANTICS_EMBEDDING_SIZE)]

        return semantics_feature_columns

    def set_feature_columns(self):
        self.dnn_feature_columns = self._get_sparse_feature_columns() + self._get_dense_feature_columns() + self._get_semantics_feature_columns()
        self.linear_feature_columns = self._get_sparse_feature_columns() + self._get_dense_feature_columns()
        self.feature_columns = self.dnn_feature_columns + self.linear_feature_columns

