import pandas as pd
import numpy as np
from typing import Union

from bdds_recommendation.utils import read_pickle, dump_pickle
from bdds_recommendation.src_v2.handler import BaseModelHandler
from bdds_recommendation.src_v2.configs import Config
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline


class SGDHandler(BaseModelHandler):

    def __init__(self, config: Config, mode: str = 'train', **kwargs):

        super(SGDHandler, self).__init__(mode)

        self.config = config
        self.mode = mode
        self.initialize(**kwargs)

    def code(cls):
        return 'SGDClassifier'

    def initialize(self, **kwargs):
        self.sgd = SGDClassifier

    def _setup_pipeline(self, train_params: dict = None):

        train_params = train_params if isinstance(train_params, dict) else {}
        pre_pipelines = getattr(self.config, 'PRE_PIPELINE', None)
        post_pipelines = getattr(self.config, 'POST_PIPELINE', None)
        pipeline = []

        if pre_pipelines:
            pipeline += pre_pipelines

        pipeline += [self.sgd(**train_params)]

        if post_pipelines:
            pipeline += post_pipelines

        pipeline = make_pipeline(*pipeline)

        return pipeline

    def set_input(self, df: pd.DataFrame, **kwargs):

        remove_columns_list = getattr(self.config, 'REMOVE_COLS', ['uuid', 'click_uuid', 'y'])

        for col in remove_columns_list:
            df = df.drop([col], axis=1) if col in df.columns else df
        return df

    def train(self, x_train: pd.DataFrame, y_train: Union[pd.Series, np.array, list], train_params: dict = None, **kwargs):
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        print(x_train.head())
        x_train = x_train.drop('y', axis=1)
        # self.model = self._setup_pipeline(train_params=train_params)
        # self.model.fit(self.set_input(x_train), y_train)
        self.clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=3, tol=1e-3))
        self.clf.fit(x_train.to_numpy(), y_train.to_numpy())

    def predict(self, df: pd.DataFrame):
        df = df.drop('y', axis=1)
        y_pred_prob = self.clf.predict(df.to_numpy())
        # df = self.set_input(df)
        # y_pred_prob = self.model.predict_proba(df)[:, 1]

        return y_pred_prob

    def load_model(self, base_path: str, model_path: str):

        self.model = read_pickle(file_name=model_path, base_path=base_path)

    def save_model(self, base_path: str, model_path: str):

        dump_pickle(file_name=model_path, base_path=base_path, data=self.model)
