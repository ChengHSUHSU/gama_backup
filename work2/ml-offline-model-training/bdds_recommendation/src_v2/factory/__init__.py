import pandas as pd
from abc import ABCMeta, abstractclassmethod
from bdds_recommendation.src_v2.configs import GeneralConfigs
from bdds_recommendation.src_v2.preprocess.base import BaseUser2ItemPreprocessor, BaseItem2ItemPreprocessor
from bdds_recommendation.src_v2.handler import BaseModelHandler


class CTRRankerFactory(metaclass=ABCMeta):

    MODE = ['train', 'validation', 'inference']

    def __init__(self, mode: str = 'train', col2label2idx: dict = None, model_path: str = '/ml-model', **kwargs):

        assert mode in self.MODE, 'only support train, validation and inference mode'
        self.mode = mode
        self._preprocessor = None
        self._config = None
        self.initialize(self.mode, col2label2idx=col2label2idx, model_path=model_path, **kwargs)

    @abstractclassmethod
    def initialize(self, mode: str, col2label2idx: dict = None, **kwargs):

        self.config = GeneralConfigs

        if self.config.SERVICE_TYPE == 'user2item':
            obj_preprocessor = BaseUser2ItemPreprocessor
        elif self.config.SERVICE_TYPE == 'item2item':
            obj_preprocessor = BaseItem2ItemPreprocessor

        if mode == 'train':
            self.preprocessor = obj_preprocessor(configs=self.config, col2label2idx=col2label2idx, mode=mode)
            self.model_handler = BaseModelHandler(mode=mode, **kwargs)
        else:
            self.preprocessor = obj_preprocessor(configs=self.config, col2label2idx=col2label2idx, mode=mode)
            self.model_handler = BaseModelHandler(mode=mode, **kwargs)
            self.model_handler.load_model(kwargs.get('model_path', './model.pth'))

    @abstractclassmethod
    def train(self, dataset: pd.DataFrame, **kwargs):

        dataset = self.preprocessor.process(dataset=dataset,
                                            requisite_cols=kwargs.get('requisite_cols', []),
                                            **kwargs)
        self.model_handler.train(x_train=dataset, y_train=dataset['y'])

    @abstractclassmethod
    def predict(self, dataset, **kwargs):

        dataset = self.preprocessor.process(dataset, **kwargs)
        scores = self.model_handler.predict(dataset)

        return scores

    @property
    def preprocessor(self):

        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, new_preprocessor):

        self._preprocessor = new_preprocessor

    @property
    def config(self):

        return self._config

    @config.setter
    def config(self, new_config):

        self._config = new_config
