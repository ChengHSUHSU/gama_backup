import os
import pandas as pd
import numpy as np
from typing import Union

from utils.logger import logger
from src_v2.configs.planet_news.planet_news_item2item import PlanetNewsItem2ItemSGDConfig
from src_v2.preprocess.planet_news.planet_news_item2item import PlanetNewsItem2ItemSGDPreprocessor
from src_v2.handler.sgd_handler import SGDHandler
from src_v2.factory import CTRRankerFactory


class PlanetNewsItem2ItemSGDFactory(CTRRankerFactory):

    def __init__(self, mode: str = 'train', col2label2idx: dict = None, model_path: str = '/ml-model/model.latest.pickle', **kwargs):

        assert mode in self.MODE, 'only support train, validation and inference mode'
        self.mode = mode
        self.initialize(self.mode, col2label2idx=col2label2idx, model_path=model_path, **kwargs)

    def initialize(self, mode: str, col2label2idx: dict = None, **kwargs):

        self.config = PlanetNewsItem2ItemSGDConfig

        if mode == 'train':

            self.preprocessor = PlanetNewsItem2ItemSGDPreprocessor(configs=self.config, col2label2idx=col2label2idx, mode=mode, **kwargs)

        else:
            if mode == 'inference':
                self.config.REQUISITE_COLS = [val for val in self.config.REQUISITE_COLS if val not in self.config.REMOVE_COLS]

            self.preprocessor = PlanetNewsItem2ItemSGDPreprocessor(configs=self.config, col2label2idx=col2label2idx, mode=mode, **kwargs)

            base_path, model_path = os.path.split(
                kwargs.get('model_path', '/ml-model/model.latest.pickle'))

            self.model_handler = SGDHandler(config=self.config, mode=mode)
            self.model_handler.load_model(base_path=base_path, model_path=model_path)

    def train(self, dataset: pd.DataFrame, y_true: Union[pd.Series, np.array, list], train_params: dict = None, **kwargs):

        dataset = self.preprocessor.process(dataset=dataset,
                                            chain_configs=kwargs.get('chain_configs', None),
                                            requisite_cols=kwargs.get('requisite_cols', None),
                                            **kwargs)

        self.processed_dataset = dataset
        logger.info(f'Processed data columns: {dataset.columns} \nProcessed data: \n {dataset.iloc[len(dataset) // 3]}')

        self.model_handler = SGDHandler(config=self.config, mode=self.mode)
        self.model_handler.train(x_train=dataset, y_train=y_true, train_params=train_params)

    def predict(self, dataset, **kwargs):

        dataset = self.preprocessor.process(dataset=dataset,
                                            chain_configs=kwargs.get('chain_configs', None),
                                            requisite_cols=kwargs.get('requisite_cols', None),
                                            **kwargs)

        scores = self.model_handler.predict(dataset)

        return scores
