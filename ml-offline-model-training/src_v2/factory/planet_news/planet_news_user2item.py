import pandas as pd
import numpy as np
from typing import Union

from utils.logger import logger
from src_v2.configs.planet_news.planet_news_user2item import PlanetNewsUser2ItemDINConfig
from src_v2.preprocess.planet_news.planet_news_user2item import PlanetNewsUser2ItemDINPreprocessor
from src_v2.handler.din_handler import DINHandler
from src_v2.factory import CTRRankerFactory


class PlanetNewsUser2ItemDINFactory(CTRRankerFactory):

    def __init__(self, mode: str = 'train', col2label2idx: dict = None, model_path: str = '/ml-model', **kwargs):

        assert mode in self.MODE, 'only support train, validation and inference mode'
        self.mode = mode
        self.initialize(self.mode, col2label2idx=col2label2idx, model_path=model_path, **kwargs)

    def initialize(self, mode: str, col2label2idx: dict = None, **kwargs):

        self.config = PlanetNewsUser2ItemDINConfig

        if mode == 'train':
            self.preprocessor = PlanetNewsUser2ItemDINPreprocessor(configs=self.config,
                                                                   col2label2idx=col2label2idx,
                                                                   mode=mode,
                                                                   **kwargs)
            self.model_params = kwargs
        else:
            if mode == 'inference':
                self.config.REQUISITE_COLS.remove('y')

            self.preprocessor = PlanetNewsUser2ItemDINPreprocessor(configs=self.config,
                                                                   col2label2idx=col2label2idx,
                                                                   mode=mode)
            self.model_handler = DINHandler(col2label2idx=self.preprocessor.col2label2idx,
                                            config=self.config,
                                            mode=mode,
                                            use_cuda=False,
                                            **self.config.MODEL_PARAMS)
            self.model_handler.load_model(kwargs.get('model_path', '/ml-model/user2item/planet_news/din/model.latest.pickle'))

    def train(self, dataset: pd.DataFrame, y_true: Union[pd.Series, np.array, list], train_params: dict = None, **kwargs):

        dataset = self.preprocessor.process(dataset=dataset,
                                            requisite_cols=kwargs.get('requisite_cols', None),
                                            **kwargs)

        self.processed_dataset = dataset
        logger.info(f'Processed data columns: {dataset.columns} \nProcessed data: \n {dataset.iloc[len(dataset) // 3]}')

        self.model_handler = DINHandler(col2label2idx=self.preprocessor.col2label2idx,
                                        config=self.config,
                                        mode=self.mode,
                                        **self.model_params)
        self.model_handler.train(x_train=dataset, y_train=y_true, train_params=train_params)

    def predict(self, dataset, **kwargs):

        dataset = self.preprocessor.process(dataset, **kwargs)
        scores = self.model_handler.predict(dataset).reshape(-1)

        return scores
