import pandas as pd

from utils.logger import logger
from src_v2.configs.planet_news.planet_news_hot import PlanetNewsHotLinUCBConfig
from src_v2.preprocess.planet_news.planet_news_hot import PlanetNewsHotLinUCBPreprocessor
from src_v2.handler.linucb_handler import LinUCBHandler
from src_v2.factory import CTRRankerFactory


class PlanetNewsHotLinUCBFactory(CTRRankerFactory):

    def __init__(self, mode: str='train', **kwargs):
        assert mode in self.MODE, 'only support train, validation and inference mode'
        self.initialize(mode, **kwargs)
        self.mode = mode

    def initialize(self, mode: str='train', **kwargs):
        self.config = PlanetNewsHotLinUCBConfig

        if mode == 'inference':
            self.config.REQUISITE_COLS.remove('reward')

        self.preprocessor = PlanetNewsHotLinUCBPreprocessor(configs=self.config,
                                                            mode=mode,
                                                            **kwargs)

    def train(self, data: pd.DataFrame, **kwargs):
        # data preprocess
        dataset = self.preprocessor.process(data=data,
                                            requisite_cols=None,
                                            **kwargs)
        logger.info(f'Processed data columns: {dataset.columns}.')

        # linucb model_handler
        self.model_handler = LinUCBHandler(data=dataset,
                                           config=self.config,
                                           mode=self.mode,
                                           **kwargs)
        # train
        self.model_handler.train()

    def predict(self, data: pd.DataFrame=None, **kwargs):
        if data is not None:
            dataset = self.preprocessor.process(data, **kwargs)
            scores = self.model_handler.predict(dataset).reshape(-1)
        else:
            scores = self.model_handler.predict()
        return scores
