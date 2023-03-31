import pandas as pd
from utils.logger import logger
from src_v2.preprocess.base import BaseHotPreprocessor


class PlanetNewsHotLinUCBPreprocessor(BaseHotPreprocessor):

    def __init__(self, configs, mode='train', logger=logger, **kwargs):
        super(PlanetNewsHotLinUCBPreprocessor, self).__init__(configs, logger, mode)
    
    def _process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.verbose:
            self.logger.info(f'---process---')  
        return data
