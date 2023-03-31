import numpy as np
import pandas as pd

from utils.logger import logger
from src_v2.configs import Config
from src_v2.handler import BaseModelHandler
from src.module.bandit.models.linucb import NumbaLinUBC


class LinUCBHandler(BaseModelHandler):

    def __init__(self, data: pd.DataFrame, config: Config, mode: str='train', **kwargs):
        super(LinUCBHandler, self).__init__(mode)
        self.mode = mode
        self.config = config
        self.model_params = kwargs
        self.dataset = data
        self.initialize(**kwargs)

    def code(cls):
        return 'LinUCB'

    def initialize(self, **kwargs):
        self.arms = self.dataset[getattr(self.config, 'ARM_COL', 'uuid')].unique()
        context_size = len(self.dataset.columns) - 3  # 3 is for arm, candidates and reward columns
        self.model = NumbaLinUBC(arms=self.arms, context_size=context_size, alpha=kwargs['alpha'], logger=logger)

    def train(self):
        logger.info('[Model Training][Planet News Hot Item] Start Training.')
        self.model.train(data=self.dataset, 
                         arm_col=getattr(self.config, 'ARM_COL', 'uuid'), 
                         candidate_col=getattr(self.config, 'CANDIDATE_COL', 'candidate_neg'),
                         reward_col=getattr(self.config, 'REWARD_COL', 'reward'), 
                         **self.model_params)
        logger.info('[Model Training][Planet News Hot Item] End Training.')

    def predict(self, data: pd.DataFrame=None):
        score_array = np.squeeze(self.model.ucb_mean)
        if data is not None:
            content_id = getattr(self.config, 'ARM_COL', 'uuid')
            content_ids = list(data[content_id].apply(lambda x: self.model.arm2index[x]))
            return score_array[content_ids]
        else:
            return score_array

    def load_model(self):
        return 
    
    def save_model(self):
        return
