import numpy as np
from bdds_recommendation.src.hot.base_handler import BanditHotModelHandler
from bdds_recommendation.src.module.bandit.models.linucb import NumbaLinUBC


class NumbaLinUCBHandler(BanditHotModelHandler):

    def __init__(self, opt, logger, configs, preprocesser, is_train=True):

        self.opt = opt
        self.configs = configs
        self.logger = logger
        self.preprocesser = preprocesser
        self.is_train = is_train

    def code(cls):
        return 'NumbaLinUCB'

    def init_model(self):
        try:
            self.arms = self.dataset[getattr(self.configs, 'ARM_COL', 'uuid')].unique()
            self.context_size = len(self.dataset.columns) - 3  # 3 is for arm, candidates and reward columns
            self.model = NumbaLinUBC(arms=self.arms, context_size=self.context_size,
                                     alpha=self.opt.alpha, logger=self.logger)
        except NameError:
            self.logger.error('preprocess must be executed before model initialization')

    def preprocess(self, **kwargs):
        # must run before model initialization
        self.dataset = self.preprocesser.process(**kwargs)

    def train(self):

        self.logger.info('Start Training')
        self.model.train(data=self.dataset, epochs=self.opt.epochs,
                         n_visits=self.opt.n_visits, shuffle=self.opt.shuffle,
                         arm_col=getattr(self.configs, 'ARM_COL', 'uuid'), candidate_col=getattr(self.configs, 'CANDIDATE_COL', 'uuids'),
                         reward_col=getattr(self.configs, 'REWARD_COL', 'uuid'), warmup_iters=self.opt.warmup_iters)

    def predict(self):

        return np.squeeze(self.model.ucb_mean)
