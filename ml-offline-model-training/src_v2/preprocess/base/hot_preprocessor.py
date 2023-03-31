import pandas as pd

from utils.logger import logger
from src_v2.configs import Config


class BaseHotPreprocessor():

    MODE = ['train', 'validation', 'inference']

    def __init__(self, configs=Config({}), logger=logger, mode='train'):

        assert mode in self.MODE, 'only support train, validation and inference mode'

        self.configs = configs
        self.mode = mode
        self.logger = logger
        self.verbose = self.mode != 'inference'

    def process(self, dataset: pd.DataFrame=None, chain_configs: dict=None, requisite_cols: list=None, **kwargs):

        chain_configs = self.configs.CHAIN_CONFIGS if chain_configs is None else chain_configs
        requisite_cols = self.configs.REQUISITE_COLS if requisite_cols is None else requisite_cols

        for func_name, func_params in chain_configs.items():

            if self.mode in func_params.get('disable_mode', []):
                continue

            try:
                chain_func = getattr(self, func_name)
            except ModuleNotFoundError:
                logger.error(f'Module {func_name} not found.')

            func_kwargs = {**kwargs, **func_params}
            dataset = chain_func(dataset=dataset, **func_kwargs)

        return dataset[requisite_cols]
