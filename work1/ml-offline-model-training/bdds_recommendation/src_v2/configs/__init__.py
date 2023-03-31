from __future__ import annotations
from typing import Union
from bdds_recommendation.utils.logger import logger


class Config(object):
    """
    Holds the configuration for anything you want it to.
    To use, just do cfg.x instead of cfg['x'].
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def update(self, new_config_dict: Union[Config, dict]) -> Config:
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))

        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict: Union[Config, dict]):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            logger.info(k, ' = ', v)


GeneralConfigs = Config({
    'REQUISITE_COLS': [],
    'CONTENT_TYPE': '',
    'SERVICE_TYPE': '',
    'BASELINE_MODEL_PATH': './ml-model',
    'PERFORMANCE_IMPROVEMENT_RATE_POSTFIX': '_pir'   # PIR means performance improvement rate
})
