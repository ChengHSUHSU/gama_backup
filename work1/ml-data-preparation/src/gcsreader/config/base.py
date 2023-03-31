from __future__ import annotations
from typing import Union
from utils.logger import logger


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


GeneralConfig = Config({
    'PROCESS_CONTENT_TYPES': [],
    'PROCESS_PROFILE_TYPES': [],
    'PROCESS_METRICS_TYPES': [],
    'CONTENT_TYPE_TO_PROPERTY_TYPE': {},

    # columns to be renamed at each step in format of {'step': [(old_col_name, new_col_name), ...]}
    'STEPS_TO_RENAMCOLS': {},

    # target columns of each step in format of {'step': [col1, col2, ...]}
    'REQUISITE_COLS': {},

    'FINAL_COLS': []
})


class BaseConfig():
    CONTENT_PATH = 'gs://content-PROJECT_ID/content_daily/property=beanfun/content_type=CONTENT_TYPE/snapshot/*.parquet'
    EVENT_PATH = 'gs://event-PROJECT_ID/event_daily/date=INPUT_DATE/property=PROPERTY/is_page_view=*/event=EVENT/*.parquet'
    VIEW_ALSO_VIEW_PATH = 'gs://pipeline-PROJECT_ID/metrics/view_also_view/CONTENT_TYPE/INPUT_DATE/view_also_view_CONTENT_TYPE.csv'
    METRICS_POPULARITY_PATH = 'gs://pipeline-PROJECT_ID/metrics/popularity/POPULARITY_FOLDER/INPUT_DATE/INPUT_HOUR/*.csv'
    USER_PROFILE_JOIN_DAY_DIFF = 5     # TODO: set default join date range to 1 after bug fix
