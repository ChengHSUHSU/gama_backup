# TODO: replace to `gcsreader.__init__.py`  after refactor
from abc import abstractmethod
from typing import Union
from datetime import datetime, timedelta
from pyspark.sql import DataFrame
from utils import DAY_STRING_FORMAT
import logging


class GcsReader():

    def __init__(self, project_id, sql_context, run_time, days=30, config=None, logger=logging):

        self.project_id = project_id
        self.sql_context = sql_context
        self.run_time = run_time
        self.days = days
        self.config = config
        self.logger = logger
        self.data = None

    @abstractmethod
    def process(self):
        pass

    def get(self) -> Union[DataFrame, dict]:
        """Function to get processed spark dataframe

        Returns:
            Union[DataFrame, dict]: For user profile data return dict of DataFrame, otherwise return DataFrame
        """
        return self.data
