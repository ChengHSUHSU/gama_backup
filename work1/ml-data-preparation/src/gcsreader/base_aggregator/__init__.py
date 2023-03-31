from abc import abstractmethod
from multipledispatch import dispatch
from pyspark.sql import DataFrame
from src.gcsreader.base_reader.event import EventReader
from src.gcsreader.base_reader.metrics import MetricsReader
from src.gcsreader.base_reader.content import ContentReader
from src.gcsreader.base_reader.user_profile import UserProfileReader


class BaseAggregator():
    def __init__(self, project_id, sql_context, run_time, days, config, logger):

        self.project_id = project_id
        self.sql_context = sql_context
        self.days = days
        self.run_time = run_time
        self.config = config
        self.logger = logger

        self.event_reader = EventReader(project_id=project_id, sql_context=sql_context,
                                        run_time=run_time, days=days, config=config, logger=logger)

        self.metrics_reader = MetricsReader(project_id=project_id, sql_context=sql_context,
                                            run_time=run_time, days=days, config=config, logger=logger)

        self.content_reader = ContentReader(project_id=project_id, sql_context=sql_context,
                                            run_time=run_time, days=days, config=config, logger=logger)

        self.user_profile_reader = UserProfileReader(project_id=project_id, sql_context=sql_context,
                                                     run_time=run_time, days=days, config=config, logger=logger)

        self.data = {'event': None, 'content': None, 'metrics': None, 'user_profile': None}

    @abstractmethod
    def read(self):
        pass

    @dispatch()
    def get(self) -> dict:
        """Function to get all aggregeted dataframe

        Returns:
            dict
        """
        return self.data

    @dispatch(str)
    def get(self, data_type: str) -> DataFrame:
        """Function to get specific aggregeted dataframe

        Args:
            data_type (str): parameter to select specific dataframe.

        Returns:
            DataFrame
        """

        if data_type not in self.data:
            raise Exception(f'Aggregator does not support data type: {data_type}')
        return self.data[data_type]
