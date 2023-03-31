from datetime import datetime, timedelta
from src.gcsreader.base_aggregator import BaseAggregator
from utils import DAY_STRING_FORMAT


class GraphEmbeddingModelAggregator(BaseAggregator):

    def read(self, property_name, content_type, profile_types, process_metrics_types):

        self.event_reader.process(property_name=property_name, content_type=content_type)

        self.content_reader.process(content_type=content_type)
        self.content_reader.data = self.content_reader.data.select(
            self.config.REQUISITE_CONTENT_COLS.get(content_type,
                                                   self.config.REQUISITE_CONTENT_COLS['default']))

        user_profile_run_time = (datetime.strptime(self.run_time, DAY_STRING_FORMAT) - timedelta(days=self.days)).strftime(DAY_STRING_FORMAT)

        for metrics_type in process_metrics_types:
            self.metrics_reader.process(content_type=content_type, metrics_type=metrics_type)

        # process different profile type by iterate list (ex. category, tag)
        for profile_type in profile_types:
            self.user_profile_reader.process(content_type=content_type, profile_type=profile_type, user_profile_run_time=user_profile_run_time)

        self.data = {'event': self.event_reader.get(),
                     'content': self.content_reader.get(),
                     'metrics': self.metrics_reader.get(),
                     'user_profile': self.user_profile_reader.get()}
