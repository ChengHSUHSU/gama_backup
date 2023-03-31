from datetime import datetime, timedelta
from src.gcsreader.base_aggregator import BaseAggregator
from utils import DAY_STRING_FORMAT


class PlanetNewsUser2ItemAggregator(BaseAggregator):

    def read(self,
             property_name='beanfun',
             content_type='planet_news',
             processed_data=['event', 'content', 'metrics', 'user_profile']):

        if 'event' in processed_data:
            self.event_reader.process(property_name=property_name, content_type=content_type)
            self.data['event'] = self.event_reader.get()

        if 'content' in processed_data:
            self.content_reader.process(content_type=content_type)
            self.data['content'] = self.content_reader.get()

        if 'user_profile' in processed_data:

            user_profile_run_time = (datetime.strptime(self.run_time, DAY_STRING_FORMAT) - timedelta(days=1)).strftime(DAY_STRING_FORMAT)
            for profile_type in self.config.REQUISITE_COLS['user_profile']:
                self.user_profile_reader.process(content_type='gamania_meta' if profile_type == 'meta' else content_type,
                                                 profile_type=profile_type,
                                                 user_profile_run_time=user_profile_run_time)

            self.data['user_profile'] = self.user_profile_reader.get()

        if 'metrics' in processed_data:

            for metrics_type in self.config.REQUISITE_COLS['metrics']:
                self.metrics_reader.process(content_type=content_type, metrics_type=metrics_type)

            self.data['metrics'] = self.metrics_reader.get()
