
from src.gcsreader.base_aggregator import BaseAggregator

import pyspark.sql.functions as f


class PlanetNewsHotModelAggregator(BaseAggregator):
    
    def read(self, property_name, content_type, process_metrics_types):
        
        self.content_type = content_type
        self.event_reader.process_hourly(property_name=property_name, content_type=content_type)
        self.content_reader.process(content_type=content_type)

        for metrics_type in process_metrics_types:
            self.metrics_reader.process(content_type=content_type, metrics_type=metrics_type)

        self.data = {
            'event': self.event_reader.get(),
            'content': self.content_reader.get(),
            'metrics': self.metrics_reader.get()
            }

    def get_reward_cnts(self, df_event_count):
        df_event_count = df_event_count.withColumn('reward',
                                        f.col('planet_content_click')
                                        + f.col('home_page_content_click')
                                        + f.col('exploration_page_content_click'))
        df_reward = df_event_count.select('date', 'hour', 'content_id', 'reward')
        return df_reward

    def get_event_count(self, df_event, dropDuplicates=False):
        condition_list = self.config.EVENT_OF_CONTENT_TYPE_CONDITIONS[self.content_type]
        df_event_count = None
        used_events = set()
        for condition in condition_list:
            event = condition['event']
            if event in used_events:
                continue
            used_events.add(event)
            df = df_event.filter(f.col('event') == event)
            # drop duplicate
            if dropDuplicates is True:
                df = df.dropDuplicates(['content_id', 'userid', 'date', 'hour'])
            # groupby (date, content_id)
            df = df.groupBy('date', 'hour', 'content_id').count().select(
                        f.col('date'), f.col('hour'), f.col('content_id'), f.col('count').alias(event))
            # join
            if df_event_count is None:
                df_event_count = df
            else:
                df_event_count = df_event_count.join(df, on=['date', 'hour', 'content_id'], how='outer')
        df_event_count = df_event_count.fillna(0)
        return df_event_count

    def get_negative_candidates(self, df_data):
        df_negative_candidates = df_data.groupby(['date', 'hour']) \
                                        .agg(f.collect_set(f.col('content_id')).alias('content_id_neg'))
        return df_negative_candidates
