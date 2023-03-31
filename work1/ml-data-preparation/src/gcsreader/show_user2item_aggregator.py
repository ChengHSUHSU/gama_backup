from collections import defaultdict
from datetime import datetime, timedelta
from src.gcsreader.base_aggregator import BaseAggregator
from utils import DAY_STRING_FORMAT

from src.gcsreader.base_reader.user_profile import UserProfileReader
from src.gcsreader.utils import rename_cols


class ShowUser2ItemAggregator(BaseAggregator):
    def read(self, property_name: str, content_type: str, profile_types: list = [], process_metrics_types: list = []):
        
        self.event_reader.process(property_name=property_name, content_type=content_type)
        
        self.content_reader.process(content_type=content_type)
        self.user_profile_run_time = (datetime.strptime(self.run_time, DAY_STRING_FORMAT) - timedelta(days=1)).strftime(DAY_STRING_FORMAT)

        for metrics_type in process_metrics_types:
            self.metrics_reader.process(content_type=content_type, metrics_type=metrics_type)

        # process different profile type by iterate list (ex. category, tag)
        for profile_type in profile_types:
            self.user_profile_reader.process(content_type='beanfun_meta' if profile_type == 'meta' else content_type,
                                             profile_type=profile_type,
                                             user_profile_run_time=self.user_profile_run_time)
        # read other data
        '''
        TODO
        把多個不同content_type 的 user profile的case整在BaseAggregator內
        '''
        other_data = self.read_other_data()

        self.data = {'event': self.event_reader.get(),
                     'content': self.content_reader.get(),
                     'metrics': self.metrics_reader.get(),
                     'user_profile': self.user_profile_reader.get(),
                     'other_data': other_data}

    def read_other_data(self):
        other_data = defaultdict(list)
        # category-profile / embedding (other content-type)
        meta_content_types_info = self.config.PROCESS_META_CONTENT_TYPES_INFO
        for meta_content_type_info in meta_content_types_info:
            # init user_profile_reader
            meta_user_profile_reader = UserProfileReader(project_id=self.project_id, sql_context=self.sql_context,
                                                        run_time=self.run_time, days=self.days, config=self.config, logger=self.logger)
            meta_content_type = meta_content_type_info['content_type']
            profile_types = meta_content_type_info['profile_types']
            rename_by = meta_content_type_info['rename_by']
            # load data
            for profile_type in profile_types:
                meta_user_profile_reader.process(content_type=meta_content_type, 
                                                 profile_type=profile_type, 
                                                 user_profile_run_time=self.user_profile_run_time)
            # rename data
            meta_user_data = meta_user_profile_reader.data
            for profile_type in profile_types:
                rename_columns = self.config.STEPS_TO_RENAMCOLS.get(rename_by, {}).get(profile_type, [])
                meta_user_data[profile_type] = [rename_cols(meta_user_data[profile_type], rename_cols=rename_columns), rename_columns[0][0]]
            other_data[meta_content_type] = meta_user_data
        return other_data
