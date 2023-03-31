from src.gcsreader.config.nsfw_text_configs import NSFWTextConfig
from src.gcsreader import GcsReader
import logging


class NSFWTextReader(GcsReader):

    CONFIG = NSFWTextConfig

    def __init__(self, project_id, sql_context, content_type, logger=logging):

        self.project_id = project_id
        self.sql_context = sql_context
        self.content_type = content_type
        self.logger = logger

    def get_event_data(self):
        pass

    def get_content_data(self):

        self.logger.info('[Data Preparation][NSFW Text] Get content data')

        input_path = self.CONFIG.CONTENT_PATH \
            .replace('PROJECT_ID', self.project_id) \
            .replace('CONTENT_TYPE', self.content_type)

        df_content = self.sql_context.read.parquet(input_path)
        df_content = df_content.select(self.CONFIG.CONTENT_COL)
        return df_content
