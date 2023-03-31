from src.gcsreader.base_reader import GcsReader
from src.gcsreader.utils import rename_cols


class ContentReader(GcsReader):

    def process(self, content_type: str):
        """Function to process content reader pipeline

        Args:
            content_type (str)
        """

        self.logger.info('[Data Preparation][Content] get content data from GCS')
        input_path = self.config.CONTENT_PATH.replace('PROJECT_ID', self.project_id).replace('CONTENT_TYPE', content_type)
        df = self.sql_context.read.parquet(input_path)

        self.logger.info('[Data Preparation][Content] rename columns')
        df = rename_cols(df, rename_cols=self.config.STEPS_TO_RENAMCOLS.get('content', []))

        self.logger.info('[Data Preparation][Content] select requisite columns')
        self.data = df.select(self.config.REQUISITE_COLS.get('content', '*'))
