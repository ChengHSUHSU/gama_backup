import pyspark.sql.functions as f
from pyspark.sql.types import StringType, MapType
from pyspark.sql import DataFrame
from src.gcsreader.base_reader import GcsReader
from src.gcsreader.utils import get_existed_blobs, rename_cols


class EventReader(GcsReader):

    def process(self, property_name: str, content_type: str):
        """Function to process event reader pipeline

        Args:
            property_name (str)
            content_type (str)
        """
        self.condition_list = self.config.EVENT_OF_CONTENT_TYPE_CONDITIONS[content_type]
        self.logger.info(f'[Data Preparation][Event] get {content_type} event data from GCS')
        df = self._load(property_name)
        self.logger.info(f'[Data Preparation][Event] add userid column')
        df = self._add_userid(df)

        self.logger.info(f'[Data Preparation][Event] filter event by \n{self.condition_list}')
        df = self._filter_event(df)

        self.logger.info(f'[Data Preparation][Event] parse event information')
        df = self._parse_event(df)

        self.logger.info(f'[Data Preparation][Event] rename columns')
        df = rename_cols(df, rename_cols=self.config.STEPS_TO_RENAMCOLS.get('event', []))

        self.logger.info('[Data Preparation][Event] select requisite columns')
        self.data = df.select(self.config.REQUISITE_COLS['event'])

    def _load(self, property_name: str) -> DataFrame:
        """Function to load data from gcs

        Args:
            property_name (str)
        """

        input_path = self.config.EVENT_PATH.replace('PROJECT_ID', self.project_id)
        base_path = input_path[:input_path.find('date=')]
        path_list = get_existed_blobs(run_time=self.run_time,
                                      input_path=input_path,
                                      days=self.days,
                                      prefix=f'event_daily/date=INPUT_DATE/property={property_name}',
                                      bucket_idx=2)

        self.logger.info(f'existed event path:\n {path_list}')
        df = self.sql_context.read.option('basePath', base_path).parquet(*path_list)

        return df

    def _add_userid(self, df, main_col='openid', sub_col='trackid', output_col='userid') -> DataFrame:
        """Add userid column using main_col, otherwise using sub_col

        Args:
            df (DataFrame)
            main_col (str, optional): main column for userid. Defaults to 'openid'.
            sub_col (str, optional): sub column for userid if main column is not exist. Defaults to 'trackid'.
            output_col (str, optional): Defaults to 'userid'.

        Returns:
            DataFrame
        """

        condition_col = ((f.col(main_col).isNull()))
        df = df.withColumn(output_col, f.when(condition_col, f.col(sub_col)).otherwise(f.col(main_col)))
        df = df.filter(df.userid.isNotNull())

        return df

    def _filter_event(self, df: DataFrame) -> DataFrame:
        """Function to filter event by condition list

        Args:
            df (DataFrame)

        Returns:
            DataFrame
        """

        or_conds = []

        for info_type in ['page_info', 'click_info']:
            df = df.withColumn(f'{info_type}_map', f.from_json(f.col(info_type), MapType(StringType(), StringType())))

        for cond in self.condition_list:
            and_conds = []
            for cond_key in cond:
                cond_value = f'"{cond[cond_key]}"'
                and_conds.append(f'{cond_key} = {cond_value}')
            or_conds.append(' and '.join(and_conds))

        _cond = ') or ('.join(or_conds)
        filter_cond = f'({_cond})'

        query = f'SELECT * FROM full_event WHERE {filter_cond}'
        df.createOrReplaceTempView('full_event')
        df_filtered = self.sql_context.sql(query)

        return df_filtered

    def _parse_event(self, df):
        """Function to parse event by condition list

        Args:
            df (DataFrame)

        Returns:
            DataFrame
        """
        df_result = None

        for cond in self.condition_list:
            cond_event = cond.get('event', None)

            if not cond_event:
                continue

            event_postfix = cond_event.split('_')[-1]
            info_prefix = 'click' if event_postfix == 'click' else 'page'
            df_cond = df.withColumn('uuid', f.get_json_object(f.col(f'{info_prefix}_info'), '$.uuid'))

            if df_result is None:
                df_result = df_cond
            else:
                df_result = df_result.unionByName(df_cond)

        df_result = df_result.withColumn('date', f.col('date').cast('string'))

        return df_result
