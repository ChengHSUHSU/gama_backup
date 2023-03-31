import pyspark.sql.functions as f
from pyspark.sql.types import StringType, ArrayType
from collections import defaultdict
from utils import convert_type
from utils.logger import logger


class ContentDistinctValueParser():
    """Class for parsing distinct values of designated pyspark columns
    """
    COL2TYPE = {
        'cat0': ArrayType(StringType()),
        'cat1': ArrayType(StringType()),
        'cat2': ArrayType(StringType()),
        'tags': ArrayType(StringType()),
        'tags_seo': ArrayType(StringType()),
        'store_id': StringType(),
        'site_name': StringType()
    }

    NER_LABELS = ['person', 'location', 'event', 'organization', 'item', 'others']

    def __init__(self):
        self.col2label = defaultdict(list)

    def parse(self, df, target_cols, add_ner=False):

        logger.info(f'[Data Preparation] Target columns: {target_cols}')

        for col in target_cols:
            logger.info(f'[Data Preparation] Process {col}')

            t_type = self.COL2TYPE.get(col)

            if t_type:
                convert_type_udf = f.udf(convert_type, t_type)
            else:
                convert_type_udf = f.udf(convert_type)

            # get unique value
            if isinstance(t_type, StringType):
                # directly do distinct to string type column values
                df_unique_pd = df.withColumn('unique', f.col(col)) \
                                 .select('unique') \
                                 .distinct() \
                                 .toPandas()
            else:
                df_unique_pd = df.withColumn('unique', convert_type_udf(f.col(col))) \
                                 .withColumn('unique', f.explode(f.col('unique'))) \
                                 .select('unique') \
                                 .distinct() \
                                 .toPandas()

            unique_list = [x.strip() for x in df_unique_pd['unique'].values.tolist() if x]
            self.col2label[col].extend(unique_list)

        if add_ner:
            self.col2label['content_ner'].extend(self.NER_LABELS)

        return self.col2label

    def reset(self):
        self.col2label = defaultdict(list)
