import json
import pyspark.sql.functions as f
import pyspark.sql.types as t
from ast import literal_eval
from datetime import datetime, timedelta
from utils import DAY_STRING_FORMAT, get_partition_path_with_blobs_check, aggregate_view_also_view_udf
from src.gcsreader import dedup_by_col
from src.gcsreader.jollybuy_goods_item2item import JollybuyGoodsItem2ItemReader
from src.gcsreader.utils import filter_dataframe
from .config.jollybuy_config import JollybuyGoodsMultiItem2ItemConfig


# function parse uuid in cart detail
# TODO: Revise to parse click event if detail is added
def parse_jb_cart_page_view(string_of_list):
    list_of_uuid = []
    for detail in literal_eval(string_of_list):
        for product in detail['detail']:
            list_of_uuid.append(product['uuid'])

    return list_of_uuid


parse_jb_cart_page_view_udf = f.udf(parse_jb_cart_page_view, t.ArrayType(t.StringType()))


# TODO: Remove if detail of click event is added
def filter_jb_cart_page_view(df, events, rules):
    # get pair of uuid (goods in cart) and session id
    df_cart_view = df.filter(df.event == events[0])
    df_cart_view = df_cart_view.filter(df_cart_view.page_info.contains(rules[0]))
    df_cart_view = df_cart_view.select(f.get_json_object('page_info', '$.detail').alias('detail'), 'session_id')
    df_cart_view = df_cart_view.filter(df_cart_view.detail != '[]')
    df_cart_view = df_cart_view.withColumn('list_of_uuid', parse_jb_cart_page_view_udf(f.col('detail')))
    df_cart_view = df_cart_view.withColumn('uuid', f.explode('list_of_uuid'))

    return df_cart_view


def filter_jb_cart_content_click(df, events, rules):
    # get pair of click uuid (maybe like sec) and session id
    df_cart_click = df.filter(df.event == events[1])
    df_cart_click = df_cart_click.filter(df_cart_click.page_info.contains(rules[0]))
    df_cart_click = df_cart_click.filter(df_cart_click.click_info.contains(rules[1]))
    df_cart_click = df_cart_click.select(f.get_json_object('click_info', '$.uuid').alias('click_uuid'),
                                         f.get_json_object('click_info', '$.name').alias('click_name'),
                                         'session_id', 'date', 'timestamp')
    df_cart_click = dedup_by_col(df_cart_click, unique_col_base=['click_uuid', 'session_id'], time_col='timestamp')

    return df_cart_click


class JollybuyGoodsMultiItem2ItemReader(JollybuyGoodsItem2ItemReader):

    CONFIG = {
        'jollybuy_goods': JollybuyGoodsMultiItem2ItemConfig
    }

    def get_view_also_view_data(self, metrics='view_also_view', config_key=''):

        all_dates = []
        all_path_list = []

        # get view-also-view scores
        view_also_view_path = self.config.VIEW_ALSO_VIEW_PATH.replace('METRICS', metrics)\
            .replace('PROJECT_ID', self.project_id)\
            .replace('CONTENT_TYPE', self.content_type)

        self.logger.info(f'[Data Preparation][Jollybuy Multi Item2Item] {metrics}_path={view_also_view_path}')
        self.logger.info(f'[Data Preparation][Jollybuy Multi Item2Item] Get {metrics} data')

        for d in range(1, int(self.days) + 1):
            current_date = (datetime.strptime(self.run_time, DAY_STRING_FORMAT) - timedelta(days=d)).strftime(DAY_STRING_FORMAT)
            prefix = f'metrics/{metrics}/{self.content_type}/{current_date}'
            path = view_also_view_path.replace('INPUT_DATE', current_date)

            cur_path_list, _ = get_partition_path_with_blobs_check(path, prefix, bucket_idx=2)

            if len(cur_path_list) != 0:
                all_path_list.extend(cur_path_list)
                all_dates.append(current_date)

        self.logger.info('[Data Preparation][Jollybuy Multi Item2Item] All exist paths')
        self.logger.info(f'[Data Preparation][Jollybuy Multi Item2Item] {metrics} path_list={all_path_list}')
        self.logger.info(f'[Data Preparation][Jollybuy Multi Item2Item] {metrics} dates_list={all_dates}')

        for i, (path, cur_date) in enumerate(zip(all_path_list, all_dates)):
            if i == 0:
                df_view_also_view = self.sql_context.read.options(**{'header': 'true', 'escape': '"'}).csv(path)
                df_view_also_view = df_view_also_view.withColumn('date', f.lit(cur_date))
            else:
                df_view_also_view_tmp = self.sql_context.read.options(**{'header': 'true', 'escape': '"'}).csv(path)
                df_view_also_view_tmp = df_view_also_view_tmp.withColumn('date', f.lit(cur_date))

                df_view_also_view = df_view_also_view.unionByName(df_view_also_view_tmp)

        if config_key:
            df_view_also_view = filter_dataframe(df_view_also_view, config_key, self.config)

            renamed_cols = getattr(self.config, 'RENAMED_COLS', {}).get(config_key, {})
            if renamed_cols:
                for old_col, new_col in renamed_cols.items():
                    df_view_also_view = df_view_also_view.withColumnRenamed(old_col, new_col)

        df_view_also_view = df_view_also_view.withColumn(f'{metrics}_json',
                                                         f.udf(lambda x: json.loads(x), t.MapType(t.StringType(), t.FloatType(), False))(f.col(f'{metrics}_json')))
        df_view_also_view = df_view_also_view.groupby('uuid').agg(f.collect_list(f.struct(f.col(f'{metrics}_json'),
                                                                                          f.col('date'))).alias(f'{metrics}_json'))
        df_view_also_view = df_view_also_view.withColumn(f'{metrics}_json',
                                                         aggregate_view_also_view_udf(view_also_view_col=f'{metrics}_json')(f.struct(f.col('uuid'), f.col(f'{metrics}_json'))))

        df_view_also_view.cache()
        self.logger.info(f'[Data Preparation][Jollybuy Multi Item2Item] {metrics} Data: {df_view_also_view.count()}')
        df_view_also_view.limit(5).show()

        return df_view_also_view

    def get_positive_data(self, df, config_key=''):

        # get positive data from parsed df by event
        if config_key:
            self.logger.info(f'[Data Preparation][Jollybuy Multi Item2Item] Get all positive data')

            events = ['jb_cart_page_view', 'jb_cart_content_click']
            rules = ['"page": "jb_cart_step1"', '"sec": "maybe_like"']

            # get uuid and click uuid pair by mapping session id
            df_cart_view = filter_jb_cart_page_view(df, events, rules)
            df_cart_click = filter_jb_cart_content_click(df, events, rules)
            df_positive = df_cart_view.select('session_id', 'uuid').join(df_cart_click, 'session_id', 'inner')\
                .drop(df_cart_view.session_id).drop(df_cart_click.session_id)
            df_positive = dedup_by_col(df_positive, unique_col_base=['uuid', 'click_uuid'], time_col='timestamp')
        else:
            df_positive = df

        self.logger.info(f'[Data Preparation][Jollybuy Multi Item2Item] Positive data: {df_positive.count()}')
        df_positive.limit(5).show()

        return df_positive
