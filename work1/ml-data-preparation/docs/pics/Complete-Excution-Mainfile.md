# 資料準備 (Data Preparation) <a name="資料準備"></a>

> 1. 利用 Reader, Aggregator 來收集整理 Event, Content, Statistics, ... 資料
> 2. 建立正負樣本


```python
# 為了要在 JupyterLab cell上 開發 
# 首先，先增加 WORK/ml-data-preparation 的路徑，這樣才能 import ml-data-preparation 裡面的物件
import sys
sys.path.insert(0, 'WORK/ml-data-preparation/')
```


```python
from utils.logger import Logger
import pyspark.sql.functions as f
from pyspark.sql.types import StringType
from src.gcsreader import dedup_by_col, join_event_with_user_profile
from src.spark_negative_sampler import generate_negative_samples
from src.gcsreader.jollybuy_goods_user2item_aggregator import JollybuyGoodsUser2ItemAggregator
from src.gcsreader.config.jollybuy_goods.jollybuy_goods_user2item import JollybuyGoodsUser2ItemConfig
from utils import initial_spark, filter_by_interaction_count, prune_preference_user_profile
from utils.parser import ContentDistinctValueParser
```

# 參數設定  <a name="資料準備"></a>


```python
# 參數設定
config = JollybuyGoodsUser2ItemConfig
project_id = 'bf-data-uat-001'
run_time = '20230201'
days = 30
content_negative_sample_size = 10
pop_negative_sample_size = 10
requisite_event_sequence_length = 10
enable_positive_sampling = False
daily_positive_sample_size = 5000
daily_sample_seed = 1024
logger = Logger(logger_name='demo', dev_mode=True)
enable_positive_sampling = False
content_property = 'beanfun'
content_type = 'jollybuy_goods'
USER_PROFILES_TO_JOINED_WITH_DATE = ['user_title_embedding', 'user_category', 'user_tag']
POP_NEGATIVE_MULTIPLIER = 100
```


```python
# 初始化 spark session
sql_context, spark_session = initial_spark(cores='5', memory='27G',
                                           overhead='5G', shuffle_partitions='1000',
                                           num_executors='3')
sql_context.sql("SET spark.sql.autoBroadcastJoinThreshold = -1")  # disable broadcast join
```

    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    23/02/20 13:39:18 INFO org.apache.spark.SparkEnv: Registering MapOutputTracker
    23/02/20 13:39:18 INFO org.apache.spark.SparkEnv: Registering BlockManagerMaster
    23/02/20 13:39:18 INFO org.apache.spark.SparkEnv: Registering BlockManagerMasterHeartbeat
    23/02/20 13:39:18 INFO org.apache.spark.SparkEnv: Registering OutputCommitCoordinator





    DataFrame[key: string, value: string]



# 用 Reader, Aggregator 來收集整理,  使用者/商品 的 event, content, title embedding, tag, popularity, statistics 資料  <a name="資料準備"></a>


```python
# 使用 Reader, Aggregator 來收集整理,  使用者/商品 的 event, content ,title embedding, tag, popularity, statistics 資料 
# 參考資料 ： https://docs.google.com/document/d/1QbA5MY5LQGmC22rzTYf_Tx0UIhzkSDls6-RIokYrTIM/edit
def get_raw_data(configs: JollybuyGoodsUser2ItemConfig, aggregator: JollybuyGoodsUser2ItemAggregator, content_type: str, content_property: str) -> dict:
    process_metrics_types = configs.PROCESS_METRICS_TYPES
    process_profile_types = configs.PROCESS_PROFILE_TYPES
    aggregator.read(content_property, content_type, process_profile_types, process_metrics_types)
    data = aggregator.get()
    df_event, df_content, df_popularity, df_user_profile = \
        data['event'], data['content'], data['metrics'], data['user_profile']
    raw_data = {
        'user_event': df_event,
        'item_content': df_content,
        'user_title_embedding': df_user_profile['title_embedding'],
        'user_category': df_user_profile['category'],
        'user_tag': df_user_profile['tag'],
        'user_meta': df_user_profile['meta'],
        'popularity': df_popularity['popularity'],
        'snapshot_popularity': df_popularity['snapshot_popularity']
    }
    return raw_data
```


```python
# 使用 Reader, Aggregator 來收集整理,  使用者/商品 的 event, content ,title embedding, tag, popularity, statistics 資料 
# 參考資料 ： https://docs.google.com/document/d/1QbA5MY5LQGmC22rzTYf_Tx0UIhzkSDls6-RIokYrTIM/edit
logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Get raw data from GCS')

aggregator = JollybuyGoodsUser2ItemAggregator(project_id, sql_context, run_time, days, config=config, logger=logger)
raw_data = get_raw_data(configs=config, aggregator=aggregator, content_type=content_type, content_property=content_property)
```

    [INFO] [Data Preparation][Jollybuy Goods User2Item] Get raw data from GCS
    [INFO] [Data Preparation][Event] get jollybuy_goods event data from GCS
    [INFO] existed event path:
     ['gs://event-bf-data-uat-001/event_daily/date=20230201/property=jollybuy/is_page_view=*/event=*/*.parquet']
    [INFO] [Data Preparation][Event] add userid column
    [INFO] [Data Preparation][Event] filter event by 
    [{'event': 'jb_goods_page_view', 'page_info_map["page"]': 'jb_goods'}]
    [INFO] [Data Preparation][Event] parse event information
    cond_event :  jb_goods_page_view
    info_prefix :  view
    [INFO] [Data Preparation][Event] rename columns
    [INFO] [Data Preparation][Event] select requisite columns
    [INFO] [Data Preparation][Content] get content data from GCS
    [INFO] [Data Preparation][Content] rename columns
    [INFO] [Data Preparation][Content] select requisite columns
    [INFO] [Data Preparation][Metrics] get jollybuy_goods popularity score data
    [INFO] [Data Preparation][Metrics] extend generated time
    [INFO] [Data Preparation][Metrics] rename columns
    [INFO] [Data Preparation][Metrics] select requisite columns
    [INFO] [Data Preparation][Metrics] get jollybuy_goods latest popularity score data
    [INFO] [Data Preparation][Metrics] rename columns
    [INFO] [Data Preparation][Metrics] select requisite columns
    [INFO] [Data Preparation][UserProfile] set user profile run time: 20230131
    [INFO] [Data Preparation][UserProfile] get user jollybuy_goods title_embedding profile


    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:01 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:02 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.


    [INFO] existed user profile path:
     ['gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230131/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230130/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230129/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230128/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230126/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230125/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230124/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230123/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230122/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230121/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230120/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230119/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230118/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230117/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230116/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230115/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230114/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230112/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230111/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230109/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230108/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230107/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230106/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230105/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230104/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/embedding/20230103/goods/*.parquet']
    [INFO] [Data Preparation][UserProfile] extend generated time
    [INFO] [Data Preparation][UserProfile] set 5 day range before event
    [INFO] [Data Preparation][UserProfile] rename columns
    [INFO] [Data Preparation][UserProfile] select requisite columns
    [INFO] [Data Preparation][UserProfile] set user profile run time: 20230131
    [INFO] [Data Preparation][UserProfile] get user jollybuy_goods category profile


    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:03 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:04 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:04 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:04 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:04 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:04 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:04 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:04 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:04 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:04 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:04 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.


    [INFO] existed user profile path:
     ['gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230131/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230130/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230129/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230128/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230126/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230125/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230124/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230123/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230122/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230121/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230120/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230119/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230118/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230117/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230116/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230115/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230114/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230112/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230111/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230109/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230108/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230107/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230106/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230105/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230104/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/category/20230103/goods/*.parquet']
    [INFO] [Data Preparation][UserProfile] extend generated time
    [INFO] [Data Preparation][UserProfile] set 5 day range before event
    [INFO] [Data Preparation][UserProfile] rename columns
    [INFO] [Data Preparation][UserProfile] select requisite columns
    [INFO] [Data Preparation][UserProfile] set user profile run time: 20230131
    [INFO] [Data Preparation][UserProfile] get user jollybuy_goods tag profile


    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:05 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:06 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:06 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:06 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:06 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:06 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.
    23/02/20 13:47:06 WARN org.apache.spark.SparkContext: The path WORK_dp/ml-data-preparation/datapreparation.zip has been added already. Overwriting of added paths is not supported in the current version.


    [INFO] existed user profile path:
     ['gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230131/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230130/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230129/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230128/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230126/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230125/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230124/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230123/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230122/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230121/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230120/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230119/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230118/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230117/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230116/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230115/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230114/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230112/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230111/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230109/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230108/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230107/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230106/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230105/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230104/goods/*.parquet', 'gs://pipeline-bf-data-uat-001/user_profiles_cassandra/jollybuy/tag/20230103/goods/*.parquet']
    [INFO] [Data Preparation][UserProfile] extend generated time
    [INFO] [Data Preparation][UserProfile] set 5 day range before event
    [INFO] [Data Preparation][UserProfile] rename columns
    [INFO] [Data Preparation][UserProfile] select requisite columns
    [INFO] [Data Preparation][UserProfile] set user profile run time: 20230131
    [INFO] [Data Preparation][UserProfile] get user gamania_meta meta profile
    [INFO] existed user profile path:
     ['gs://pipeline-bf-data-uat-001/user_profiles_cassandra/gamania/meta/meta/20230131/*.parquet']
    [INFO] [Data Preparation][UserProfile] extend generated time
    [INFO] [Data Preparation][UserProfile] rename columns
    [INFO] [Data Preparation][UserProfile] select requisite columns



```python
# prune data by userid and content_id
df_distinct_userid = raw_data['user_event'].select('userid').dropDuplicates(subset=['userid'])
df_distinct_content_id = raw_data['user_event'].select('content_id').dropDuplicates(subset=['content_id'])

pruned_by_userid_cols = getattr(config, 'PRUNED_BY_USERID_COLS', [])
pruned_by_content_id_cols = getattr(config, 'PRUNED_BY_CONTENT_ID_COLS', [])

for profile_type in raw_data.keys():
    logger.info(f'Origin {profile_type} profile count : {raw_data[profile_type].count()}')
    if profile_type in pruned_by_userid_cols:
        raw_data[profile_type] = raw_data[profile_type].join(df_distinct_userid, how='inner', on=['userid'])
    elif profile_type in pruned_by_content_id_cols:
        raw_data[profile_type] = raw_data[profile_type].join(df_distinct_content_id, how='inner', on=['content_id'])
    logger.info(f'Pruned {profile_type} profile count : {raw_data[profile_type].count()}')
```

    [INFO] Origin user_event profile count : 2134
    [INFO] Pruned user_event profile count : 2134
    [INFO] Origin item_content profile count : 16802
    [INFO] Pruned item_content profile count : 16


                                                                                    

    [INFO] Origin user_title_embedding profile count : 911


                                                                                    

    [INFO] Pruned user_title_embedding profile count : 295


                                                                                    

    [INFO] Origin user_category profile count : 189517


                                                                                    

    [INFO] Pruned user_category profile count : 6957
    [INFO] Origin user_tag profile count : 242


                                                                                    

    [INFO] Pruned user_tag profile count : 76
    [INFO] Origin user_meta profile count : 65420
    [INFO] Pruned user_meta profile count : 480
    [INFO] Origin popularity profile count : 46
    [INFO] Pruned popularity profile count : 46
    [INFO] Origin snapshot_popularity profile count : 46
    [INFO] Pruned snapshot_popularity profile count : 46



```python
# prune preference profile
logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Preference profile filtering')

for profile_type, type2cond in getattr(config, 'PREFERENCE_FILTER_CONDITIONS', {}).items():
    logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Filter {profile_type}')
    for tag_type, cond in type2cond.items():
        raw_data[profile_type] = raw_data[profile_type].withColumn(profile_type,
            f.udf(lambda x: prune_preference_user_profile(x, tag_type=tag_type, condition=cond), StringType())(f.col(profile_type)))
```

    [INFO] [Data Preparation][Jollybuy Goods User2Item] Preference profile filtering
    [INFO] [Data Preparation][Jollybuy Goods User2Item] Filter user_tag



```python
pruned_by_content_id_cols
```




    ['item_content']




```python
pruned_by_userid_cols
```




    ['user_title_embedding', 'user_category', 'user_tag', 'user_meta']



# 建立正樣本 <a name="資料準備"></a>


```python
df_positive = raw_data['user_event']
# 建立正樣本的 dataframe
if bool(requisite_event_sequence_length):
    logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Filter out event with small click count')
    logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Filter threshold {requisite_event_sequence_length}')
    df_positive = filter_by_interaction_count(df_positive, primary_col='userid', requisite_sequence_length=requisite_event_sequence_length)
    logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Positive event count {df_positive.count()}')
# 抽樣 訓練用的正樣本資料
if enable_positive_sampling:
    logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Positive Sampling by Date')
    logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Random seed: {daily_sample_seed}')
    logger.info(f'[Data Preparation][Jollybuy Goods User2Item] {daily_positive_sample_size} samples per day')
    df_positive = sample_positive_data(df_positive, daily_positive_sample_size, daily_sample_seed)
df_positive.toPandas().head(2)
```

    [INFO] [Data Preparation][Jollybuy Goods User2Item] Filter out event with small click count
    [INFO] [Data Preparation][Jollybuy Goods User2Item] Filter threshold 10
    [INFO] [Data Preparation][Jollybuy Goods User2Item] Positive event count 588





<div>
<style scoped>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userid</th>
      <th>content_id</th>
      <th>date</th>
      <th>hour</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42db33f702aecadbb553ec4a02828a95</td>
      <td>P08232311669</td>
      <td>20230201</td>
      <td>00</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42db33f702aecadbb553ec4a02828a95</td>
      <td>P11243443377</td>
      <td>20230201</td>
      <td>01</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 建立正樣本 
logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Join positive event data with user profile data')

for profile_key in USER_PROFILES_TO_JOINED_WITH_DATE:
    logger.info(f'processing join {profile_key} data')
    df_profile = raw_data[profile_key]
    condition = [df_positive.userid == df_profile.userid, df_positive.date == df_profile.date]
    df_positive = join_event_with_user_profile(df_positive, df_profile, cond=condition, how='left')
```

    [INFO] [Data Preparation][Jollybuy Goods User2Item] Join positive event data with user profile data
    [INFO] processing join user_title_embedding data
    [INFO] processing join user_category data
    [INFO] processing join user_tag data



```python
# append meta
user_profile_col = df_positive.columns
df_positive = df_positive.join(raw_data['popularity'], on=['content_id', 'date', 'hour'], how='left')
df_positive = df_positive.join(raw_data['user_meta'], on=['userid'], how='left')
df_positive = df_positive.join(raw_data['item_content'], on=['content_id'], how='inner')

# negative sampling
df_content_pool = df_positive.select(['content_id', 'publish_time']).distinct()

# negative sampling
df_dedup_positive = dedup_by_col(df_positive, unique_col_base=['userid'], time_col='publish_time')
df_dedup_positive.toPandas().head(2)
```

                                                                                    




<div>
<style scoped>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content_id</th>
      <th>userid</th>
      <th>date</th>
      <th>hour</th>
      <th>count</th>
      <th>user_title_embedding</th>
      <th>user_category</th>
      <th>user_tag</th>
      <th>final_score</th>
      <th>gender</th>
      <th>age</th>
      <th>item_title_embedding</th>
      <th>cat0</th>
      <th>cat1</th>
      <th>tags</th>
      <th>publish_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P15253343663</td>
      <td>1010207000816002123</td>
      <td>20230201</td>
      <td>06</td>
      <td>31</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>m</td>
      <td>46</td>
      <td>[-0.0061460197903215885, -0.014970659278333187...</td>
      <td>['玩具公仔']</td>
      <td>['盒玩']</td>
      <td>[]</td>
      <td>1672380498000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P05260469872</td>
      <td>1010377220816002617</td>
      <td>20230201</td>
      <td>08</td>
      <td>16</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>m</td>
      <td>38</td>
      <td>[0.0009404125157743692, -0.014105713926255703,...</td>
      <td>['玩具公仔']</td>
      <td>['轉蛋']</td>
      <td>[]</td>
      <td>1675237045000</td>
    </tr>
  </tbody>
</table>
</div>



# 建立負樣本 <a name="資料準備"></a>


```python
# 建立負樣本 df_negative_total
# generate negative by top-k popular items
logger.info(
    f'[Data Preparation][Jollybuy Goods User2Item] Generate negative sample from top-{pop_negative_sample_size * POP_NEGATIVE_MULTIPLIER} popular items')
df_snapshot_pop = raw_data['snapshot_popularity']
df_snapshot_pop = df_snapshot_pop.limit(pop_negative_sample_size * POP_NEGATIVE_MULTIPLIER)
df_snapshot_pop = df_snapshot_pop.join(raw_data['item_content'], on=['content_id'], how='inner').select(['content_id', 'publish_time'])
df_pop_negative = generate_negative_samples(df_event=df_positive, df_content=df_snapshot_pop,
                                            major_col='userid', candidate_col='content_id',
                                            time_col='publish_time', sample_size=pop_negative_sample_size)
logger.info('[Data Preparation][Jollybuy Goods User2Item] pop negative samples')

# generate negative by all content pool
logger.info(f'[Data Preparation][Jollybuy Goods User2Item] Generate negative sample from content pool')

df_negative = generate_negative_samples(df_event=df_positive, df_content=df_content_pool,
                                        major_col='userid', candidate_col='content_id',
                                        time_col='publish_time', sample_size=content_negative_sample_size)

logger.info('[Data Preparation][Jollybuy Goods User2Item] content negative samples')
#df_negative.show()
logger.info('[Data Preparation][Jollybuy Goods User2Item] final negative sample')
df_negative_total = df_negative.unionByName(df_pop_negative)
# append profile and meta to negative samples
df_negative_total = df_negative_total.join(df_dedup_positive[user_profile_col], on=['userid'], how='left') \
                                     .drop(df_dedup_positive.content_id)
df_negative_total = df_negative_total.join(raw_data['user_meta'], on=['userid'], how='left')
df_negative_total = df_negative_total.join(raw_data['item_content'], on=['content_id'], how='inner')
df_negative_total = df_negative_total.join(raw_data['popularity'], on=['content_id', 'date', 'hour'], how='left')
df_negative_total.toPandas().head(2)
```

    [INFO] [Data Preparation][Jollybuy Goods User2Item] Generate negative sample from content pool


                                                                                    

    [INFO] [Data Preparation][Jollybuy Goods User2Item] content negative samples
    [INFO] [Data Preparation][Jollybuy Goods User2Item] final negative sample


                                                                                    

<div>
<style scoped>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userid</th>
      <th>content_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1010207000816002123</td>
      <td>P15252998327</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1010207000816002123</td>
      <td>P15252229989</td>
    </tr>
  </tbody>
</table>
</div>



# 正樣本 用y=1表示, 負樣本 用y=0表示 <a name="資料準備"></a>


```python
# 正樣本 用y=1表示, 負樣本 用y=0表示
logger.info('[Data Preparation][Jollybuy Goods User2Item] Integration positive and negative dataset')
df_positive = df_positive.withColumn('y', f.lit(1))
df_negative_total = df_negative_total.withColumn('y', f.lit(0))
dataset = df_positive.unionByName(df_negative_total)
dataset = dataset.select(getattr(config, 'FINAL_COLS', '*'))
dataset = dataset.toPandas()
logger.info(f"Len of positive sample: {len(dataset[dataset['y']==1])}")
logger.info(f"Len of negative sample: {len(dataset[dataset['y']==0])}")
logger.info(f"Len of dataset: {len(dataset)}")
dataset.head(2)
```

    [INFO] [Data Preparation][Jollybuy Goods User2Item] Integration positive and negative dataset


    [Stage 888:===================================================>   (14 + 1) / 15]

    [INFO] Len of positive sample: 225
    [INFO] Len of negative sample: 75
    [INFO] Len of dataset: 300


                                                                                    




<div>
<style scoped>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userid</th>
      <th>content_id</th>
      <th>age</th>
      <th>gender</th>
      <th>user_title_embedding</th>
      <th>user_category</th>
      <th>user_tag</th>
      <th>final_score</th>
      <th>tags</th>
      <th>item_title_embedding</th>
      <th>cat0</th>
      <th>cat1</th>
      <th>y</th>
      <th>publish_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1010008000817002909</td>
      <td>P05260422396</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>[]</td>
      <td>[-0.027697425335645676, 0.0201114434748888, 0....</td>
      <td>['Apple空機']</td>
      <td>['iPhone 14 Pro Max']</td>
      <td>1</td>
      <td>1675243451000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1010008000817002909</td>
      <td>P05260422396</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>[]</td>
      <td>[-0.027697425335645676, 0.0201114434748888, 0....</td>
      <td>['Apple空機']</td>
      <td>['iPhone 14 Pro Max']</td>
      <td>1</td>
      <td>1675243451000</td>
    </tr>
  </tbody>
</table>
</div>



# 收集 cat0, cat1 有哪些標籤 <a name="資料準備"></a>


```python
# 收集 cat0, cat1 有哪些標籤
distinct_content_parser = ContentDistinctValueParser()
col2label = distinct_content_parser.parse(raw_data['item_content'], config.COLS_TO_ENCODE['content'], add_ner=False)
col2label
```

    [INFO] [Data Preparation] Target columns: ['cat0', 'cat1', 'tags']
    [INFO] [Data Preparation] Process cat0


                                                                                    

    [INFO] [Data Preparation] Process cat1


                                                                                    

    [INFO] [Data Preparation] Process tags





    defaultdict(list,
                {'cat0': ['電玩遊戲', '手機平板與周邊', '動漫周邊', 'Apple空機', '出國網卡', '玩具公仔'],
                 'cat1': ['Sony PS4主機',
                  'iPhone 14 Pro Max',
                  '道具、飾品配件',
                  'iPhone 13/13 mini',
                  '可動玩偶',
                  '盒玩',
                  'iPhone保護殼',
                  '東南亞',
                  '任天堂周邊',
                  '同人妝品',
                  '18禁玩具',
                  '轉蛋',
                  '日本'],
                 'tags': []})




```python

```
