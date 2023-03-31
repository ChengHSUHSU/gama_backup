class BaseConfig():
    BASELINE_MODEL_PATH = './ml-model'
    PERFORMANCE_IMPROVEMENT_RATE_POSTFIX = '_pir'   # PIR means performance improvement rate


class BaseItem2itemConfig(BaseConfig):
    SERVICE_TYPE = 'item2item'
    NDCG_GROUPBY_KEY = 'page_uuid'


class BaseUser2itemConfig(BaseConfig):
    SERVICE_TYPE = 'user2item'
    NDCG_GROUPBY_KEY = 'userid'
