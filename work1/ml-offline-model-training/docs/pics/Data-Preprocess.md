# 資料前處理(Data Preprocess) <a name="資料前處理"></a>

> 這份扣存在的目的，主要是 之後再模型優化時，可能會需要用pandas dataframe 比較細緻地看 資料前處理的過程
> 因此，我們特別把 資料前處理的物件 JollybuyGoodsUser2ItemDINPreprocessor ，給搬出來給大家看


```python
# 為了要在 JupyterLab cell上 開發 
# 首先，先增加 WORK/ml-offline-model-training 的路徑，這樣才能 import ml-offline-model-training 裡面的物件
import sys
sys.path.insert(0, 'WORK/ml-offline-model-training/')
```


```python
import pandas as pd
from utils.logger import logger
from src.preprocess.utils import process_age
from src.options.train_options import TrainDINOptions
from src_v2.preprocess import BaseUser2ItemPreprocessor
from utils import download_blob, dump_pickle, read_pickle
from src_v2.configs.jollybuy_goods.jollybuy_goods_user2item import JollybuyGoodsUser2ItemDINConfig
```

    /opt/conda/miniconda3/lib/python3.8/site-packages/scipy/__init__.py:138: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.24.2)
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion} is required for this version of "


# 參數設定 <a name="訓練模型, 模型評估"></a>


```python
# 參數設定
config = JollybuyGoodsUser2ItemDINConfig
opt = TrainDINOptions().parse()
kwargs_ = vars(opt)
bucket_storage = f'machine-learning-models-{opt.project_id}'
base_path = f'{opt.checkpoints_dir}/{opt.experiment_name}'
```

    ------------ Options -------------
    api_model_version_url: http://10.128.33.3:5000/api/model_descript
    att_activation: Dice
    att_weight_normalization: False
    batch_size: 64
    checkpoints_dir: checkpoints
    dataroot: ./dataset
    dataset: dataset.pickle
    dataset_blob_path: dataset/jollybuy_goods/user2item/20230219
    deploy: False
    dnn_activation: relu
    dnn_dropout: 0
    dnn_use_bn: False
    download_dataset: False
    epochs: 1
    experiment_name: 6098dcad6dad432db043aeda6bab9b91
    gpu_ids: 0
    init_std: 0.0001
    is_train: False
    l2_reg_dnn: 0
    l2_reg_embedding: 1e-06
    logger_name: 
    mail_server: smtp.gmail.com
    mail_server_account: bfdataprod002@gmail.com
    mail_server_password: davjnuouxzwpiqus
    mail_server_port: 587
    metrics: ['binary_crossentropy']
    mlflow_experiment_id: None
    mlflow_experiment_run_name: None
    mlflow_host: http://10.32.192.39:5000
    monitoring: False
    ndcg: 5,10,20
    num_folds: 5
    objective_function: binary_crossentropy
    optimizer: adagrad
    project_id: bf-data-prod-001
    save: False
    seed: 1024
    task: binary
    verbose: False
    -------------- End ----------------



```python
class JollybuyGoodsUser2ItemDINPreprocessor(BaseUser2ItemPreprocessor):

    def __init__(self, configs, col2label2idx={}, logger=logger, mode='train', **kwargs):

        super(JollybuyGoodsUser2ItemDINPreprocessor, self).__init__(configs, col2label2idx, logger, mode)

    def process(self, dataset=None, user_data=None, realtime_user_data=None, requisite_cols=[], **kwargs):

        if self.mode == 'inference':
            dataset = self.append_user_data(dataset, user_data, realtime_user_data)
            dataset = self.append_metrics_data(dataset)

        # convert data type
        dataset = self.convert_data_type(dataset, **kwargs)

        # convert columns name
        dataset = self.convert_columns_name(dataset, **kwargs)

        # special encoding to age feature
        dataset = self.process_age(dataset, **kwargs)

        # handle missing data
        dataset = self.handle_missing_data(dataset, **kwargs)

        # aggregate preference
        dataset = self.aggregate_preference(dataset, **kwargs)

        # process categorical features
        dataset = self.encode_features(dataset, **kwargs)

        # process user behavior sequence
        dataset = self.process_user_behavior_sequence(dataset, **kwargs)

        if requisite_cols:
            return dataset[requisite_cols]
        elif self.configs.REQUISITE_COLS:
            return dataset[self.configs.REQUISITE_COLS]

        return dataset

    def process_age(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---process age feature---')

        dataset = process_age(dataset, columns='age', age_range=20)

        return dataset

    def append_metrics_data(self, dataset: pd.DataFrame) -> pd.DataFrame:

        if self.verbose:
            self.logger.info(f'---process metrics feature---')

        dataset['final_score'] = dataset['statistics'].apply(lambda x: x.get('last_7_day', {}).get('popularity_score', 0.0) if bool(x) else 0.0)

        return dataset
```

# 下載之前 data-preparation 所得到的 dataset.pickle <a name="訓練模型, 模型評估"></a>


```python
# 下載之前 data-preparation 所得到的 dataset.pickle
# 參考資料 : data-preparation: https://gitlab.data.gamania.com/datascience/data-jarvis/ml-data-preparation
file_to_download = [opt.dataset]

for file in file_to_download:
    logger.info(f'[Jollybuy Goods User2Item][Data Preprocessing] Download {file} from the bucket')
    download_blob(bucket_name=bucket_storage,
                  source_blob_name=f'{opt.dataset_blob_path}/{base_path}/{file}',
                  destination_file_name=f'{opt.dataroot}/{file}')

dataset = read_pickle(fn='dataset.pickle', base_path=opt.dataroot)
logger.info('[Jollybuy Goods User2Item][Data Preprocessing] Logging dataset information')
dataset.head(2)
```

    [INFO] [Jollybuy Goods User2Item][Data Preprocessing] Download dataset.pickle from the bucket
    Downloaded storage object dataset/jollybuy_goods/user2item/20230219/checkpoints/6098dcad6dad432db043aeda6bab9b91/dataset.pickle from bucket machine-learning-models-bf-data-prod-001 to local file ./dataset/dataset.pickle.
    [INFO] [Jollybuy Goods User2Item][Data Preprocessing] Logging dataset information





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
      <td>1018830700816002410</td>
      <td>P04233492545</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>[]</td>
      <td>[0.008073552511632442, -0.007438497617840767, ...</td>
      <td>['戶外與運動用品']</td>
      <td>['旅行相關配件']</td>
      <td>1</td>
      <td>1580405131000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2300927389120598016</td>
      <td>P04233492545</td>
      <td>None</td>
      <td>m</td>
      <td>[-0.011687994003295898, -0.005800542887300253,...</td>
      <td>{"click": {"cat1": {"生活家電": {"cnt": 1, "pref":...</td>
      <td>None</td>
      <td>None</td>
      <td>[]</td>
      <td>[0.008073552511632442, -0.007438497617840767, ...</td>
      <td>['戶外與運動用品']</td>
      <td>['旅行相關配件']</td>
      <td>1</td>
      <td>1580405131000</td>
    </tr>
  </tbody>
</table>
</div>




```python
preprocessor = JollybuyGoodsUser2ItemDINPreprocessor(configs=config,
                                                     col2label2idx=dict(),
                                                     mode='train',
                                                     **kwargs_)
```


```python
dataset = preprocessor.process(dataset=dataset,
                               requisite_cols=config.REQUISITE_COLS,
                               **kwargs_)
```

    [INFO] ---convert data type---
    [INFO] mode: ast - column: cat0
    [INFO] mode: ast - column: cat1
    [INFO] mode: json - column: user_title_embedding
    [INFO] mode: json - column: item_title_embedding
    [INFO] mode: float - column: final_score
    [INFO] ---convert columns name---
    [INFO] ---process age feature---
    [INFO] ---handle missing data---
    [INFO] col: gender - value: UNKNOWN
    [INFO] col: cat0 - value: ['PAD']
    [INFO] col: cat1 - value: ['PAD']
    [INFO] col: final_score - value: 0.0
    [INFO] col: user_title_embedding - value: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    [INFO] col: item_title_embedding - value: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    [INFO] col: user_category - value: {}
    [INFO] ---aggregate preference---
    [INFO] category pref score: {'level': ['click', 'cat0'], 'cat_col': 'user_category', 'score_col': 'cat0_pref_score'}
    [INFO] category pref score: {'level': ['click', 'cat1'], 'cat_col': 'user_category', 'score_col': 'cat1_pref_score'}
    [INFO] tag pref score: {'tag_entity_list': ['others'], 'user_tag_col': 'user_tag', 'item_tag_col': 'tags', 'tagging_type': 'editor', 'score_col': 'tag_other_pref_score'}
    [INFO] ---process categorical data---
    [INFO] column: age - mode: LabelEncoding
    [INFO] column: gender - mode: LabelEncoding
    [INFO] column: tags - mode: LabelEncoding
    [INFO] column: cat0 - mode: LabelEncoding
    [INFO] column: cat1 - mode: LabelEncoding
    [INFO] ---process user behavior sequence---
    [INFO] col: user_category - params: ['cat0', 'click', 'cat0', 'hist_', 'seq_length_']
    [INFO] col: user_category - params: ['cat1', 'click', 'cat1', 'hist_', 'seq_length_']



```python
dataset.head(2)
```




<div>
<style scoped>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>cat0</th>
      <th>cat1</th>
      <th>final_score</th>
      <th>user_title_embedding</th>
      <th>item_title_embedding</th>
      <th>cat0_pref_score</th>
      <th>cat1_pref_score</th>
      <th>tag_other_pref_score</th>
      <th>hist_cat0</th>
      <th>seq_length_cat0</th>
      <th>hist_cat1</th>
      <th>seq_length_cat1</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0]</td>
      <td>[0]</td>
      <td>[20]</td>
      <td>[192]</td>
      <td>0.0</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>[0.008073552511632442, -0.007438497617840767, ...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>0</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0]</td>
      <td>[1]</td>
      <td>[20]</td>
      <td>[192]</td>
      <td>0.0</td>
      <td>[-0.011687994003295898, -0.005800542887300253,...</td>
      <td>[0.008073552511632442, -0.007438497617840767, ...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>[28, 11, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>2</td>
      <td>[199, 264, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```
