Project Name: ml-data-preparation
======================

Contents
--------
*	[1. 文件資訊(History)](#1-文件資訊history-)   
    *	[1.1 文件制／修訂履歷(Table of Changes)](#文件制／修訂履歷)  
    *	[1.2 文件存放位置(Document Location)](#文件存放位置)  
    *	[1.3 文件發佈資訊(Distribution)](#文件發佈資訊)  
*	[2. 架構設計(Architecture Design)](#2-架構設計architecture-design-)  		
    *	[2.1 Add the Preparation Pipeline](#add-the-preparation-pipeline)
*	[3. 程式相關(Programming)](#3-程式相關programming-)	
    *	[3.1 Getting Started](#getting-started)

* * *
# 1. 文件資訊(History) <a name="文件資訊"></a>

> ## 1.1 文件制／修訂履歷(Table of Changes) <a name="文件制／修訂履歷"></a>
>
> > Modification History
>

|    Data    |    Author   | Description |
|:----------:|:-----------:|:-----------:|
| 2022/01/24 | Sean Chen |     初版    |


> ## 1.2 文件存放位置(Document Location) <a name="文件存放位置"></a>
> 路徑 : 
> - [data-jarvis/ml-data-preparation](https://gitlab.beango.com/datascience/data-jarvis/ml-data-preparation)
> - [[News][planet] user2item recommendation system](https://docs.google.com/document/d/10Jqr3ZYwGv4OZ5QRcHo5QGOkOmU02zbqsrDXMtqSedg/edit)
> - [ML 推薦系統開發規範](https://docs.google.com/document/d/1ZFJ4sH98tlu7tbBVGp_VYzP-eQEqWvnuOeHQ2y2tPQ0/edit#)
> ## 1.3 文件發佈資訊(Distribution) <a name="文件發佈資訊"></a>
> 本文件提供以下的人員閱讀:
>
> * Data & Science Team


* * *

# 2. 架構設計(Architecture Design) <a name="架構設計"></a>  

![Architecture][logo]

[logo]: docs/pics/ml-planet-recommendation-architecture.png "Architecture"

* * *

# 3. 程式相關(Programming) <a name="程式相關"></a>

> 資料準備是從 datalake 透過 pyspark 進行原始資料的撈取包括：
>  content , event, pipeline data … etc
>  最後落下一份 dataset 以提供後續模型訓練
>  
>  此階段不具邏輯處理，僅透過一些基礎數據分析進行特徵選曲(feature selection)
>  目的是從眾多的資料欄位中提取出適合的原始資料出來


## Add the Preparation Pipeline


### 1. Complete Excution Mainfile (Required)

**`excution/prepare_<SERVICE>_<SERVICE_TYPE>_dataset.py`**

For example:

-  execution/prepare_planet_news_user2item_dataset.py
-  execution/prepare_planet_news_item2item_dataset.py
-  execution/prepare_nownews_news_item2item_dataset.py

### 2. Complete GCS Reader (Required)

**`src/gcsreader/<SERVICE>_<SERVICE_TYPE>.py`**

For example:

-  src/gcsreader/nownews_news_item2item.py
-  src/gcsreader/planet_news_user2item.py


### 3. Add Service Setting (Optional)

**`src/options/<SERVICE>_options.py`**

For example:

-  src/options/nownews_news_option.py
-  src/options/planet_news_options.py



## Getting Started
### Submit Training Job on Dataproc

1. Package Data-preparation project

```
zip -r datapreparation *
```

2. Execute training process

```sh
EXPERIMENT_NAME = <EXPERIMENT_NAME>
PREPARATION_MAINFILE = <PREPARATION_MAINFILE>
CONTENT_TYPE = <CONTENT_TYPE>
CONTENT_PROPERTY = <CONTENT_PROPERTY>
INPUT_DATE = <INPUT_DATE>

gcloud dataproc jobs submit pyspark \
    execution/"$PREPARATION_MAINFILE" \
    --cluster <CLUSTER_NAME> \
    --region <REGION> \
    --files datapreparation.zip \
    -- --project_id='bf-data-uat-001' \
    --name="$EXPERIMENT_NAME" \
    --run_time="$INPUT_DATE" \
    --content_type="$CONTENT_TYPE" \
    --content_property="$CONTENT_PROPERTY" \
    --checkpoints_dir='checkpoints' \
    --days=30 \
    --negative_sample_size=20 \
    --save \
    --upload_gcs
```
