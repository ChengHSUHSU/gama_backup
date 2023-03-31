Project Name: ml-offline-model-training
======================

Contents
--------
*	[1. 文件資訊(History)](#1-文件資訊history-)   
	* [1.1 文件制／修訂履歷(Table of Changes)](#文件制／修訂履歷)  
	* [1.2 文件存放位置(Document Location)](#文件存放位置)  
	* [1.3 文件發佈資訊(Distribution)](#文件發佈資訊)  
*	[2. 架構設計(Architecture Design)](#2-架構設計architecture-design-)  		
*	[3. 程式相關(Programming)](#3-程式相關programming-)
    * [3.1 Feature Column Naming Rule](#feature-column-naming-rule)	
    * [3.2 Getting Started](#getting-started)	

* * *
# 1. 文件資訊(History) <a name="文件資訊"></a>

> ## 1.1 文件制／修訂履歷(Table of Changes) <a name="文件制／修訂履歷"></a>
>
> > Modification History
>

|    Data    |    Author   | Description |
|:----------:|:-----------:|:-----------:|
| 2021/07/07 | Sean Chen |     建立新聞推薦模型(xgboost)訓練流程     |
| 2021/08/16 | Sean Chen |     建立小說推薦模型(DIN)訓練流程，checkpoint與handler架構設計，      |
| 2021/08/21 | Sean Chen |     建立 CI Pipeline      |
| 2021/08/24 | Sean Chen |     建立模型預測流程，小說推薦 xgboost 版本      |
| 2021/08/25 | Sean Chen |     建立前處理測試、修訂文件      |
| 2021/08/26 | Andy Chen |     建立影音推薦模型(xgboost)訓練流程    |
| 2021/09/30 | Andy Chen |     更新星球影音輸入特徵   |
| 2021/10/08 | Andy Chen |     建立有閑推薦模型(xgboost)訓練流程    |
| 2021/11/18 | Sean Chen |     建立 MLOps workflow pipeline    |
| 2022/01/24 | Sean Chen |     新增特徵命名規範    |


> ## 1.2 文件存放位置(Document Location) <a name="文件存放位置"></a>
> 路徑 : 
> - [data-jarvis/ml-planet-recommendation](https://gitlab.beango.com/datascience/data-jarvis/ml-planet-recommendation)
> - [[News][planet] user2item recommendation system](https://docs.google.com/document/d/10Jqr3ZYwGv4OZ5QRcHo5QGOkOmU02zbqsrDXMtqSedg/edit)
> - [ML 推薦系統開發規範](https://docs.google.com/document/d/1ZFJ4sH98tlu7tbBVGp_VYzP-eQEqWvnuOeHQ2y2tPQ0/edit#)
> - [Refactoring ML Model Training](https://docs.google.com/document/d/1i43w0QEkLCFn8kjIws77LATJK4IAcmchsmiOooTT8u8/edit)
> ## 1.3 文件發佈資訊(Distribution) <a name="文件發佈資訊"></a>
> 本文件提供以下的人員閱讀:
>
> * Data & Science Team


* * *

# 2. 架構設計(Architecture Design) <a name="架構設計"></a>  

![Architecture][logo]

[logo]: docs/pics/ml-planet-recommendation-architecture_v2.png "Architecture"

* * *

# 3. 程式相關(Programming) <a name="程式相關"></a>

**Training Step**
1. Problem define (input, output) (regression or classification)
2. Get user/item interaction data and any feature you need from GCS.
   - Negative sampling (size:10)
3. Feature extraction – prepare model input
     - One-hot	(e.g. category) `[0,0,0,1,0]`
     - Multi-hot	(e.g. tags) `[1,0,0,1,1]`
     - Numerical 	(e.g. age) `[23]`
4. Model training (train / validation / test)
5. Evaluation (metrics: precision, recall, f1-score, AUC)



## Feature Column Naming Rule


Feature column naming rule follow: 

`<prefix>_<feature_name>_<feature_type>`

**prefix** (optional)

> **page**_tag  
> **click**_tag  
> **click**_tag_dot  


**feature_name** (required)

> **tag**  
> click_**tag**  
> **tag**_dot  

**feature_type** (optional)

- dot product : 		dot
- cosin similarity : 	cos
- sum :			sum
- embedding : 		emb


> user_title_**emb**  
> click_tag_**cos**  
> tag_**dot**  




## Getting Started
### Training on dataproc

1. Run [data preparation](https://gitlab.beango.com/datascience/data-jarvis/ml-data-preparation) process to get raw dataset.

2. Package offline training project

```
zip -r offline-model src utils
```

3. Execute training process

```sh
EXPERIMENT_NAME = <here_is_your_experiment_name>
TRAINING_MAINFILE = <here_is_your_training_mainfile>

gcloud dataproc jobs submit pyspark \
    execution/train/"$TRAINING_MAINFILE" \
    --cluster <CLUSTER_NAME> \
    --region <REGION> \
    --files offline-model.zip \
    -- --project_id='bf-data-uat-001' \
    --name="$EXPERIMENT_NAME" \
    --save \
    --is_train \
    --upload_gcs \
    --download_dataset
```

### Training Job on JupyterLab

1. Move ML-offline-model-training project to JupyterLab.

2. Follow the example below to run model training.

> - [Data Preprocess](docs/pics/Data-Preprocess.md)
> - [Complete Excution Mainfile](docs/pics/Complete-Excution-Mainfile2.md)
