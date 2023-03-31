

gcloud dataproc jobs submit pyspark \
    execution/hot/train/jollybuy_goods_hot_training.py \
    --cluster albert-test \
    --region us-west1 \
    --files offline-model.zip \
    -- --project_id='bf-data-uat-001' \
    --dataset_blob_path='dataset/jollybuy_goods/hot/2023030606' \
    --experiment_name= 'test' \
    --is_train \
    --download_dataset \
    --save  \
    --upload_gcs \
    --deploy \
    --monitoring \
    --ndcg='5,10,20'

