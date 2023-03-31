


gcloud dataproc jobs submit pyspark \
    execution/hot/train/planet_news_hot_training.py \
    --cluster albert-large-test \
    --region us-west1 \
    --files offline-model.zip \
    -- --project_id='bf-data-prod-001' \
    --dataset_blob_path='dataset/planet_news/hot/2023031004' \
    --experiment_name= 'test' \
    --is_train \
    --download_dataset \
    --save  \
    --upload_gcs \
    --deploy \
    --monitoring \
    --ndcg='5,10,20'

