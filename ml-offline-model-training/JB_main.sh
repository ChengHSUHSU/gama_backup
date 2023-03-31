


gcloud dataproc jobs submit pyspark \
    execution/user2item/train/jollybuy_goods_din_training.py \
    --cluster albert-test \
    --region us-west1 \
    --files offline-model.zip \
    -- --project_id='bf-data-uat-001' \
    --dataset_blob_path='dataset/jollybuy_goods/user2item/20220821' \
    --experiment_name= '57dc4d3f4ad149a7b025d02c53205c3e' \
    --is_train \
    --download_dataset \
    --save  \
    --upload_gcs \
    --deploy \
    --monitoring \
    --ndcg='5,10,20'


