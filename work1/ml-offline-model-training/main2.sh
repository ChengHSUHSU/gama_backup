

gcloud dataproc jobs submit pyspark \
    execution/user2item/train/show_din_training.py \
    --cluster albert-large-test2 \
    --region us-west1 \
    --files offline-model.zip \
    -- --project_id="bf-data-prod-001" \
    --dataset_blob_path="dataset/show/user2item/20230311" \
    --experiment_name= "d5a78f6cc3ba46f68effde2bf33ae93c" \
    --run_date= "20230311"\
    --is_train \
    --download_dataset \
