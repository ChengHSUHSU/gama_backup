

gcloud dataproc jobs submit pyspark \
    execution/user2item/train/planet_novel_book_din_training.py \
    --cluster albert-large-test \
    --region us-west1 \
    --files offline-model.zip \
    -- --project_id="bf-data-prod-001" \
    --dataset_blob_path="dataset/planet_novel_book/user2item/20230319" \
    --experiment_name= "e5231f154d154a5bb8c8d69ed4066069" \
    --run_date= "20230319"\
    --is_train \
    --download_dataset \
