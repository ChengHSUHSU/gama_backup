

gcloud dataproc jobs submit pyspark \
    execution/user2item/train/planet_novel_book_din_trainingV22.py \
    --cluster albert-large-test2 \
    --region us-west1 \
    --files offline-model.zip \
    -- --project_id="bf-data-prod-001" \
    --dataset_blob_path="dataset/planet_novel_book/user2item/20230312" \
    --experiment_name= "d5a78f6cc3ba46f68effde2bf33ae93c" \
    --run_date= "20230312"\
    --is_train \
    --download_dataset \


machine-learning-models-bf-data-prod-001/dataset/planet_novel_book/user2item/20230312/checkpoints/024ccf71fe4a4114a43c4027e4681971