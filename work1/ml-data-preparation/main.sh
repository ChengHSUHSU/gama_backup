gcloud dataproc jobs submit pyspark \
    execution/prepare_show_user2item_dataset.py \
    --cluster albert-large-test \
    --region us-west1 \
    --files datapreparation.zip \
    -- --project_id='bf-data-prod-001' \
    --name="jb_exp_test" \
    --experiment_name='test' \
    --run_time="20230312" \
    --content_type="show" \
    --content_property="beanfun" \
    --item_content_type='others'\
    --checkpoints_dir='checkpoints' \
    --days=30 \
    --negative_sample_size=20 \
    --save \
    --upload_gcs


