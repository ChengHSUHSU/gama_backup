


gcloud dataproc jobs submit pyspark \
    execution/prepare_jollybuy_goods_hot_model_dataset.py \
    --cluster albert-test \
    --region us-west1 \
    --files datapreparation.zip \
    -- --project_id='bf-data-uat-001' \
    --name="jb_exp_test" \
    --experiment_name='test' \
    --run_time="" \
    --content_type="jollybuy_goods" \
    --content_property="beanfun" \
    --item_content_type='others'\
    --checkpoints_dir='checkpoints' \
    --days=30 \
    --negative_sample_size=20 \
    --save \
    --upload_gcs

