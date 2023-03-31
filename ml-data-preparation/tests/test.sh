


gcloud dataproc jobs submit pyspark \
    tests/test_udfV.py \
    --cluster albert-test \
    --region us-west1 \
    --files datapreparation.zip 
