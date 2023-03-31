apt-get update && apt-get install -y procps
gcloud auth activate-service-account $1 --key-file GCP_SA_KEY.json
export GOOGLE_APPLICATION_CREDENTIALS=GCP_SA_KEY.json
gcloud config set account $1
gcloud config set project $2

zip -r datapreparation *
gsutil cp datapreparation.zip gs://mainfile-$2/offline-model-training/
gsutil cp execution/* gs://mainfile-$2/offline-model-training/execution/
