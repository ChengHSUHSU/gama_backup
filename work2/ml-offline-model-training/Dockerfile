FROM python:3.8.5

RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN  apt-get -y update \
     && apt-get -y autoremove \
     && apt-get clean \
     && apt-get install -y zip

WORKDIR /src

COPY ./requirements.txt /src/requirements.txt
RUN pip3 install -r requirements.txt

ENV PYTHONPATH="$PYTHONPATH:/src"

COPY ./tests /src/tests
COPY ./bdds_recommendation/utils /src/bdds_recommendation/utils
COPY ./execution /src/execution
COPY ./bdds_recommendation/src /src/bdds_recommendation/src
COPY ./bdds_recommendation/src_v2 /src/bdds_recommendation/src_v2
COPY ./run.sh /src/run.sh
