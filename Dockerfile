FROM python:3

WORKDIR .

ENV BROKER_URI="kafka-cluster:9092"

COPY streaming_app ./streaming_app
COPY bentoml /root/bentoml
COPY requirements.txt .
COPY Makefile .

# ENV: Set path to model
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
ENTRYPOINT bash


