"""Configures a Faust agent that processes tweets from Kafka and returns inferences.

Notes
-----
In this module, the streaming application is created. The application
contains an agent, `inference_agent, that is subscribed to the topic `incoming_tweet`
and processes it, returning a prediction about whether or not the tweet mentions a
natural disaster. This prediction is broadcasted to the `tweet_disaster_inference`
topic.

Example
-------
If you have successfully setup Kafka and created an `incoming_tweet`, you
can start up an app's worker with the following command::

$ faust -A streaming_app worker -l info

To send and event to the `incoming_tweet` topic, you can do the following::

$ faust -A streaming_app send disaster_tweet_detector_input '{"value": "Hello world"}'

"""
import functools
import os
import typing

import faust
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.utils import (  # pylint: disable=import-error,no-name-in-module
    pad_sequences,
)

from streaming_app.prediction import model
from streaming_app.prediction import preprocessors
from streaming_app.prediction.preprocessors import (
    clean_tweet,
)

BROKER_URI = os.environ.get("BROKER_URI", "localhost:29092")

app = faust.App("model_prediction", broker=f"kafka://{BROKER_URI}", store="rocksdb://")
bento_model = model.load_model()
runnable_model = bento_model.to_runner()
runnable_model.init_local()
tokenizer = bento_model.custom_objects["tokenizer"]


class ModelInputRecord(faust.Record):  # pylint: disable=abstract-method
    """Record that defines the format of the data that the agent will receive.

    Notes
    -----
    The input of the model consists on tweets, formatted as str.

    """

    value: str


class ModelOutputRecord(faust.Record):  # pylint: disable=abstract-method
    """Record that defines the format of the data that the agent will return.

    Notes
    -----
    The inferences returned by the model will be formatted as list of floats.

    """

    value: typing.List[float]


input_topic = app.topic(
    "incoming_tweet",
    key_type=None,
    value_type=ModelInputRecord,
)
output_topic = app.topic(
    "tweet_disaster_inference",
    key_type=None,
    value_type=ModelOutputRecord,
)

_lemmatizer = functools.partial(
    preprocessors.lemmatize_text,
    lemmatizer=functools.partial(WordNetLemmatizer().lemmatize, pos="v"),
    filter_fun=lambda x: (len(x) > 3) and x not in preprocessors.STOPWORDS_PUNCTUATION,
)

_cnn_text_digestor = functools.partial(
    preprocessors.digest_sentences,
    tokenizer=tokenizer.texts_to_sequences,
    preprocessor=functools.partial(pad_sequences, maxlen=121, padding="post"),
)


disaster_classifier = model.Model(
    runner=runnable_model,
    preprocess=lambda x: tf.convert_to_tensor(
        _cnn_text_digestor([_lemmatizer(clean_tweet(sentence)) for sentence in x]),
        dtype=tf.float32,
    ),
    postprocess=lambda x: x,
)


@app.agent(input_topic, sink=[output_topic])
async def inference_agent(stream):
    """Agent that receives the model and input and outputs the inference."""
    async for model_input in stream:
        prediction = disaster_classifier.predict([model_input.value])
        prediction = [float(x[0]) for x in prediction]
        yield prediction
