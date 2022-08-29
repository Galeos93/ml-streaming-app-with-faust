import functools
from lib2to3.pgen2 import token

import faust
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences

from streaming_app.prediction import model
from streaming_app.prediction import preprocessors
from streaming_app.prediction.preprocessors import (
    clean_tweet,
)

app = faust.App("model_prediction", broker="kafka://", store="rocksdb://")
bento_model = model.load_model()
runnable_model = bento_model.to_runner()
runnable_model.init_local()
tokenizer = bento_model.custom_objects["tokenizer"]


class ModelInputRecord(faust.Record):
    name: str
    value: str


class ModelOutputRecord(faust.Record):
    name: str
    value: float


input_topic = app.topic(
    f"{model.MODEL_ID}_input",
    key_type=bytes,
    value_type=ModelInputRecord,
)
output_topic = app.topic(
    f"{model.MODEL_ID}_output",
    key_type=bytes,
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
    async for _, value in stream.items():
        prediction = disaster_classifier.predict([value])
        yield prediction


if __name__ == "__main__":
    app.main()
