import functools

import faust
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.utils import pad_sequences

from streaming_app.prediction import model, Model
from streaming_app.prediction import preprocessors
from streaming_app.prediction.model import tokenizer, model_id
from streaming_app.prediction.preprocessors import (
    clean_tweet,
)

app = faust.App("model_prediction", broker="kafka://", store="rocksdb://")


class ModelInputRecord(faust.Record):
    name: str
    value: str


class ModelOutputRecord(faust.Record):
    name: str
    value: float


input_topic = app.topic(
    f"{model_id}_input",
    key_type=bytes,
    value_type=ModelInputRecord,
)
output_topic = app.topic(
    f"{model_id}_output",
    key_type=bytes,
    value_type=ModelOutputRecord,
)

"""
encoded_docs = tokenizer.texts_to_sequences(X_train.tolist())

# embedding layer require all the encoded sequences to be of the same length, lets take max lenght as 100
# and apply padding on the sequences which are of lower size

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


"""


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


disaster_classifier = Model(
    model=model,
    preprocess=lambda x: _cnn_text_digestor(
        [_lemmatizer(clean_tweet(sentence)) for sentence in x]
    ),
    postprocess=lambda x: x,
)


@app.agent(input_topic, sink=[output_topic])
async def inference_agent(stream):
    """Agent that receives the model and input and outputs the inference."""
    async for _, value in stream.items():
        prediction = disaster_classifier.predict(value)
        yield prediction


if __name__ == "__main__":
    app.main()
