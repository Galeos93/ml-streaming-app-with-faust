"""Module to """

import os
import pathlib

import bentoml
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from streaming_app.prediction import resources


MODEL_ID = os.environ.get("MODEL_IDENTIFIER", "disaster_tweet_detector:latest")


def load_model():
    return bentoml.tensorflow.get(MODEL_ID)


with open(pathlib.Path(resources.__file__).parent / "tokenizer.json", "r") as f_hdl:
    tokenizer = tokenizer_from_json(f_hdl.read())


class Model:
    def __init__(self, runner, preprocess, postprocess):
        self.runner = runner
        self.preprocess = preprocess
        self.postprocess = postprocess

    def predict(self, inputs):
        inputs = self.preprocess(inputs)
        outputs = self.runner.run(inputs)
        return self.postprocess(outputs)
