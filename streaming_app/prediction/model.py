"""Module to """

import os
import pathlib

import bentoml
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model

from streaming_app.prediction import resources


model_id = os.environ.get("MODEL_IDENTIFIER", "")

# TODO: Find a fast-fast NLP model that we can use for real-time tasks.
# model = bentoml.tensorflow.load_model(model_id)
model = load_model(pathlib.Path(resources.__file__).parent / "model.h5")

with open(pathlib.Path(resources.__file__).parent / "tokenizer.json", "r") as f_hdl:
    tokenizer = tokenizer_from_json(f_hdl.read())


"""
bentoml.pytorch.save_model(
    "demo_mnist",  # model name in the local model store
    trained_model,  # model instance being saved
    labels={  # user-defined labels for managing models in Yatai
        "owner": "nlp_team",
        "stage": "dev",
    },
    metadata={  # user-defined additional metadata
        "acc": acc,
        "cv_stats": cv_stats,
        "dataset_version": "20210820",
    },
    custom_objects={  # save additional user-defined python objects
        "tokenizer": tokenizer_object,
    },
)
"""


class Model:
    def __init__(self, model, preprocess, postprocess):
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def predict(self, inputs):
        inputs = self.preprocess(inputs)
        outputs = self.model.predict(inputs)
        return self.postprocess(outputs)
