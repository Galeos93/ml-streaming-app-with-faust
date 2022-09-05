import os

import pytest
import pathlib

import bentoml

from tests import resources


@pytest.fixture()
def model():
    os.environ["BENTOML_HOME"] = str(pathlib.Path(resources.__file__).parent)
    bento_model = bentoml.tensorflow.load_model(
        "disaster_tweet_detector:h7o2qrrhysi6cme4"
    )
    return bento_model
