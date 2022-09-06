"""Model definition and handlers (such as loaders) are contained here."""

import os

import bentoml


MODEL_ID = os.environ.get("MODEL_IDENTIFIER", "disaster_tweet_detector:latest")


def load_model():
    """Loads the disaster tweet model, saved on the bento model storage.

    Notes
    -----
    The original model is a TensorFlow model.

    """
    return bentoml.tensorflow.get(MODEL_ID)


class Model:  # pylint: disable=too-few-public-methods
    """A Model adapts data, introduces it to a model, and processes the inferences."""

    def __init__(self, runner, preprocess, postprocess):
        self.runner = runner
        self.preprocess = preprocess
        self.postprocess = postprocess

    def predict(self, inputs):
        """Adds extra steps to the prediction.

        Notes
        -----
        `inputs` is adapted before they is is fed to the model. Finally, a
        postprocessing is applied to the model's predictions.

        """
        inputs = self.preprocess(inputs)
        outputs = self.runner.run(inputs)
        return self.postprocess(outputs)
