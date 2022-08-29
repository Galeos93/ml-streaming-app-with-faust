from bentoml import Model

from streaming_app.prediction import model


def test_load_model():
    model_instance = model.load_model()
    assert isinstance(model_instance, Model)
    assert model_instance.to_runner().run
    assert model_instance.custom_objects["tokenizer"]
