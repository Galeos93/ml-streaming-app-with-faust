import time
from unittest.mock import Mock, MagicMock

import numpy as np
import pytest

from streaming_app.streaming import model_inference


@pytest.fixture()
def bento_model(model, monkeypatch):
    monkeypatch.setattr(model_inference, "bento_model", model)


def test_given_sentences_model_output_is_expected(bento_model):
    output = model_inference.disaster_classifier.predict(
        [
            "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all",
            "Forest fire near La Ronge Sask. Canada",
            (
                "All residents asked to 'shelter in place' are being notified"
                " by officers. No other evacuation or shelter in place orders"
                " are expected"
            ),
        ]
    )
    assert output.shape == (3, 1)
    np.testing.assert_almost_equal(
        np.array([[0.98977554], [0.9858864], [0.99459094]]), output
    )


def test_disaster_classifier_speed(bento_model):
    times = []
    repetitions = 100

    # First inference is discarded.
    model_inference.disaster_classifier.predict(
        [
            "All residents asked to 'shelter in place' are being notified"
            " by officers. No other evacuation or shelter in place orders"
            " are expected",
        ]
    )

    for _ in range(repetitions):
        start = time.time()
        model_inference.disaster_classifier.predict(
            [
                "All residents asked to 'shelter in place' are being notified"
                " by officers. No other evacuation or shelter in place orders"
                " are expected",
            ]
        )
        elapsed = time.time() - start
        times.append(elapsed)

    print(f"Mean {np.mean(times)*100} ms; STD {np.std(times)*100}")
    print(f"99.9 percentile {np.percentile(times, q=99.0)*100} ms")


@pytest.fixture()
def output_topic(return_value=None, **kwargs):
    return MagicMock()


@pytest.fixture()
def test_app(event_loop):
    """passing in event_loop helps avoid 'attached to a different loop' error"""
    model_inference.app.finalize()
    model_inference.app.conf.store = "memory://"
    model_inference.app.flow_control.resume()
    return model_inference.app


class TestInferenceAgent:
    @staticmethod
    @pytest.mark.asyncio()
    async def test_given_message_to_agent_output_topic_is_correct(
        test_app, output_topic, bento_model
    ):
        message_value = (
            "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"
        )
        async with model_inference.inference_agent.test_context(
            sink=[output_topic]
        ) as agent:
            model_input = model_inference.ModelInputRecord(value=message_value)
            # Sent model_input to the test agents local channel, and wait
            # the agent to process it.
            event = await agent.put(model_input)
            # Check that the agent returns the correct inference
            np.testing.assert_almost_equal(
                agent.results[event.message.offset], [0.98977554]
            )

    @staticmethod
    @pytest.mark.asyncio()
    async def test_given_message_then_agent_outputs_inference_to_output_topic(
        test_app, output_topic, bento_model
    ):
        message_value = (
            "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"
        )
        async with model_inference.inference_agent.test_context(
            sink=[output_topic]
        ) as agent:
            model_input = model_inference.ModelInputRecord(value=message_value)
            await agent.put(model_input)
            np.testing.assert_almost_equal(
                [0.98977554], output_topic.call_args_list[0][0][0]
            )
