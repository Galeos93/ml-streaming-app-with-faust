import time

import numpy as np
import pytest

from streaming_app.streaming import model_inference


def test_given_sentences_model_output_is_expected():
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


def test_disaster_classifier_speed(repetitions=100):
    times = []

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


@pytest.fixture()
def input_topic():
    # TODO: Check how to mock a topic
    pass


@pytest.fixture()
def output_topic():
    # TODO: Check how to mock a topic
    pass


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
    async def test_given_message_to_agent_output_topic_is_correct(test_app):

        message_value = (
            "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"
        )
        async with model_inference.inference_agent.test_context() as agent:
            model_input = model_inference.ModelInputRecord(value=message_value)
            # Sent model_input to the test agents local channel, and wait
            # the agent to process it.
            event = await agent.put(model_input)
            # Check that the agent returns the correct inference
            np.testing.assert_almost_equal(
                agent.results[event.message.offset], 0.98977554
            )

        @staticmethod
        def test_given_message_then_agent_outputs_inference_to_output_topic(
            monkeypatch, input_topic, output_topic
        ):
            monkeypatch.setattr(model_inference, "input_topic", input_topic)
            monkeypatch.setattr(model_inference, "output_topic", output_topic)
