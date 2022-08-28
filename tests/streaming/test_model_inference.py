import time

import numpy as np

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
    assert False
