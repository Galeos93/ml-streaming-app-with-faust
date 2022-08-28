from distutils.command.clean import clean
import functools
from turtle import clear

from nltk.stem import WordNetLemmatizer
import numpy as np
import pytest
from tensorflow.keras.preprocessing.text import Tokenizer

from streaming_app.prediction import preprocessors


def test_given_tweet_emoji_is_removed():
    tweet = "There was an explosion in here \U0001f62c"
    expected_clean_tweet = "There was an explosion in here "
    clean_tweet = preprocessors.remove_emojis(tweet)
    assert clean_tweet == expected_clean_tweet


class TestCleanTweet:
    @staticmethod
    @pytest.mark.parametrize(
        "tweet,expected_clean_tweet",
        [
            ("Check out this website https://t.co/foo.", "check out this website  "),
            ("Check out this website http://t.co/bar.", "check out this website  "),
            (
                "Everything around me was moving. #earthquake",
                "everything around me was moving   earthquake",
            ),
            (
                "It is 8:00 am. Good morning!!!",
                "it is   am  good morning   ",
            ),
        ],
    )
    def test_given_tweet_output_is_expected(tweet, expected_clean_tweet):
        clean_tweet = preprocessors.clean_tweet(tweet)
        assert clean_tweet == expected_clean_tweet


class TestLemmatizeText:
    @staticmethod
    @pytest.mark.parametrize(
        "tweet,expected_clean_tweet",
        [
            ("check out this website  ", "check website"),
            (
                "everything around me was moving   earthquake",
                "everything around move earthquake",
            ),
            (
                "it is   am  good morning   ",
                "good morning",
            ),
        ],
    )
    def test_given_sentence_output_is_expected(tweet, expected_clean_tweet):
        clean_tweet = preprocessors.lemmatize_text(
            tweet,
            lemmatizer=functools.partial(WordNetLemmatizer().lemmatize, pos="v"),
            filter_fun=lambda x: (len(x) > 3)
            and x not in preprocessors.STOPWORDS_PUNCTUATION,
        )
        assert clean_tweet == expected_clean_tweet


@pytest.fixture()
def all_ones_tokenizer():
    def inner_fun(sentences):
        return [[1] * len(x.split(" ")) for x in sentences]

    return inner_fun


@pytest.fixture()
def keras_tokenizer():
    lines = ["hi how are you doing I am fine, how about you Here it is 8:00 am"]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer.texts_to_sequences


class TestDigestSentences:
    @staticmethod
    @pytest.mark.parametrize(
        "tokenizer,expected_output",
        [
            (
                "all_ones_tokenizer",
                np.array(
                    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=object
                ),
            ),
            (
                "keras_tokenizer",
                np.array(
                    [[4, 1, 5, 2, 6], [7, 3, 8, 1, 9, 2], [10, 11, 12, 13, 14, 3]],
                    dtype=object,
                ),
            ),
        ],
    )
    def test_given_sentences_output_array_is_correct(
        tokenizer, expected_output, request
    ):

        sentences = [
            "hi how are you doing",
            "I am fine, how about you",
            "Here it is 8:00 am",
        ]
        tokenizer = request.getfixturevalue(tokenizer)
        model_input = preprocessors.digest_sentences(
            sentences,
            tokenizer=tokenizer,
            preprocessor=functools.partial(np.array, dtype=object),
        )
        np.testing.assert_array_equal(
            model_input,
            expected_output,
        )
