from distutils.command.clean import clean
import functools
from turtle import clear

from nltk.stem import WordNetLemmatizer
import numpy as np
import pytest

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


class TestDigestSentences:
    @staticmethod
    def test_given_sentences_output_array_is_correct():
        def _tokenizer(sentences):
            return [[1] * len(x.split(" ")) for x in sentences]

        sentences = [
            "hi how are you doing",
            "I am fine, how about you",
            "Here it is 8:00 am",
        ]
        model_input = preprocessors.digest_sentences(
            sentences, tokenizer=_tokenizer, preprocess=np.array
        )
        np.testing.assert_array_equal(
            model_input,
            np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        )
