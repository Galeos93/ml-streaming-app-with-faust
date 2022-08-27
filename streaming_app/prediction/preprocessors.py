import re
import string
import typing

from nltk.corpus import stopwords
import numpy as np
from tensorflow.keras.utils import pad_sequences


STOPWORDS = list(stopwords.words("english"))
PUNCTUATION = list(string.punctuation)
STOPWORDS_PUNCTUATION = STOPWORDS + PUNCTUATION


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def clean_tweet(text: str) -> str:
    text = remove_emojis(text)
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)  # removing urls
    text = re.sub(
        "[^\w]", " ", text
    )  # remove embedded special characters in words (for example #earthquake)
    text = re.sub("[\d]", "", text)  # this will remove numeric characters
    text = text.lower()
    return text


def lemmatize_text(
    text: str,
    lemmatizer: typing.Callable[[str], typing.List[str]],
    filter_fun: typing.Callable[[str], str],
) -> str:
    words = text.split()
    lemmatized_words = [lemmatizer(w) for w in words]
    sentence = " ".join([w for w in filter(filter_fun, lemmatized_words)])
    return sentence


def digest_sentences(
    sentences: typing.List[str],
    tokenizer: typing.Callable[[typing.List[str]], typing.List[int]],
    preprocess: typing.Callable[[typing.List[typing.List[int]]], np.ndarray],
):
    text_sequences = tokenizer(sentences)
    model_input = preprocess(text_sequences)
    return model_input
