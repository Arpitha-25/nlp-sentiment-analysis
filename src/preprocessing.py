import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

import nltk

resources = [
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
]

for resource_path, resource_name in resources:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name)


# Initialize stopwords
stop_words = set(stopwords.words("english"))

# Keep negation words for sentiment analysis
negation_words = {"not", "no", "nor", "never"}
stop_words = stop_words - negation_words

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def remove_html(text):
    """Remove HTML tags from text."""
    return re.sub(r"<.*?>", "", text)


def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_punctuation(text):
    """Remove punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text):
    """Remove numeric characters."""
    return re.sub(r"\d+", "", text)


def get_wordnet_pos(treebank_tag):
    """
    Convert NLTK POS tags to WordNet POS tags.
    """

    if treebank_tag.startswith("J"):
        return wordnet.ADJ

    elif treebank_tag.startswith("V"):
        return wordnet.VERB

    elif treebank_tag.startswith("N"):
        return wordnet.NOUN

    elif treebank_tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN


def remove_stopwords(text):
    """
    Remove stopwords while preserving negation words.
    """

    words = word_tokenize(text)

    filtered_words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)


def lemmatize_text(text):
    """
    Perform POS-aware lemmatization.
    """

    words = word_tokenize(text)

    tagged_words = pos_tag(words)

    lemmatized_words = []

    for word, tag in tagged_words:

        pos = get_wordnet_pos(tag)

        lemma = lemmatizer.lemmatize(word, pos)

        lemmatized_words.append(lemma)

    return " ".join(lemmatized_words)


def preprocess_text(text):
    """
    Complete preprocessing pipeline.
    """

    text = text.lower()

    text = remove_html(text)

    text = remove_urls(text)

    text = remove_punctuation(text)

    text = remove_numbers(text)

    text = remove_stopwords(text)

    text = lemmatize_text(text)

    return text.strip()