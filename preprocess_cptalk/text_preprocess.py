import re
import os
import json
import unicodedata
from collections import Counter

import spacy
from bs4 import BeautifulSoup
from textblob import TextBlob, Word
from textblob.sentiments import NaiveBayesAnalyzer
from spacy.lang.en.stop_words import STOP_WORDS as stop_words

# Load contractions.json using a path relative to THIS file's location
file_path = os.path.join(os.path.dirname(__file__), 'data', 'contractions.json')
with open(file_path) as f:
    contractions = json.load(f)

# Load the spaCy English model once, reused by every function that needs it
nlp = spacy.load('en_core_web_sm')


# =======================================================================
# PART A: FEATURE EXTRACTION
# =======================================================================

def word_count(x):
    return len(x.split())


def character_count(x):
    return len(re.sub(r'\s', '', x))


def avg_word_length(x):
    return character_count(x) / word_count(x)


def stop_words_count(x):
    return len([word for word in x.lower().split() if word in stop_words])


def hashtags_count(x):
    return len(re.findall(r'#\w+', x))


def mentions_count(x):
    return len(re.findall(r'@\w+', x))


def numerics_count(x):
    return len(re.findall(r'\b\d+\b', x))


def uppercase_count(x):
    return len([word for word in x.split() if word.isupper()])


# =======================================================================
# PART B: PRE-PROCESSING AND CLEANING
# =======================================================================

def to_lower_case(x):
    return x.lower()


def contraction_to_expansion(x):
    return ' '.join([contractions.get(word.lower(), word) for word in x.split()])


EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'


def count_emails(x):
    return len(re.findall(EMAIL_PATTERN, x))


def remove_emails(x):
    return re.sub(EMAIL_PATTERN, '', x)


URL_PATTERN = r'(?:http|https)\S+|www\.\S+'


def count_urls(x):
    return len(re.findall(URL_PATTERN, x))


def remove_urls(x):
    return re.sub(URL_PATTERN, '', x)


RETWEET_PATTERN = r'\bRT\s@\w+'


def count_retweets(x):
    return len(re.findall(RETWEET_PATTERN, x))


def remove_retweets(x):
    return re.sub(RETWEET_PATTERN, '', x)


def remove_html_tags(x):
    return BeautifulSoup(x, 'lxml').get_text()


def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x


MENTION_PATTERN = r'@\w+'


def remove_mentions(x):
    return re.sub(MENTION_PATTERN, '', x)


SPECIAL_CHAR_PATTERN = r'[^\w\s]'


def remove_special_chars(x):
    return re.sub(SPECIAL_CHAR_PATTERN, '', x)


REPEATED_CHAR_PATTERN = r'(.)\1+'


def remove_repeated_chars(x):
    return re.sub(REPEATED_CHAR_PATTERN, r'\1\1', x)


def remove_stopwords(x):
    return ' '.join([word for word in x.split() if word not in stop_words])


def lemmatize_noun_verb(x):
    doc = nlp(x)
    tokens = []
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB']:
            tokens.append(token.lemma_)
        else:
            tokens.append(token.text)
    x = ' '.join(tokens)
    x = re.sub(r'\s+\.', '.', x)
    return x


def lemmatize(x):
    doc = nlp(x)
    return ' '.join([token.lemma_ for token in doc])


def remove_common_words(x, common_words):
    return ' '.join([word for word in x.split() if word not in common_words])


def remove_rare_words(x, rare_words):
    return ' '.join([word for word in x.split() if word not in rare_words])


def correct_spelling(x):
    words = x.split()
    corrected = []
    for word in words:
        corrected.append(str(Word(word).correct()))
    return ' '.join(corrected)


def get_noun_phrase(x):
    blob = TextBlob(x)
    return list(blob.noun_phrases)


def ngram(x, n=2):
    blob = TextBlob(x)
    return list(blob.ngrams(n))


def singularize_words(x):
    blob = TextBlob(x)
    result = []
    for word, tag in blob.tags:
        if tag == 'NNS':
            result.append(word.singularize())
        else:
            result.append(word)
    return ' '.join(result)


def pluralize_words(x):
    blob = TextBlob(x)
    result = []
    for word, tag in blob.tags:
        if tag == 'NN':
            result.append(word.pluralize())
        else:
            result.append(word)
    return ' '.join(result)


def sentiment_analysis(x):
    return TextBlob(x, analyzer=NaiveBayesAnalyzer()).sentiment.classification


def detect_language(x):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source='auto', target='en').translate(x)


def translate(x, destination='en'):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source='auto', target=destination).translate(x)