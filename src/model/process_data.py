
import sys
import nltk
from collections import Counter, defaultdict
from stop_list import closed_class_stop_words
from nltk.tokenize import word_tokenize
import math
import string
import argparse
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet

nltk.download('punkt')

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import string

nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def tokenize(text):
    punctuation_set = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text.lower())

    pos_tags = pos_tag(tokens)
    cleaned_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos) or wordnet.NOUN)  # Default to NOUN if POS is None
        for token, pos in pos_tags
        if token.isalpha() and token not in closed_class_stop_words and token not in punctuation_set
    ]

    return cleaned_tokens

def get_tf(word_list):
    term_count = Counter(word_list)
    total_terms = len(word_list)
    tf_scores = {term: count / total_terms for term, count in term_count.items()}
    return tf_scores

def get_idf(documents):
    total_docs = len(documents)
    doc_freq = Counter()
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            doc_freq[term] += 1
    idf_scores = {term: math.log(total_docs / (freq + 1)) for term, freq in doc_freq.items()}
    return idf_scores

def calculate_tfidf(word_list, idf_scores):
    tf_scores = get_tf(word_list)
    tfidf_scores = {term: tf_scores[term] * idf_scores.get(term, 0) for term in tf_scores}
    return tfidf_scores