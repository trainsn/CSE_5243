import os
import numpy as np
from utils import *
import nltk
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix

import pdb

class UnigramFeatureExtractor:
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer, train_exs, stop_words):
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        for sentimentExample in train_exs:
            words = sentimentExample.words
            for word in words:
                lowercase = word.lower()
                if not lowercase in stop_words:
                    indexer.add_and_get_index(lowercase)
        self.indexer = indexer
        self.corpus_length = len(indexer)

        self.feats = []
        for i, sentimentExample in enumerate(train_exs):
            sentence = sentimentExample.words
            self.feats.append(self.calculate_sentence_probability(sentence))


    def calculate_sentence_probability(self, sentence):
        col = [self.indexer.index_of(word.lower()) for word in sentence if self.indexer.contains(word.lower())]
        row = np.zeros(len(col), dtype=np.int)
        data = np.ones(len(col), dtype=np.int)
        feat = csr_matrix((data, (row, col)), shape=(1, self.corpus_length))
        if len(col) > 0:
            feat = feat * (1. / len(col))
        return feat

def train_model(train_exs):
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    feat_extractor = UnigramFeatureExtractor(Indexer(), train_exs, stop_words)



