import os
import numpy as np
from utils import *
import nltk
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix

import pdb


class UnigramFeatureExtractor:
    """
    Extracts unigram bag-of-words features from a sentence.
    """
    def __init__(self, indexer: Indexer, train_exs, stop_words):
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

class ImportantFeatureExtractor:
    """
    Extracts important bag-of-words features from a sentence.
    """
    def __init__(self, indexer: Indexer, train_exs, stop_words, appear):
        for sentimentExample in train_exs:
            words = sentimentExample.words
            for word in words:
                lowercase = word.lower()
                if not lowercase in stop_words:
                    indexer.add_and_get_index(lowercase)

        important_indexer = Indexer()
        for i in range(len(indexer)):
            word = indexer.get_object(i)
            if indexer.count_of(word) >= appear:
                important_indexer.add_and_get_index(word)
        self.indexer = important_indexer
        self.corpus_length = len(important_indexer)
        print("corpus length is {:d}".format(self.corpus_length))

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

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, ex_words) -> int:
        """
        :param ex_words: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")

class KNearestNeighborClassifier(SentimentClassifier):
    def __init__(self, train_exs, feat_extractor, k):
        self.train_exs = train_exs
        self.feat_extractor = feat_extractor
        self.k = k

    def predict(self, sentence):
        feat = self.feat_extractor.calculate_sentence_probability(sentence).toarray()
        feat_norm = feat
        if abs(np.linalg.norm(feat)) > 1e-4:
            feat_norm = feat / np.linalg.norm(feat)
        similarities, labels = [], []
        for i in range(len(self.feat_extractor.feats)):
            tmp_feat = self.feat_extractor.feats[i].toarray().T
            tmp_feat_norm = np.linalg.norm(tmp_feat)
            similarities.append((feat_norm.dot(tmp_feat)[0, 0] / tmp_feat_norm) if abs(tmp_feat_norm) > 1e-4 else 0)
            labels.append(self.train_exs[i].label)
        similarities = np.array(similarities)
        labels = np.array(labels)

        knn = np.argpartition(-similarities, self.k)[:self.k]
        if labels[knn].sum() > self.k // 2:
            return 1
        else:
            return 0

class LogisticRegressionClassifier(SentimentClassifier):
    def __init__(self, feat_size, feat_extractor):
        self.w = np.zeros(feat_size)
        self.feat_extractor = feat_extractor

    def predict(self, sentence):
        feat = self.feat_extractor.calculate_sentence_probability(sentence)
        return int(feat.dot(np.expand_dims(self.w, axis=1))[0, 0] > 0)

def train_logistic_regression(train_exs, feat_extractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    lr = LogisticRegressionClassifier(feat_extractor.corpus_length, feat_extractor)
    alpha = 1e0
    # beta = 1e-4
    for epoch in range(8):
        loss = 0.
        acc = 0
        indices = np.arange(len(train_exs))
        np.random.shuffle(indices)
        for i in indices:
            feat = feat_extractor.feats[i]
            sentimentExample = train_exs[i]
            y = sentimentExample.label
            z = 1 / (1 + np.exp(-feat.dot(np.expand_dims(lr.w, axis=1))))[0, 0]
            loss += -y * np.log(z) - (1 - y) * np.log(1 - z) \
                    # + beta * np.expand_dims(lr.w, axis=0).dot(np.expand_dims(lr.w, axis=1))[0, 0]
            predict = int(feat.dot(np.expand_dims(lr.w, axis=1))[0, 0] > 0)
            acc += (predict == y)
            grad = (z - y) * feat.toarray()[0] # + 2 * beta * lr.w
            lr.w = lr.w - alpha * grad
        print("epoch {:d}, loss: {:f}, accuracy: {:f}".format(epoch, loss / len(train_exs), acc / len(train_exs)))

    for i in indices:
        feat = feat_extractor.feats[i]
        sentimentExample = train_exs[i]
        y = sentimentExample.label
        z = 1 / (1 + np.exp(-feat.dot(np.expand_dims(lr.w, axis=1))))[0, 0]
        loss += -y * np.log(z) - (1 - y) * np.log(1 - z)
    print("training loss: {:f}".format(loss / len(train_exs)))

    return lr

def train_model(args, train_exs):
    """
    Main entry point. Trains and returns one of several models depending on the args
    passed in from the main method.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    if args.feats == "UNIGRAM":
        feat_extractor = UnigramFeatureExtractor(Indexer(), train_exs, stop_words)
    elif args.feats == "IMPORTANT":
        feat_extractor = ImportantFeatureExtractor(Indexer(), train_exs, stop_words, args.appear)

    if args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    elif args.model == "KNN":
        model = KNearestNeighborClassifier(train_exs, feat_extractor, args.k)

    return model



