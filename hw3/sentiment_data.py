import os
import numpy as np
from typing import List

import pdb

class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (str): list of words
        label (int): 0 or 1 (0 = negative, 1 = positive)
    """

    def __init__(self, words, label):
        self.words = words
        self.label = label

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()

def read_sentiment_examples():
    tmps = []
    fp = open(os.path.join("sentiment labelled sentences", "amazon_cells_labelled.txt"))
    tmps.extend(fp.readlines())
    fp = open(os.path.join("sentiment labelled sentences", "imdb_labelled.txt"))
    tmps.extend(fp.readlines())
    fp = open(os.path.join("sentiment labelled sentences", "yelp_labelled.txt"))
    tmps.extend(fp.readlines())
    fp.close()

    items = []
    for line in tmps:
        fields = line.split('\t')
        sentence = fields[0]
        sentence = sentence.replace(',', ' ')
        sentence = sentence.replace('!', ' ')
        sentence = sentence.replace('.', ' ')
        sentence = sentence.replace(';', ' ')
        sentence = sentence.replace('(', ' ')
        sentence = sentence.replace(')', ' ')
        sentence = sentence.replace('-', ' ')
        sentence = sentence.replace('\"', ' ')
        sentence = sentence.replace('/', ' ')
        tokenized_cleaned_sent = list(filter(lambda x: x != '', sentence.rstrip().split(" ")))
        label = 0 if "0" in fields[1] else 1
        items.append(SentimentExample(tokenized_cleaned_sent, label))

    num_items = len(items)
    num_train_items, num_dev_items = int(num_items * 0.7), int(num_items * 0.15)
    rnd_idx = np.random.permutation(num_items)
    train_items, dev_items, test_items = [], [], []
    for i in range(num_items):
        if i < num_train_items:
            train_items.append(items[rnd_idx[i]])
        elif i < num_train_items + num_dev_items:
            dev_items.append(items[rnd_idx[i]])
        else:
            test_items.append(items[rnd_idx[i]])

    return train_items, dev_items, test_items
