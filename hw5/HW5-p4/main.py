import os
import numpy as np
from utils import *
from exact_similarity import ExactSimilarity
from minhash_similarity import MinHashSimilarity

import pdb

items = []
fp = open(os.path.join("sentiment labelled sentences", "amazon_cells_labelled.txt"))
items.extend(fp.readlines())
fp = open(os.path.join("sentiment labelled sentences", "imdb_labelled.txt"))
items.extend(fp.readlines())
fp = open(os.path.join("sentiment labelled sentences", "yelp_labelled.txt"))
items.extend(fp.readlines())

indexer = Indexer()
sentences = []
for item in items:
    sentence = item.split('\t')[0]
    sentence = sentence.replace(',', ' ')
    sentence = sentence.replace('!', ' ')
    sentence = sentence.replace('.', ' ')
    sentence = sentence.replace(';', ' ')
    sentence = sentence.replace('(', ' ')
    sentence = sentence.replace(')', ' ')
    sentence = sentence.replace('-', ' ')
    sentence = sentence.replace('\"', ' ')
    sentence = sentence.replace('/', ' ')
    sentences.append(sentence.split(' '))
    for word in sentence.split(' '):
        if word != "":
            indexer.add_and_get_index(word.lower())

M, N = len(sentences), len(indexer)
D = np.zeros((M, N), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for word in sentence:
        if indexer.contains(word.lower()):
            D[i][indexer.index_of(word.lower())] = True

ExactSimilarity(D)
MinHashSimilarity(D, 16)
MinHashSimilarity(D, 128)
