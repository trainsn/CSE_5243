import os
import numpy as np
from utils import *

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
D = np.zeros((M, N))

for i, sentence in enumerate(sentences):
    for word in sentence:
        if indexer.contains(word.lower()):
            D[i][indexer.index_of(word.lower())] += 1

rnd_idx = np.random.randint(M, size=5)
for i in range(rnd_idx.shape[0]):
    print(items[rnd_idx[i]].split('\t')[0])
    for j in range(N):
        if D[rnd_idx[i]][j] > 0:
            print("\t{:d}\t{}\t{}".format(j, indexer.get_object(j), D[rnd_idx[i]][j]))
