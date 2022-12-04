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
D = np.zeros((M, N), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for word in sentence:
        if indexer.contains(word.lower()):
            D[i][indexer.index_of(word.lower())] = True


simi = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        simi[i, j] = (D[i] * D[j]).sum() / (D[i] + D[j]).sum()
    if i % 100 == 0:
        print("finish calculating similarity for sentence {:d}".format(i))

np.save("exact_similarity.npy", simi)

pdb.set_trace()
