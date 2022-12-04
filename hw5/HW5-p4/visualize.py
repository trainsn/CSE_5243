import numpy as np
import matplotlib.pyplot as plt

simi = np.load("exact_similarity.npy")

fig, ax = plt.subplots(figsize=(13, 13))
cax = ax.matshow(simi)
fig.colorbar(cax)
plt.title("Exact Jaccard Similarity")
plt.show()
