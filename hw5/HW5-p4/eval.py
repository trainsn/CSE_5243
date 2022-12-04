import numpy as np
import matplotlib.pyplot as plt

exact_similarity = np.load("exact_similarity.npy")

fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(exact_similarity)
fig.colorbar(cax)
plt.title("Exact Jaccard Similarity")
plt.show()

k16minhash_estimated_similarity = np.load("k16-minhash_estimated_similarity.npy")

fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(k16minhash_estimated_similarity)
fig.colorbar(cax)
plt.title("16-MinHash Estimate of Similarity")
plt.show()

k128minhash_estimated_similarity = np.load("k128-minhash_estimated_similarity.npy")

fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(k128minhash_estimated_similarity)
fig.colorbar(cax)
plt.title("128-MinHash Estimate of Similarity")
plt.show()

k16_rmse = ((k16minhash_estimated_similarity - exact_similarity) ** 2).mean()
print("Mean Square Error for k=16 is {:.6f}".format(k16_rmse))
k128_rmse = ((k128minhash_estimated_similarity - exact_similarity) ** 2).mean()
print("Mean Square Error for k=128 is {:.6f}".format(k128_rmse))