import numpy as np
import time

def ExactSimilarity(D):
    # Time this step.
    t0 = time.time()

    M = D.shape[0]
    simi = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            simi[i, j] = (D[i] * D[j]).sum() / (D[i] + D[j]).sum()
        if i % 100 == 0:
            print("Finish calculating similarity for sentence {:d}".format(i))

    # Calculate the elapsed time (in seconds)
    elapsed = (time.time() - t0)

    print("Generating Exact Similarity took {:.2f} sec".format(elapsed))

    np.save("exact_similarity.npy", simi)
