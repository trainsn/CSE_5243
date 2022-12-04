import pdb

import numpy as np
import random
import time

# Record the maximum shingle ID that we assigned.
maxShingleID = 2 ** 32 - 1

# We need the next largest prime number above 'maxShingleID'.
# I looked this value up here:
# http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
nextPrime = 4294967311

# Our random hash function will take the form of:
#   h(x) = (a*x + b) % c
# Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
# a prime number just greater than maxShingleID.

# Generate a list of 'k' random coefficients for the random hash functions,
# while ensuring that the same value does not appear multiple times in the
# list.
def pickRandomCoeffs(k):
    # Create a list of 'k' random values.
    randList = []

    while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID)

        # Ensure that each random number is unique.
        while randIndex in randList:
            randIndex = random.randint(0, maxShingleID)

            # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1

    return randList

# This is the number of components in the resulting MinHash signatures.
# Correspondingly, it is also the number of random hash functions that
# we will need in order to calculate the MinHash.
def MinHashSimilarity(D, numHashes = 16):
    print('Generating random hash functions...')

    # For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.
    coeffA = pickRandomCoeffs(numHashes)
    coeffB = pickRandomCoeffs(numHashes)

    print('Generating MinHash signatures for all documents...')

    # List of documents represented as signature vectors
    M, N = D.shape
    signatures = np.zeros((M, numHashes))

    for docID in range(M):

        # Get the shingle set for this document.
        shingleIDSet = np.nonzero(D[docID])[0]

        # For each of the random hash functions...
        for i in range(0, numHashes):

            # For each of the shingles actually in the document, calculate its hash code
            # using hash function 'i'.

            # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
            # the maximum possible value output by the hash.
            minHashCode = nextPrime + 1

            # For each shingle in the document...
            for j in range(shingleIDSet.shape[0]):
                # Evaluate the hash function.
                shingleID = shingleIDSet[j]
                hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime

                # Track the lowest hash code seen.
                if hashCode < minHashCode:
                    minHashCode = hashCode

            # Add the smallest hash code value as component number 'i' of the signature.
            signatures[docID][i] = minHashCode

        if docID % 1000 == 0:
            print("Finish generating MinHash signatures for sentence {:d}".format(docID))

    simi = np.zeros((M, M))

    # Time this step.
    t0 = time.time()

    # For each of the test documents...
    for i in range(0, M):
        # Get the MinHash signature for document i.
        signature1 = signatures[i]

        # For each of the other test documents...
        for j in range(i + 1, M):
            # Get the MinHash signature for document j.
            signature2 = signatures[j]

            simi[i][j] = (signature1 == signature2).sum() / numHashes

        if i % 100 == 0:
            print("Finish calculating similarity for sentence {:d}".format(i))

    # Calculate the elapsed time (in seconds)
    elapsed = (time.time() - t0)

    print("Generating {:d}-MinHashEstimate of Similarity took {:.2f} sec".format(numHashes, elapsed))

    np.save("k{:d}-minhash_estimated_similarity.npy".format(numHashes), simi)
