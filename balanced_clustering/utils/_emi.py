# Authors: Robert Layton <robertlayton@gmail.com>
#           Corey Lynch <coreylynch9@gmail.com>
# License: BSD 3 clause

import numpy as np
import numba
from scipy.sparse import spmatrix
from math import exp, lgamma

@numba.njit(fastmath=True, cache=True, parallel=True)
def _emi(a, b, N):
    R = len(a)
    C = len(b)
    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.
    # While nijs[0] will never be used, having it simplifies the indexing.
    nijs = np.arange(0.0, float(max(np.max(a), np.max(b)) + 1))
    nijs[0] = 1  # Stops divide by zero warnings. As its not used, no issue.
    # term1 is nij / N
    term1 = nijs / N
    # term2 is log((N*nij) / (a * b)) == log(N * nij) - log(a * b)
    log_a = np.log(a)
    log_b = np.log(b)
    # term2 uses log(N * nij) = log(N) + log(nij)
    log_Nnij = np.log(N) + np.log(nijs)
    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    gln_a = []
    gln_Na = []
    for ai in a:
        gln_a.append(lgamma(ai + 1))
        gln_Na.append(lgamma(N - ai + 1))
    gln_b = []
    gln_Nb = []
    for bi in b:
        gln_b.append(lgamma(bi + 1))
        gln_Nb.append(lgamma(N - bi + 1))
    gln_N = lgamma(N + 1)
    gln_nij = [lgamma(nijs_i + 1) for nijs_i in nijs]
    # start and end values for nij terms for each summation.
    # start = np.array([[v - N + w for w in b] for v in a])
    start = np.zeros((R, C))
    end = np.zeros((R, C))
    for r in range(R):
        for c in range(C):
            start[r, c] = a[r] + b[c] - N
            end[r, c] = min(a[r], b[c]) + 1
    start = np.maximum(start, 1)
    # emi itself is a summation over the various values.
    emi = 0.0
    for i in range(R):
        for j in range(C):
            for nij in range(start[i, j], end[i,j]):
                term2 = log_Nnij[nij] - log_a[i] - log_b[j]
                # Numerators are positive, denominators are negative.
                gln = (gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j]
                     - gln_N - gln_nij[nij] - lgamma(a[i] - nij + 1)
                     - lgamma(b[j] - nij + 1)
                     - lgamma(N - a[i] - b[j] + nij + 1))
                term3 = exp(gln)
                emi += (term1[nij] * term2 * term3)
    return emi


def expected_mutual_information(contingency: spmatrix, n_samples: int):
    """Calculate the expected mutual information for two labelings."""
    N = n_samples
    a = np.ravel(contingency.sum(axis=1).astype(np.int32, copy=False))
    b = np.ravel(contingency.sum(axis=0).astype(np.int32, copy=False))
    return _emi(a, b, N)

