#!/usr/bin/env python3
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.linalg as linalg



def main(seed = 1425, n = 1000, m = 30, num_nonzero=15):
    np.random.seed(seed)

    # create 30 exponentially distributed features for 1000 samples
    lambdas = np.random.uniform(low=0.001, high=5, size=m)

    X = np.array([np.random.exponential(scale=l, size=n) for l in lambdas]).T

    # half of the features don't have an effect on y
    non_zeros = np.random.choice(m, num_nonzero)
    betas = np.zeros(m)
    
    # some normally distributed betas with mean 0
    betas[non_zeros] = np.random.normal(loc=0, scale=5, size=num_nonzero)

    # add some noise
    Y = X@betas + np.random.normal(scale=0.2, size=n)

    data = {"Y": Y, "X": X, "betas": betas, "n": n, "m": m, "num_nonzero": num_nonzero}
    np.save("art_data", data)


if __name__ == "__main__":
    main()
