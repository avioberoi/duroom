"""Leftover share of a reading under a reference: the part of its squared distance from the reference mean that the K leading directions do not describe (0 = in their span, 1 = orthogonal). Float64.
Usage: import leftover_fn as lf; lf.fit, lf.share, lf.crossfit, lf.null, lf.z"""
import numpy as np


def unit(E):
    E = np.atleast_2d(np.asarray(E, dtype=np.float64))
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def fit(E, K):
    """Mean and K leading directions (rows, eigh of the centred scatter, largest spread first) of the unit readings E."""
    X = unit(E)
    mean = X.mean(0)
    c = X - mean
    return mean, np.ascontiguousarray(np.linalg.eigh(c.T @ c)[1][:, ::-1][:, :K].T)


def share(E, mean, V):
    c = unit(E) - mean
    r = c - (c @ V.T) @ V
    return (r ** 2).sum(1) / (c ** 2).sum(1)


def crossfit(E, artist, K, folds=5, seed=0):
    """Artist-out cross-fit null: artists, permuted by default_rng(seed), go to fold j % folds; each fold is scored on a basis fitted on the others. Returns (shares, fold per row)."""
    artist = np.array([str(a) for a in artist])
    arts = np.random.default_rng(seed).permutation(np.unique(artist))
    fold_of = {a: j % folds for j, a in enumerate(arts)}
    fold = np.array([fold_of[a] for a in artist])
    L = np.full(len(artist), np.nan)
    for f in range(folds):
        test = fold == f
        L[test] = share(E[test], *fit(E[~test], K))
    return L, fold


def null(L):
    """Median, unscaled MAD and 99th percentile of the null shares."""
    med = float(np.median(L))
    return med, float(np.median(np.abs(L - med))), float(np.percentile(L, 99))


def z(L, med, mad):
    """(L - median) / (MAD + 1e-8), the told value on a fixed scale."""
    return (np.asarray(L, dtype=np.float64) - med) / (mad + 1e-8)
