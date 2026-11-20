"""Shared pieces: capped sampling per artist, a mean-and-directions basis, and what k directions keep."""
import numpy as np
import pandas as pd

from data import labels

artist = labels['artist']
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def capped(rows, cap, rng):
    """Shuffled rows, at most cap per artist."""
    rows = rng.permutation(rows)
    return rng.permutation(rows[pd.Series(artist[rows]).groupby(artist[rows]).cumcount().values < cap])    # reshuffled so truncating stays random


def code(Y):
    """The average, and directions sorted by spread, as rows. Same directions as an SVD of Y - mean."""
    mean = Y.mean(0)
    c = Y - mean
    return mean, np.ascontiguousarray(np.linalg.eigh(c.T @ c)[1][:, ::-1].T)    # contiguous: matmul is about 100x slower otherwise


def each_kept(Y, mean, V):
    c = Y - mean
    return np.cumsum((c @ V.T) ** 2, axis=1)[:, [k - 1 for k in KS]], (c ** 2).sum(1, keepdims=True)
