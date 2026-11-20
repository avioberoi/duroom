"""Leftover residual, cosine similarity and top-k helpers used by onoff_v3.py (loaded by path)."""
import numpy as np


def unit(X):
    X = np.atleast_2d(np.asarray(X, np.float64))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def resid(E, mean, V):
    """Leftover vector: the part of a unit reading's offset from the reference average that the K directions miss (leftover_fn.share = |r|^2 / |c|^2)."""
    c = unit(E) - mean
    return c - (c @ V.T) @ V


def sims(A, B):
    return unit(A) @ unit(B).T


def topk(S, k):
    i = np.argsort(-S, axis=1, kind='stable')[:, :k]
    return np.take_along_axis(S, i, 1), i
