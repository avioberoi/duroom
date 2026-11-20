"""Is the record's path through the decades more one-directional than chance? Decade-mean path: share of variance on the first direction, straightness and persistence, against random-walk, step-order and no-time nulls, with an artist-block bootstrap.
Usage: path_null.py OUT_DIR [LENS name, default clip-vit-l-14]"""
import json
import os
import sys

import numpy as np

from data import labels, lens, year

NAME = sys.argv[2] if len(sys.argv) > 2 else 'clip-vit-l-14'
OUT = f'{sys.argv[1]}/{NAME}'
X = lens(f'{os.environ["DUROOM_EMB"]}/{NAME}.npz').astype('float64')
X /= np.linalg.norm(X, axis=1, keepdims=True)
os.makedirs(OUT, exist_ok=True)
usable = (labels['artist'] != '') & (year >= 1000) & (year <= 2026)
decade = np.where(usable, np.floor(np.nan_to_num(year) / 10) * 10, np.nan)
WINDOWS = {'1900-2010': (1900, 2010), '1900-2020': (1900, 2020), '1780-1890': (1780, 1890), '1660-1770': (1660, 1770)}
DRAWS, BOOT = 10000, 200


def stats(P):
    S = np.diff(P, axis=0)
    c = P - P.mean(0)
    ev = np.linalg.svd(c, compute_uv=False) ** 2    # the same eigenvalues as the covariance, without forming a d x d matrix
    L = np.linalg.norm(S, axis=1)
    cos = (S[1:] * S[:-1]).sum(1) / (L[1:] * L[:-1])
    return {'share1': float(ev[0] / ev.sum()), 'share2': float(ev[1] / ev.sum()), 'straightness': float(np.linalg.norm(P[-1] - P[0]) / L.sum()), 'persistence': float(cos.mean())}


def walk(lengths, rng, pool=None):
    """Steps of the real lengths; directions are differences of two random pool works (the data's own spread), or isotropic without a pool."""
    if pool is None:
        D = rng.standard_normal((len(lengths), X.shape[1]))
    else:
        D = X[rng.choice(pool, len(lengths))] - X[rng.choice(pool, len(lengths))]
        while (bad := np.linalg.norm(D, axis=1) < 1e-9).any():    # the same work drawn twice, or a duplicate vector
            D[bad] = X[rng.choice(pool, bad.sum())] - X[rng.choice(pool, bad.sum())]
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    return np.vstack([np.zeros(X.shape[1]), np.cumsum(D * lengths[:, None], axis=0)])


def means(groups):
    return np.stack([X[g].mean(0) for g in groups])


def summary(real, null, skip=()):
    return {k: {'null_mean': float(np.mean([n[k] for n in null])), 'null_p95': float(np.percentile([n[k] for n in null], 95)),
                'p_at_least_real': float((1 + sum(n[k] >= real[k] for n in null)) / (len(null) + 1))} for k in real if k not in skip}


artist = labels['artist']
result = {'lens': NAME, 'draws': DRAWS, 'boot': BOOT, 'windows': {}}
for wname, (a, b) in WINDOWS.items():
    decs = [d for d in range(a, b + 1, 10) if (decade == d).sum() >= 50]
    if len(decs) < 6:    # too few points for a path
        result['windows'][wname] = {'skipped': f'{len(decs)} decades with 50 or more works'}
        continue
    groups = [np.flatnonzero(decade == d) for d in decs]
    P = means(groups)
    real = stats(P)
    rng = np.random.default_rng([a, b])
    lengths = np.linalg.norm(np.diff(P, axis=0), axis=1)
    pool, sizes = np.concatenate(groups), [len(g) for g in groups]
    null_walk = [stats(walk(lengths, rng, pool)) for _ in range(DRAWS)]
    null_iso = [stats(walk(lengths, rng)) for _ in range(DRAWS)]
    S = np.diff(P, axis=0)
    null_order = [stats(np.vstack([P[0], P[0] + np.cumsum(S[rng.permutation(len(S))], axis=0)])) for _ in range(DRAWS)]
    # Works cluster by artist, so the resampling units are an artist's works within a decade, not single works
    blocks = [[g[artist[g] == n] for n in np.unique(artist[g])] for g in groups]
    flat, counts = [x for bl in blocks for x in bl], [len(bl) for bl in blocks]
    null_time, steps_time = [], []
    for _ in range(BOOT):
        order = rng.permutation(len(flat))
        parts = np.split(order, np.cumsum(counts)[:-1])    # the same number of artist blocks per decade, assigned at random
        Q = means([np.concatenate([flat[i] for i in part]) for part in parts])
        null_time.append(stats(Q))
        steps_time.append(float(np.linalg.norm(np.diff(Q, axis=0), axis=1).mean()))
    boot = [stats(means([np.concatenate([bl[i] for i in rng.integers(0, len(bl), len(bl))]) for bl in blocks])) for _ in range(BOOT)]
    result['windows'][wname] = {
        'decades': decs, 'works': sizes, 'artists': counts, 'step_lengths': lengths.tolist(), 'real': real,
        'real_interval_95_artist_bootstrap': {k: np.percentile([x[k] for x in boot], [2.5, 97.5]).tolist() for k in real},
        'null_random_walk_data_shaped': summary(real, null_walk), 'null_random_walk_isotropic': summary(real, null_iso),
        'null_step_order': summary(real, null_order, skip=('straightness',)),    # straightness does not change when steps are reordered
        'null_no_time': summary(real, null_time),
        'step_length_mean': float(lengths.mean()), 'no_time_step_length': {'mean': float(np.mean(steps_time)), 'p95': float(np.percentile(steps_time, 95))}}
    json.dump(result, open(f'{OUT}/path_null.json', 'w'), indent=1)
    w = result['windows'][wname]
    print(NAME, wname, len(decs), 'decades; real', {k: round(v, 3) for k, v in real.items()},
          '| p walk (data-shaped)', {k: round(v['p_at_least_real'], 4) for k, v in w['null_random_walk_data_shaped'].items()},
          '| p step order', {k: round(v['p_at_least_real'], 4) for k, v in w['null_step_order'].items()},
          '| p no time', {k: round(v['p_at_least_real'], 4) for k, v in w['null_no_time'].items()},
          '| step', round(w['step_length_mean'], 4), 'no-time step', round(w['no_time_step_length']['mean'], 4), flush=True)
print('done', flush=True)
