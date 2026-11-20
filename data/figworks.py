"""Typical works along the decade path: for every decade with at least 50 works, its mean point in (shift, axis 1, axis 2) from newaxis.py's loadings at cutoff 1900, and the N nearest works (at most one per artist) as cards with thumbnails.
Usage: figworks.py LOADS_NPZ NEWAXIS_JSON OUT_DIR [N=3]"""
import json
import os
import sys

import numpy as np

from data import card, labels, year

artist = labels['artist']

LOADS, NEWAXIS, OUT = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 3
z = np.load(LOADS)
rows, yr = z['rows'], z['year']
assert np.array_equal(year[rows], yr), 'npz rows are not the same works as data.py'
P = np.stack([z['shift'], z['axis1'], z['axis2']], 1).astype('float64')
dec = np.floor(yr / 10) * 10
run = {d[0]: (np.array(d[1:4]), d[4]) for d in json.load(open(NEWAXIS, encoding='utf8'))['cutoffs']['1900']['by_decade_plane']}

full = {float(d) for d, c in zip(*np.unique(dec, return_counts=True)) if c >= 50}
assert full == set(map(float, run)), 'decade sets differ'
out = []
for d in sorted(run):
    m = dec == d
    mean = P[m].mean(0)
    assert m.sum() == run[d][1] and np.abs(mean - run[d][0]).max() < 1e-6, (d, mean, run[d])    # gate: same works, same loadings as the run
    idx = np.flatnonzero(m)
    order = idx[np.argsort(np.linalg.norm(P[idx] - mean, axis=1))]
    picks, seen = [], set()
    for j in order:
        if artist[rows[j]] not in seen:
            seen.add(artist[rows[j]])
            picks.append({**card(rows[j], OUT), 'coords': np.round(P[j], 4).tolist(), 'dist_to_mean': round(float(np.linalg.norm(P[j] - mean)), 4)})
        if len(picks) == N:
            break
    out.append({'decade': int(d), 'n': int(m.sum()), 'mean': np.round(mean, 4).tolist(), 'works': picks})
os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/figworks.json', 'w') as f:
    json.dump({'source': [LOADS, NEWAXIS], 'axes': ['shift', 'axis1', 'axis2'], 'n_per_decade': N, 'decades': out}, f, indent=1)
print('done', len(out), 'decades', flush=True)
