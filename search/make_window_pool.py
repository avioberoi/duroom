"""Pool of named dated works in [LO, HI), at most CAP per artist, in the form of make_pools.py's pre-cutoff pool.
Usage: python make_window_pool.py OUT_NPZ LO HI [CAP=3]   (CPU; PYTHONPATH holding data.py and basis.py)"""
import sys

import numpy as np

from basis import capped
from data import CLIP, ids, keep, labels, lens, year

OUT, LO, HI = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 3
artist = labels['artist']
usable = (artist != '') & (year >= LO) & (year < HI)
rows = np.flatnonzero(usable)
full = np.flatnonzero(keep)
X = lens(CLIP).astype('float64')
X /= np.linalg.norm(X, axis=1, keepdims=True)
pool = np.sort(capped(rows, CAP, np.random.default_rng(0)))
s = lambda a: np.asarray([str(x) for x in a], dtype=str)    # noqa: E731
np.savez(OUT, rows=pool, rows_full=full[pool], ids=ids[pool], artist=s(artist[pool]), year=year[pool], vec=X[pool].astype('float32'))
print(f'[{LO}, {HI}): {len(rows)} named dated works, {len(np.unique(artist[rows]))} artists; cap {CAP} pool: {len(pool)} works, {len(np.unique(artist[pool]))} artists -> {OUT}', flush=True)
