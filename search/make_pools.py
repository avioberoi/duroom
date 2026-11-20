"""Writes pre{CUT}_cap{CAP}.npz (named dated works before CUT, at most CAP per artist, with unit CLIP vectors) and starts_pre{CUT}.npz (all of them).
Usage: python make_pools.py OUT_DIR [CUT=1900] [CAP=3]   (CPU; run next to data.py and basis.py)"""
import os
import sys

import numpy as np

from basis import capped
from data import CLIP, ids, keep, labels, lens, year

full = np.flatnonzero(keep)    # row in the unfiltered observer file for each filtered row (search_v2 indexes the unfiltered file)

OUT = sys.argv[1]
CUT = int(sys.argv[2]) if len(sys.argv) > 2 else 1900
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 3
os.makedirs(OUT, exist_ok=True)
artist = labels['artist']
usable = (artist != '') & (year >= 1000) & (year <= 2026)
past = np.flatnonzero(usable & (year < CUT))
X = lens(CLIP).astype('float64')
X /= np.linalg.norm(X, axis=1, keepdims=True)
pool = np.sort(capped(past, CAP, np.random.default_rng(0)))
s = lambda a: np.asarray([str(x) for x in a], dtype=str)    # noqa: E731  # plain string arrays: no pickle across numpy versions
np.savez(f'{OUT}/pre{CUT}_cap{CAP}.npz', rows=pool, rows_full=full[pool], ids=ids[pool], artist=s(artist[pool]), year=year[pool], vec=X[pool].astype('float32'))
np.savez(f'{OUT}/starts_pre{CUT}.npz', rows=past, rows_full=full[past], ids=ids[past], artist=s(artist[past]), year=year[past],
         mean_pre=X[past].mean(0), mean_record=X[np.flatnonzero(usable)].mean(0))    # unit-vector means (not renormalised)
print(f'before {CUT}: {len(past)} named dated works, {len(np.unique(artist[past]))} artists; cap {CAP} pool: {len(pool)} works, '
      f'{len(np.unique(artist[pool]))} artists; median raw-length rule is in the config, vectors here are unit', flush=True)
