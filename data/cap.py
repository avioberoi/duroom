"""Do the works sit in one cap of the sphere? Per lens, every work's angle to the average direction and how far out the farthest go, for the record and for all kept works.
Usage: cap.py OUT_DIR"""
import json
import os
import sys

import numpy as np

from data import labels, lens, title, ids, year, CLIP

OBS = os.environ['DUROOM_EMB']
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
record = (labels['artist'] != '') & (year >= 1000) & (year <= 2026)      # the same rule as distance.py and newaxis.py
Q = [0, 0.1, 1, 10, 50, 90, 99, 99.9, 100]
result = {'percentiles': Q}
for file in (CLIP, f'{OBS}/dinov3-l.npz', f'{OBS}/siglip-so400m.npz'):
    X = lens(file).astype('float64')
    assert np.isfinite(X).all() and (np.linalg.norm(X, axis=1) > 0.5).all(), file    # one bad row would turn every angle into NaN silently
    X /= np.linalg.norm(X, axis=1, keepdims=True)                         # stored vectors are unit length to 0.001; make it exact
    name, r = os.path.splitext(os.path.basename(file))[0], {'d': int(X.shape[1])}
    for set_name, rows in (('record', np.flatnonzero(record)), ('all', np.arange(len(X)))):
        m = X[rows].mean(0)
        cos = X[rows] @ (m / np.linalg.norm(m))
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        o = np.argsort(-ang)[:12]
        r[set_name] = {'n': int(len(rows)), 'R': float(np.linalg.norm(m)), 'share_along_mean': float((cos ** 2).mean()),
                       'angle_percentiles': np.percentile(ang, Q).round(2).tolist(),
                       'beyond_deg': {str(t): int((ang > t).sum()) for t in (60, 70, 80, 90)},
                       'random_sd_deg': float(np.degrees(1 / np.sqrt(X.shape[1]))),
                       'farthest': [{'id': int(ids[i]), 'title': title[i][:70], 'artist': labels['artist'][i], 'style': labels['style'][i],
                                     'year': None if np.isnan(year[i]) else int(year[i]), 'angle': round(float(a), 2)}
                                    for i, a in zip(rows[o], ang[o])]}
    result[name] = r
    print(name, {s: (r[s]['angle_percentiles'], r[s]['beyond_deg']) for s in ('record', 'all')}, flush=True)
    with open(f'{OUT}/cap.json', 'w') as f:      # after each lens, so a crash on the last keeps the first
        json.dump(result, f, indent=1)
