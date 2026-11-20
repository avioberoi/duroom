"""Does scaling CLIP vectors to length 1 change anything? Gate against the stored vectors, raw length against year and labels, 15 nearest neighbours raw vs unit against the re-encode floor, and the time and artist gaps on raw and unit vectors; reads reembed/wikiart_catfm.npz (clip_raw.py), compared inside clip_raw, not against the stored file.
Usage: norm_effect.py OUT_DIR"""
import json
import os
import sys

OUT = sys.argv[1]
GROUPS = ['after_1900', 'before_1900', 'after_1900_no_shared_artists', 'artist']
sys.argv = ['common.py', f'{OUT}/q1', '5', ','.join(GROUPS)]    # common reads its arguments at import
import numpy as np    # noqa: E402
import pandas as pd    # noqa: E402
from multiprocessing import get_context    # noqa: E402

import common    # noqa: E402
from data import ids, labels, year    # noqa: E402

os.makedirs(OUT, exist_ok=True)
res = {}
z = np.load(f'{os.environ["DUROOM_OUT"]}/reembed/wikiart_catfm.npz')
pos = {int(i): r for r, i in enumerate(z['ids'])}
assert len(pos) == len(z['ids']), 'duplicate ids in clip_raw'
missing = [int(i) for i in ids if int(i) not in pos]
assert not missing, f'{len(missing)} corpus ids missing from clip_raw'
raw = z['clip_raw'][[pos[int(i)] for i in ids]].astype('float64')    # row r = the same artwork as common.X row r
assert np.isfinite(raw).all()
norm = np.linalg.norm(raw, axis=1)
assert (norm > 0).all()
unit = raw / norm[:, None]
stored = common.X    # already unit length
res['n'] = int(len(ids))
res['norm_percentiles_1_10_50_90_99'] = np.percentile(norm, [1, 10, 50, 90, 99]).tolist()
res['norm_cv'] = float(norm.std() / norm.mean())

# gate: the raw rows are the corpus CLIP, re-encoded
cos = (unit * stored).sum(1)
gate = {'median': float(np.median(cos)), 'p05': float(np.percentile(cos, 5)), 'min': float(cos.min())}
gate['pass'] = gate['median'] >= 0.995 and gate['p05'] >= 0.98
res['gate'] = gate
print('gate', gate, flush=True)
if not gate['pass']:
    json.dump(res, open(f'{OUT}/norm_effect.json', 'w'), indent=1)
    sys.exit('gate failed: the raw rows are not the corpus CLIP vectors')


def rank(v):
    return pd.Series(v).rank().values    # average ranks: years tie by the hundred


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def eta2(v, g):
    keep = g != ''
    v, g = v[keep], g[keep]
    total = ((v - v.mean()) ** 2).sum()
    return float(sum(len(v[g == k]) * (v[g == k].mean() - v.mean()) ** 2 for k in np.unique(g)) / total)


dated = ~np.isnan(year)
mdir = unit.mean(0) / np.linalg.norm(unit.mean(0))
angle = np.degrees(np.arccos(np.clip(unit @ mdir, -1, 1)))
decade = np.where(dated, (np.nan_to_num(year) // 10 * 10).astype(int).astype(str), '')
res['length'] = {'spearman_year': spearman(norm[dated], year[dated]), 'spearman_angle_to_mean': spearman(norm, angle),
                 'eta2_style': eta2(norm, labels['style']), 'eta2_genre': eta2(norm, labels['genre']), 'eta2_decade': eta2(norm, decade)}
gm = {g: float(norm[labels['genre'] == g].mean()) for g, c in zip(*np.unique(labels['genre'], return_counts=True)) if g and c >= 500}
res['length']['genre_means_low_high'] = sorted(gm.items(), key=lambda x: x[1])[:5] + sorted(gm.items(), key=lambda x: x[1])[-5:]
print('length', res['length'], flush=True)


def knn(Q, Y, k, euclid):
    out = []
    y2 = (Y ** 2).sum(1)
    for s in range(0, len(Q), 500):
        q = Q[s:s + 500]
        d = y2[None, :] - 2 * q @ Y.T if euclid else -(q @ Y.T)    # |q|^2 is constant per row, so left out
        d[np.arange(len(q)), qi[s:s + 500]] = np.inf    # a work is not its own neighbour
        out.append(np.argpartition(d, k, axis=1)[:, :k].copy())    # copy, so the full index array is freed
    return np.vstack(out)


qi = np.random.default_rng(0).choice(len(ids), 5000, replace=False)
overlap = lambda a, b: float(np.mean([len(np.intersect1d(x, y)) / 15 for x, y in zip(a, b)]))    # noqa: E731
n_raw, n_unit, n_stored = knn(raw[qi], raw, 15, True), knn(unit[qi], unit, 15, False), knn(stored[qi], stored, 15, False)
res['neighbours'] = {'raw_vs_unit': overlap(n_raw, n_unit), 'floor_stored_vs_unit': overlap(n_stored, n_unit), 'queries': 5000, 'k': 15}
print('neighbours', res['neighbours'], flush=True)
json.dump(res, open(f'{OUT}/norm_effect.json', 'w'), indent=1)    # kept if the group part fails

# group-test cells on raw and on unit(clip_raw)
common.cells = [(g, 20, common.ALL) for g in GROUPS]
jobs = [(g, c, f, s) for g, c, f in common.cells for s in range(common.SEEDS)]
workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
res['q1'] = {}
for name, M in (('stored', stored), ('unit', unit), ('raw', raw)):    # stored vs unit = the re-encode floor for the gap
    common.X = M    # cell() reads the module global at call time; fork shares it
    with get_context('fork').Pool(workers) as p:
        rows = list(p.imap(common.cell, jobs, chunksize=1))
    k32 = common.KS.index(32)
    res['q1'][name] = {g: 100 * float(np.mean([r['unseen'][k32] - r['random'][k32] for r in rows if r['group'] == g])) for g in GROUPS}
    print('q1', name, res['q1'][name], flush=True)
res['q1']['moved_points'] = {g: res['q1']['raw'][g] - res['q1']['unit'][g] for g in GROUPS}
res['q1']['reencode_floor_points'] = {g: res['q1']['unit'][g] - res['q1']['stored'][g] for g in GROUPS}
res['q1']['pass'] = all(abs(v) <= 1 for v in res['q1']['moved_points'].values())
json.dump(res, open(f'{OUT}/norm_effect.json', 'w'), indent=1)
print('done', res['q1']['moved_points'], flush=True)
