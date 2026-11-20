"""Do CLIP, DINOv3 and SigLIP agree on which real works are far? Reads the per-work test-1 scores (S1_k15_excl) of the same works from three t1 runs and the 32-region fits; prints Spearman agreement, far counts and regions under 1% pre-1900 share, and writes agree_p1_v1.npz.
Usage: python -B agree_p1_v1.py (refuses to overwrite its output)"""
import csv
import os
import sys

import numpy as np

D = os.environ['DUROOM_OUT'] + '/tests12/t1'
OUT = os.environ['DUROOM_OUT'] + '/tests12/agree/agree_p1_v1.npz'
SP = ('clip', 'dino', 'siglip')
NAME = {'clip': 'CLIP', 'dino': 'DINOv3', 'siglip': 'SigLIP'}
RUNS = {'clip': (f'{D}/run_rob_clip', 'clip-vit-l-14'), 'dino': (f'{D}/run_headline_dinov3', 'dinov3-l'),
        'siglip': (f'{D}/run_headline_siglip', 'siglip-so400m')}
LEVELS = (99, 95)
fails = []


def check(name, ok, detail=''):
    print(f'CHECK {"PASS" if ok else "FAIL"}: {name} {detail}', flush=True)
    if not ok:
        fails.append(name)
        print('STOP: a check failed', flush=True)
        sys.exit(1)


def rank(x):
    """average ranks (ties share the mean rank), as scipy.stats.rankdata(method='average')."""
    x = np.asarray(x, np.float64)
    o = np.argsort(x, kind='mergesort')
    xs = x[o]
    b = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1], True])
    st, en = b[:-1], b[1:]
    r = np.empty(len(x))
    r[o] = np.repeat((st + en - 1) / 2.0 + 1.0, en - st)
    return r


def spearman(a, b):
    assert not np.isnan(a).any() and not np.isnan(b).any()
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


assert not os.path.exists(OUT), 'output exists: refusing to overwrite'
pw, h2, reg_csv = {}, {}, {}
for s in SP:
    run, enc = RUNS[s]
    pw[s] = np.load(f'{run}/{enc}_H1_perwork.npz', allow_pickle=True)
    h2[s] = np.load(f'{run}/{enc}_H2_allfit_K32.npz', allow_pickle=True)
    reg_csv[s] = [r for r in csv.DictReader(open(f'{run}/{enc}_H2_regions_K32.csv')) if r['fit'] == 'A_plus_later']
print('perwork keys (clip run):', sorted(pw['clip'].files), flush=True)

ids = pw['clip']['id']
same = all(np.array_equal(pw[s]['id'], ids) for s in SP)
print(f'three files list the same ids in the same order: {same}', flush=True)
if same:
    idx = {s: np.arange(len(ids)) for s in SP}
else:
    check('same id sets', all(set(pw[s]['id'].tolist()) == set(ids.tolist()) for s in SP))
    idx = {s: np.argsort(pw[s]['id'])[np.searchsorted(np.sort(pw[s]['id']), ids)] for s in SP}
    check('alignment by id', all(np.array_equal(pw[s]['id'][idx[s]], ids) for s in SP))
check('ids unique', len(np.unique(ids)) == len(ids), f'(n={len(ids)})')
g = pw['clip']['group_0A_1B_2later']
yr = np.asarray(pw['clip']['year'], np.float64)
check('group and year equal across the three files',
      all(np.array_equal(pw[s]['group_0A_1B_2later'][idx[s]], g) and np.array_equal(np.asarray(pw[s]['year'], np.float64)[idx[s]], yr, equal_nan=True) for s in SP))
check('artist equal across the three files', all(np.array_equal(np.array([str(a) for a in pw[s]['artist']])[idx[s]], np.array([str(a) for a in pw['clip']['artist']])) for s in SP))
S1 = {s: pw[s]['S1_k15_excl'][idx[s]] for s in SP}
print('S1 dtype:', {s: str(S1[s].dtype) for s in SP}, flush=True)
A, B, Lt = g == 0, g == 1, g == 2
e1, e2 = Lt & (yr < 1960), Lt & (yr >= 1960)
check('no NaN in S1 of A, B, later', all(not np.isnan(S1[s][A | B | Lt]).any() for s in SP))

# lines: np.percentile of B on the stored dtype; far = strictly above
line = {s: {L: float(np.percentile(S1[s][B], L)) for L in LEVELS} for s in SP}
far = {(s, L): S1[s] > line[s][L] for s in SP for L in LEVELS}

GROUPS = [('B (held-out pre-1900)', B), ('later (all)', Lt), ('later 1900-1959', e1), ('later 1960 on', e2), ('A (fit half; extra)', A)]
try:
    from scipy.stats import spearmanr
    a_, b_ = S1['clip'][Lt], S1['dino'][Lt]
    rs = float(spearmanr(a_, b_).correlation)
    check('rank function agrees with scipy spearmanr (later, CLIP-DINOv3)', abs(rs - spearman(a_, b_)) < 1e-9, f'({rs:.12f})')
except ImportError:
    print('scipy not available: rank function used without cross-check', flush=True)

print('\n### 1a. Spearman rank agreement of S1_k15_excl between spaces\n')
print('| group | n | CLIP-DINOv3 | CLIP-SigLIP | DINOv3-SigLIP |\n|---|---|---|---|---|')
rho = {}
for gname, m in GROUPS:
    r = [spearman(S1[a][m], S1[b][m]) for a, b in (('clip', 'dino'), ('clip', 'siglip'), ('dino', 'siglip'))]
    rho[gname] = r
    print(f'| {gname} | {int(m.sum()):,} | {r[0]:.4f} | {r[1]:.4f} | {r[2]:.4f} |')

print('\n### 1b. Far counts (far = S1 above the space\'s own L-th percentile of B; strict L = 99, looser L = 95)\n')
print('| line | group | n | far CLIP | far DINOv3 | far SigLIP | far DINOv3 and SigLIP (search rule) | far all three | of far DINOv3 and SigLIP: far CLIP | of far CLIP: far DINOv3 and SigLIP | DINOv3 and SigLIP, not CLIP | CLIP, not both DINOv3 and SigLIP |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
c = lambda x: int(x.sum())  # noqa: E731
for L in LEVELS:
    for gname, m in GROUPS[:4]:
        fc, fd, fs = far[('clip', L)] & m, far[('dino', L)] & m, far[('siglip', L)] & m
        ds = fd & fs
        print(f'| {L}th | {gname} | {c(m):,} | {c(fc):,} | {c(fd):,} | {c(fs):,} | {c(ds):,} | {c(ds & fc):,} | {c(ds & fc):,} of {c(ds):,} | {c(ds & fc):,} of {c(fc):,} | {c(ds & ~fc):,} | {c(fc & ~ds):,} |')

print('\n### 1c. Regions with under 1% pre-1900 share (pre1900_share_of_A_plus_later < 0.01, fit A_plus_later, K = 32), works from 1960 on\n')
low, region = {}, {}
for s in SP:
    q = h2[s]['Q_region']
    check(f'{NAME[s]}: allfit Q_ids equal perwork id order', np.array_equal(h2[s]['Q_ids'], pw[s]['id']))
    q = q[idx[s]]
    region[s] = q
    rows = sorted(reg_csv[s], key=lambda r: int(r['region']))
    check(f'{NAME[s]}: 32 A_plus_later rows in the regions csv', len(rows) == 32 and [int(r['region']) for r in rows] == list(range(32)))
    share = np.array([float(r['pre1900_share_of_A_plus_later']) for r in rows])
    nA, nL = np.bincount(q[A], minlength=32), np.bincount(q[Lt], minlength=32)
    check(f'{NAME[s]}: csv share equals A / (A + later) from Q_region', np.allclose(share, nA / (nA + nL), rtol=0, atol=1e-12)
          and np.array_equal(np.array([int(r['n_A']) for r in rows]), nA) and np.array_equal(np.array([int(r['n_later']) for r in rows]), nL))
    lr = np.flatnonzero(share < 0.01)
    low[s] = np.isin(q, lr)
lc, ld, ls_ = low['clip'] & e2, low['dino'] & e2, low['siglip'] & e2
print(f'works from 1960 on: n = {c(e2):,}\n')
print('| in an under-1% region in | works from 1960 on |\n|---|---|')
for nm, v in (('CLIP', lc), ('DINOv3', ld), ('SigLIP', ls_), ('all three', lc & ld & ls_), ('CLIP and at least one of DINOv3, SigLIP', lc & (ld | ls_)),
              ('DINOv3 and SigLIP (either way on CLIP)', ld & ls_), ('none of the three', e2 & ~low['clip'] & ~low['dino'] & ~low['siglip'])):
    print(f'| {nm} | {c(v):,} |')
print('\nall 8 patterns (CLIP, DINOv3, SigLIP; 1 = in an under-1% region), works from 1960 on:\n')
print('| CLIP | DINOv3 | SigLIP | works |\n|---|---|---|---|')
tot = 0
for a in (1, 0):
    for b in (1, 0):
        for d in (1, 0):
            v = e2 & (low['clip'] == bool(a)) & (low['dino'] == bool(b)) & (low['siglip'] == bool(d))
            tot += c(v)
            print(f'| {a} | {b} | {d} | {c(v):,} |')
check('patterns sum to the 1960-on works', tot == c(e2))

np.savez(OUT, id=ids, group_0A_1B_2later=g, year=yr, artist=np.array([str(a) for a in pw['clip']['artist']]),
         **{f'S1_{s}': S1[s] for s in SP}, **{f'line_{s}_{L}': np.float64(line[s][L]) for s in SP for L in LEVELS},
         **{f'far_{s}_{L}': far[(s, L)] for s in SP for L in LEVELS}, **{f'region_{s}': region[s] for s in SP},
         **{f'lowregion_{s}': low[s] for s in SP})
print(f'\nwrote {OUT}', flush=True)
print('ALL CHECKS PASSED' if not fails else f'CHECKS FAILED: {fails}')
