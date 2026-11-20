#!/bin/bash
# Read-only checks of the held-out and new-axis outputs (parts a-d below); writes nothing. Each part replays a stored run's draws, and a GATE line says whether a stored number was reproduced.
# Needs run/02 and run/08 (newaxis, newaxis_check, newaxis_more, common_moves).
set -u
E=$(dirname "$(dirname "$(command -v "${DUROOM_PY:-python}")")")
export LD_LIBRARY_PATH=$E/lib PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
$E/bin/python -B - "$DUROOM_OUT" "$DUROOM_REPO/data" <<'PY'
import json, os, sys, zlib

O, S = sys.argv[1], sys.argv[2]


def need(path):
    """Stop with the parent folder's listing instead of guessing a name."""
    if os.path.exists(path):
        return path
    d = os.path.dirname(path)
    while d and not os.path.isdir(d):
        d = os.path.dirname(d)
    print(f'STOP: {path} not found. Contents of {d}:')
    for x in sorted(os.listdir(d)):
        print('   ', x)
    sys.exit(2)


FILES = {'data': f'{S}/data.py', 'basis': f'{S}/basis.py',
         'clip_npz': f'{O}/newaxis_check/k32/loads_1900.npz', 'clip_json': f'{O}/newaxis_check/k32/newaxis.json',
         'dino_npz': f'{O}/newaxis/k32_lensdinov3-l/loads_1900.npz', 'sig_npz': f'{O}/newaxis/k32_lenssiglip-so400m/loads_1900.npz',
         'moves_clip': f'{O}/common_moves/clip-vit-l-14/heldout.json'}
for p in FILES.values():    # every input checked before any work
    need(p)
print('inputs:', json.dumps(FILES, indent=1), flush=True)

sys.path.insert(0, S)
import numpy as np    # noqa: E402
from basis import KS, capped, code, each_kept    # noqa: E402
from data import CLIP, labels, lens, year    # noqa: E402

artist, style, genre = labels['artist'], labels['style'], labels['genre']
K32 = KS.index(32)

# ---------------------------------------------------------------- (a) label means on 1900-2026 only
print('\n==== (a) label means along axis 1 and axis 2, all works (stored) vs works 1900-2026 only', flush=True)
z = np.load(FILES['clip_npz'])
rows = z['rows']
assert np.array_equal(year[rows], z['year']), 'npz rows are not the same works as data.py'
J = json.load(open(FILES['clip_json'], encoding='utf8'))['cutoffs']['1900']
full = {}
for k in ('axis1', 'axis2', 'shift'):
    a = np.full(len(year), np.nan)
    a[rows] = z[k].astype('float64')
    full[k] = a


def by_label(load, v, idx, least=200):    # as newaxis.py
    vv = v[idx]
    names, counts = np.unique(vv[vv != ''], return_counts=True)
    return sorted(((n, float(load[idx][vv == n].mean()), int(c)) for n, c in zip(names, counts) if c >= least), key=lambda x: x[1])


later = rows[year[rows] >= 1900]
print(f'works: all usable {len(rows)}, 1900-2026 {len(later)}')
for key, ax, v in (('by_style', 'axis1', style), ('by_genre', 'axis1', genre), ('by_style_axis2', 'axis2', style), ('by_genre_axis2', 'axis2', genre)):
    stored = J[key]
    redo_all = by_label(full[ax], v, rows)
    diff = max(abs(m - dict((n, mm) for n, mm, _ in redo_all)[n]) for n, m in stored)
    gate = 'GATE OK' if diff < 1e-5 and len(stored) == len(redo_all) else 'GATE FAILED: these loads are not the stored run'
    print(f'\n-- {key} ({ax}); recomputed on all works vs stored: max abs diff {diff:.2e}, {len(stored)} vs {len(redo_all)} labels; {gate}')
    late = by_label(full[ax], v, later)
    rank_late = {n: i for i, (n, _, _) in enumerate(late[::-1], 1)}
    print('   top 10, stored (all works 1000-2026)          | top 10, works 1900-2026 only (200+ works)')
    for i in range(10):
        s = stored[-1 - i] if i < len(stored) else ('', np.nan)
        t = late[-1 - i] if i < len(late) else ('', np.nan, 0)
        print(f'   {i + 1:2d} {s[0][:30]:30s} {s[1]:+.3f} (rank late {rank_late.get(s[0], "-")!s:>3}) | {t[0][:30]:30s} {t[1]:+.3f} n {t[2]}')
    print('   bottom 5 stored:', [(n, round(m, 3)) for n, m in stored[:5]], '| bottom 5 late:', [(n, round(m, 3)) for n, m, _ in late[:5]])

# ---------------------------------------------------------------- (b) rank correlation along each encoder's own shift
print('\n==== (b) rank correlation of the works along each encoder\'s own shift (seed 0 directions, sign: later works positive)', flush=True)
Z = {'clip': z, 'dinov3': np.load(FILES['dino_npz']), 'siglip': np.load(FILES['sig_npz'])}
for n, zz in Z.items():
    assert np.array_equal(zz['rows'], rows), f'{n}: rows differ from CLIP'
rank = lambda a: np.argsort(np.argsort(a))    # noqa: E731
for subset, idx in (('all usable works 1000-2026', np.arange(len(rows))), ('works 1900-2026', np.flatnonzero(z['year'] >= 1900)), ('works before 1900', np.flatnonzero(z['year'] < 1900))):
    r = {n: rank(zz['shift'][idx]) for n, zz in Z.items()}
    print(f'   {subset} (n {len(idx)}):', {f'{a}-{b}': round(float(np.corrcoef(r[a], r[b])[0, 1]), 3) for a, b in (('clip', 'dinov3'), ('clip', 'siglip'), ('dinov3', 'siglip'))})

# ---------------------------------------------------------------- shared: CLIP as in common.py and newaxis.py
X = lens(CLIP).astype('float64')
X /= np.linalg.norm(X, axis=1, keepdims=True)
named = artist != ''
dated = ~np.isnan(year)
later_q1, earlier_q1 = dated & (year >= 1900), dated & (year < 1900)

# ---------------------------------------------------------------- (c) composition of the random basis (common.py, copied, not imported)
print('\n==== (c) what the random basis B holds, replaying common.py cells (cap 20, all works, seeds 0-4)', flush=True)


def kmeans(rows_, k, rng, iters=25):    # as common.py, which creates folders and thumbnails when imported
    fit = X[capped(rows_, 20, rng)]
    C = fit[rng.choice(len(fit), k, replace=False)]
    for _ in range(iters):
        lab = np.argmax(fit @ C.T, axis=1)
        C = np.stack([fit[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    return np.argmax(X @ C.T, axis=1), C


cluster, _ = kmeans(np.flatnonzero(named), 16, np.random.default_rng(16))
m15 = named & (cluster == 15)
top = sorted(zip(*np.unique(style[m15][style[m15] != ''], return_counts=True)), key=lambda x: -x[1])[:3]
print(f'   cluster 15: {int(m15.sum())} works, top styles {top}', '(GATE OK)' if top[0][0] == 'Ukiyo-e' else '(GATE FAILED: clusters differ, part (c) for the prints does not count)')
stored = {(r['group'], r['seed']): r for r in json.load(open(FILES['moves_clip']))['rows'] if r['cap'] == 20 and r['fit'] not in (1000, 5000, 10000)}
GROUPS = {'after_1900': (later_q1, earlier_q1), 'cluster:15': (cluster == 15, cluster != 15)}
for name, (t, a) in GROUPS.items():
    out = []
    for seed in range(5):
        rng = np.random.default_rng([seed, zlib.crc32(f'{name}{20}{10 ** 9}'.encode())])    # as common.py, fit = ALL
        t_pool, a_pool = np.flatnonzero(t & named), np.flatnonzero(a & named)            # by_mask
        T = capped(t_pool, 20, rng)
        A = capped(np.setdiff1d(a_pool, T), 20, rng)
        pool = rng.permutation(np.setdiff1d(np.flatnonzero(named), T))
        Bb, B2 = capped(pool[:len(pool) // 2], 20, rng), capped(pool[len(pool) // 2:], 20, rng)
        n = min(len(A), len(Bb), len(B2))
        Bb = Bb[:n]
        mB, VB = code(X[Bb])
        cum, total = each_kept(X[T], mB, VB)
        kept_random = float(cum.sum(0)[K32] / total.sum())
        s = stored.get((name, seed))
        gate = abs(kept_random - s['random'][K32]) < 1e-9 and s['n_test'] == len(T) and s['fit'] == n if s else False
        out.append((seed, n, len(T), float(t[Bb].mean()), float(np.isnan(year[Bb]).mean()), float((year[Bb] >= 1900).mean()), gate))
    print(f'   {name}: per seed (n_basis, n_test, share of B in the group, share undated, share dated 1900 on, GATE vs stored random kept)')
    for o in out:
        print(f'      seed {o[0]}: {o[1]} {o[2]} group {o[3]:.4f} undated {o[4]:.4f} 1900-on {o[5]:.4f} {"GATE OK" if o[6] else "GATE FAILED"}')
    print(f'   {name}: mean share of B in the group {np.mean([o[3] for o in out]):.4f}')

# ---------------------------------------------------------------- (d) past spread along directions outside the past basis
print('\n==== (d) past works\' spread (RMS loading from the past mean) along directions outside the 32-direction past basis', flush=True)
usable = named & (year >= 1000) & (year <= 2026)
past = np.flatnonzero(usable & (year < 1900))
rng = np.random.default_rng([1900, 0])    # newaxis.py seed 0: the first draw is capped(past) inside axes()
mean, Vfull = code(X[capped(past, 20, rng)])
D = J['directions']
w1 = np.array(D['axis1'])
g = np.abs(np.abs((X[rows] - mean) @ w1) - np.abs(z['axis1'].astype('float64'))).max()
print(f'   GATE: axis-1 loadings rebuilt from this basis mean vs the stored npz, max abs diff {g:.2e}', '(GATE OK)' if g < 1e-5 else '(GATE FAILED: not the stored basis, part (d) does not count)')
chance = np.random.default_rng([1900, 7]).standard_normal(X.shape[1])    # newaxis.py, the stored random_direction
chance -= Vfull[:32].T @ (Vfull[:32] @ chance)
chance /= np.linalg.norm(chance)
rk = int(np.searchsorted(np.cumsum((Vfull @ chance) ** 2), 0.5)) + 1
print(f'   GATE: random direction rank_half {rk} (stored {J["past_rank"]["random_direction"]["rank_half"]})')
P = X[past] - mean
rms = lambda v: float(np.sqrt(((P @ v) ** 2).mean()))    # noqa: E731
print(f'   past n {len(past)} (all usable works before 1900, as the loadings use)')
print(f'   RMS along axis 1 {rms(w1):.4f}, shift {rms(np.array(D["shift"])):.4f}, axis 2 {rms(np.array(D["axis2"])):.4f}')
print(f'   RMS along the stored random direction {rms(chance):.4f}')
Q = P @ Vfull[32:].T    # past works in the 736 directions outside the basis
G = np.random.default_rng(0).standard_normal((Q.shape[1], 1000))
G /= np.linalg.norm(G, axis=0)
r = np.sqrt(((Q @ G) ** 2).mean(0))
print(f'   RMS along 1,000 random directions outside the basis: median {np.median(r):.4f}, 10th-90th pct {np.percentile(r, 10):.4f}-{np.percentile(r, 90):.4f}')
print(f'   root mean variance over the 736 directions outside the basis {float(np.sqrt((Q ** 2).mean())):.4f}; along direction 33 {float(np.sqrt((Q[:, 0] ** 2).mean())):.4f}, 768 {float(np.sqrt((Q[:, -1] ** 2).mean())):.4f}')
print('\ndone (nothing written)', flush=True)
PY
