"""Which definition of "unexplored" (knn15, knn10, local, centre, leftover, loner; cutoff = R's own p99) separates later works from the past and from noise?
Usage: unexp.py [smoke|full]   (smoke = 3000 random R works and 3000 random later works, all noise)"""
import os
import sys
from multiprocessing import get_context

for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'
sys.path.insert(0, os.environ.get('RETRO_DIR', os.path.dirname(os.path.abspath(__file__))))
import numpy as np    # noqa: E402

from basis import capped, code    # noqa: E402
from data import ids, labels, lens, year    # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else 'full'
OBS = os.environ['DUROOM_EMB']
NOISE = os.environ['DUROOM_OUT'] + '/sphere/eval/bound/noise_check/noise_emb.npz'
ENCODERS = ('clip-vit-l-14', 'dinov3-l', 'siglip-so400m')
K, KDIR, CAP, CUT = 15, 32, 20, 1900
CANDS = ('knn15', 'knn10', 'local', 'centre', 'leftover', 'loner')
artist = labels['artist']
usable = (artist != '') & (year >= 1000) & (year <= 2026)
rows = np.flatnonzero(usable)
past, later = rows[year[rows] < CUT], rows[year[rows] >= CUT]
if MODE == 'smoke':
    past = np.sort(np.random.default_rng(0).choice(past, 3000, replace=False))
    later = np.sort(np.random.default_rng(1).choice(later, 3000, replace=False))
    rows = np.sort(np.concatenate([past, later]))
decade = np.floor(np.nan_to_num(year) / 10) * 10
_, CODES = np.unique(artist, return_inverse=True)
Z = np.load(NOISE)
paths = [str(p) for p in Z['paths']]
kind = np.array([p.rsplit('/', 1)[1][0] for p in paths])
assert len(paths) == 160 and all((kind == k).sum() == 32 for k in 'abcde')
AC = np.concatenate([CODES, -np.ones(160, dtype=CODES.dtype)])    # noise has no artist
NOISE_ROWS = len(artist) + np.arange(160)
QUERY = np.concatenate([rows, NOISE_ROWS])    # every row that gets a score
G = {}    # per-pool state shared with forked workers


def auc(a, b):
    """P(a > b) with ties split (rank form of Mann-Whitney)."""
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def pmap(fn, blocks):
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', len(os.sched_getaffinity(0))))) as p:
        return p.map(fn, blocks, chunksize=1)


def blocks_of(idx, n=500):
    return [idx[i:i + n] for i in range(0, len(idx), n)]


def sortedK(d):
    part = np.argpartition(d, K, axis=1)[:, :K]
    v = np.take_along_axis(d, part, axis=1)
    o = np.argsort(v, axis=1)
    return np.take_along_axis(v, o, 1), np.take_along_axis(part, o, 1)


def knn_block(idx):
    """Sorted K nearest distances to the pool, itself excluded ('all'), and with own artist excluded ('others')."""
    XA, pool, apool = G['XA'], G['pool'], G['apool']
    d = np.sqrt(np.maximum(2 - 2 * (XA[idx] @ G['XP'].T), 0))
    pos = np.searchsorted(pool, idx)
    own = (pos < len(pool)) & (pool[np.minimum(pos, len(pool) - 1)] == idx)
    d[np.flatnonzero(own), pos[own]] = np.inf
    da, ia = sortedK(d)
    d[AC[idx][:, None] == apool[None, :]] = np.inf
    do, _ = sortedK(d)
    return da, pool[ia], do


def knn_pass(XA, pool, query):
    G.update(XA=XA, pool=pool, apool=AC[pool], XP=XA[pool])
    parts = pmap(knn_block, blocks_of(query))
    da, ia, do = (np.concatenate(x) for x in zip(*parts))
    return da, ia, do


def loner_block(idx):
    XA, pool = G['XA'], G['lpool']
    c = XA[idx] - G['mean']
    r = c - (c @ G['V'].T) @ G['V']
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    s = r @ G['RP'].T
    pos = np.searchsorted(pool, idx)
    own = (pos < len(pool)) & (pool[np.minimum(pos, len(pool) - 1)] == idx)
    s[np.flatnonzero(own), pos[own]] = -np.inf
    s[AC[idx][:, None] == AC[pool][None, :]] = -np.inf
    part = np.argpartition(-s, 10, axis=1)[:, :10]
    return G['flag'][part].sum(1)    # how many of its 10 nearest are over the leftover cutoff


def loner_pass(XA, mean, V, pool, flag, query):
    c = XA[pool] - mean
    RP = c - (c @ V.T) @ V
    RP /= np.linalg.norm(RP, axis=1, keepdims=True)
    G.update(XA=XA, mean=mean, V=V, lpool=pool, RP=RP, flag=flag)
    return np.concatenate(pmap(loner_block, blocks_of(query)))


def run(enc):
    print(f'\n==================== {enc} ({MODE}) ====================', flush=True)
    X = lens(f'{OBS}/{enc}.npz').astype('float64')
    N = Z[f'emb_{enc}'].astype('float64')
    assert np.isfinite(X).all() and (np.linalg.norm(X, axis=1) > 1e-8).all() and np.isfinite(N).all()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    XA = np.vstack([X, N])
    del X
    nA = len(XA)
    isnoise = np.zeros(nA, bool)
    isnoise[NOISE_ROWS] = True
    da, ia, do = knn_pass(XA, past, QUERY)
    S = {k: np.full(nA, np.nan) for k in CANDS}
    abs_all, abs_oth = np.full(nA, np.nan), np.full(nA, np.nan)
    abs_all[QUERY], abs_oth[QUERY] = da.mean(1), do.mean(1)
    S['knn15'], S['knn10'][QUERY] = abs_oth, do[:, 9]
    nb = np.zeros((nA, K), dtype=int)
    nb[QUERY] = ia
    S['local'][QUERY] = abs_all[QUERY] / np.median(abs_all[nb[QUERY]], axis=1)    # the neighbours' own leave-one-out level
    m = XA[past].mean(0)
    m /= np.linalg.norm(m)
    S['centre'] = np.degrees(np.arccos(np.clip(XA @ m, -1, 1)))
    rng = np.random.default_rng(0)
    names = rng.permutation(np.unique(artist[past]))
    fit_half = np.isin(artist[past], names[:len(names) // 2])
    fit_rows = capped(past[fit_half], CAP, rng)
    mean, V = code(XA[fit_rows])
    V = V[:KDIR]
    c = XA - mean
    proj = c @ V.T
    S['leftover'] = 1 - (proj ** 2).sum(1) / (c ** 2).sum(1)
    del c
    held = past[~fit_half]    # the other half of R's artists: the leftover's reference, never in the fit
    REF = {k: past for k in CANDS}
    REF['leftover'] = REF['loner'] = held
    cut = {k: float(np.percentile(S[k][REF[k]], 99)) for k in CANDS[:5]}
    over = {k: S[k] > cut[k] for k in CANDS[:5]}
    test = QUERY[over['leftover'][QUERY]]
    flag_h, flag_a = S['leftover'][held] > cut['leftover'], S['leftover'][past] > cut['leftover']
    over_l = over['leftover']
    lon = {}
    for name, pool, flag in (('loner', held, flag_h), ('loner_allR', past, flag_a)):
        cnt = loner_pass(XA, mean, V, pool, flag, test)
        o = np.zeros(nA, bool)
        o[test[cnt == 0]] = True
        lon[name] = o
    over['loner'] = lon['loner']
    S['loner'] = lon['loner'].astype(float)
    S['loner_allR'] = lon['loner_allR'].astype(float)
    REF['loner_allR'] = held
    over['loner_allR'] = lon['loner_allR']
    print(f'R {len(past)} works (held-out half for leftover/loner {len(held)}, fit half {int(fit_half.sum())}, fit sample {len(fit_rows)}); later {len(later)}; noise 160')
    print('cutoffs (R p99): ' + ', '.join(f'{k} {cut[k]:.4f}' for k in CANDS[:5]) + f' | leftover cutoff is used by loner; knn15 shown as mean distance, centre in degrees')
    dec = sorted(int(d) for d in np.unique(decade[later]))
    print('later works per decade: ' + ' '.join(f'{d}:{int((decade[later] == d).sum())}' for d in dec))
    era = {'all': later, '1900-1959': later[year[later] < 1960], '1960 on': later[year[later] >= 1960]}
    print('\nR pass-rate check: share of the reference over the cutoff (reference = R; leftover and loner: held-out half)')
    print('  ' + ' | '.join(f'{k} {over[k][REF[k]].mean():.4f}' for k in CANDS + ('loner_allR',)))
    print(f'  loners in R (held-out half, pool=held-out): {int(over["loner"][held].sum())} of {len(held)} ({over["loner"][held].mean():.4f}); of R held-out over the leftover cutoff {int(over_l[held].sum())};'
          f' pool=all R: {int(over["loner_allR"][held].sum())} of {len(held)} ({over["loner_allR"][held].mean():.4f})')
    print(f'  (info) knn/centre/local R pass on the fit half too: leftover on the fit half {over_l[past[fit_half]].mean():.4f} (in-sample, not a check)')
    print('\nmain table: share of later works over the cutoff; AUC later vs R; noise over the cutoff of 160 and per kind a uniform / b gaussian / c solid / d smooth / e blocks (of 32 each)')
    print(f'{"candidate":11s} {"later all":>9s} {"1900-59":>8s} {"1960on":>7s} {"AUC":>6s} {"noise/160":>9s}   a  b  c  d  e')
    for k in CANDS + ('loner_allR',):
        ov = over[k]
        a = auc(S[k][later], S[k][REF[k]])
        nz = ov[NOISE_ROWS]
        print(f'{k:11s} {ov[era["all"]].mean():9.4f} {ov[era["1900-1959"]].mean():8.4f} {ov[era["1960 on"]].mean():7.4f} {a:6.3f} {int(nz.sum()):9d}  ' + ' '.join(f'{int(nz[kind == q].sum()):2d}' for q in 'abcde'))
    print('\nby decade, share of later works over the cutoff (n in header)')
    print(f'{"candidate":11s} ' + ' '.join(f'{d:>6d}' for d in dec))
    print(f'{"n":11s} ' + ' '.join(f'{int((decade[later] == d).sum()):6d}' for d in dec))
    for k in CANDS + ('loner_allR',):
        print(f'{k:11s} ' + ' '.join(f'{over[k][later[decade[later] == d]].mean():6.3f}' for d in dec))
    print('\ncap sweep, knn15 (mean distance to the 15 nearest R works); R capped per artist (seed 0); cutoff = p99 of that R\'s own scores')
    print(f'{"cap":>6s} {"own-artist":>10s} {"nR":>6s} {"cutoff":>8s} {"later all":>9s} {"later 1960on":>12s} {"noise/160":>9s}')
    sweep = {}
    for cap in (1, 3, 10, None):
        if cap is None:
            Rc, a_all, a_oth = past, abs_all, abs_oth
        else:
            Rc = np.sort(capped(past, cap, np.random.default_rng(0)))
            d2, _, o2 = knn_pass(XA, Rc, np.concatenate([Rc, later, NOISE_ROWS]))
            a_all, a_oth = np.full(nA, np.nan), np.full(nA, np.nan)
            q = np.concatenate([Rc, later, NOISE_ROWS])
            a_all[q], a_oth[q] = d2.mean(1), o2.mean(1)
        for name, a in (('excluded', a_oth), ('allowed', a_all)):
            ct = float(np.percentile(a[Rc], 99))
            ov = a > ct
            r = (float(ct), float(ov[later].mean()), float(ov[era['1960 on']].mean()), int(ov[NOISE_ROWS].sum()))
            sweep[(cap, name)] = r
            print(f'{"none" if cap is None else cap:>6} {name:>10s} {len(Rc):6d} {r[0]:8.4f} {r[1]:9.4f} {r[2]:12.4f} {r[3]:9d}   (R pass check {ov[Rc].mean():.4f})')
    for name in ('excluded', 'allowed'):
        a, b = sweep[(3, name)], sweep[(None, name)]
        print(f'cap 3 -> no cap, own-artist {name}: later 1960on {a[2]:.4f} -> {b[2]:.4f} (ratio {b[2] / a[2]:.3f}); noise {a[3]} -> {b[3]} (ratio {b[3] / a[3] if a[3] else float("nan"):.3f}); cutoff {a[0]:.4f} -> {b[0]:.4f}')
    sys.stdout.flush()


if __name__ == '__main__':
    print(f'unexp.py mode {MODE}; usable works {len(rows)}; R {len(past)}; later {len(later)}', flush=True)
    for enc in ENCODERS:
        run(enc)
    print('done', flush=True)
