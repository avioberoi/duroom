"""smooth.py's stages plus stage ax, the artist 2x2: T1's held-out share and T3's random-partner overlap with the anchor's artist excluded or included, fit and test halves split by artist or by work (real, gauss and mix:K only; writes ax.json and ax.npz).
Usage: smooth_ax.py OUT STAGES CLOUD [LENS file, default CLIP] [SEED=0]; STAGES a comma list of t4,calib,t1,t2,t3,grid,verdict,ax; SMOOTH_SMOKE=1 shrinks every sample"""
import csv
import json
import os
import sys
import time
import traceback
import zlib
from multiprocessing import get_context
from multiprocessing.pool import ThreadPool

for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'    # one BLAS thread per process; the processes are the parallelism
import numpy as np    # noqa: E402
from scipy import stats    # noqa: E402
from scipy.interpolate import LinearNDInterpolator    # noqa: E402

from data import CLIP, card, ids, labels, lens, year    # noqa: E402

SMOKE = os.environ.get('SMOOTH_SMOKE') == '1'
NH, NE, NT3, NLB, NCAL, NREF, NPLANT, NPAIR, NBOOT, NGRIDQ = (150, 80, 60, 600, 300, 400, 12, 60, 100, 100) if SMOKE else (2000, 1000, 1000, 5000, 1000, 2000, 200, 500, 1000, 1000)
KS, MS, RANKS = [10, 30, 100, 300] + [1000] * (not SMOKE), [1, 2, 5, 10, 20, 50, 100], [1, 3, 10, 30, 100, 300, 1000, 3000]
CAL_D, CAL_DU = [20, 24, 28, 32, 36, 40, 48], [24, 32, 40, 48, 60, 72, 84, 96, 112]
GRID_D, GRID_S = ([20, 32], [0, 0.5]) if SMOKE else ([20, 26, 32, 38, 44], [0, 0.25, 0.5, 0.65])
RHOS, KEEPS, BINS = [1.0, 1.2, 1.4, 1.7, 2.0], [0, 0.2], [0, 40, 50, 60, 181]
ERAS = {'before_1900': (0, 1900), '1900_1959': (1900, 1960), '1960_on': (1960, 3000)}
ORDER = ['t4', 'calib', 't1', 't2', 't3', 'grid', 'verdict', 'ax']
AXK = [60, 100, 300] + [1000] * (not SMOKE)    # ax: neighbours per half; 100 and 300 are the main ones, the ladder is for radius matching
CELLS = {'EA': (True, True), 'EW': (True, False), 'IA': (False, True), 'IW': (False, False)}    # ax: (anchor's artist excluded, split by artist)
f64, f32 = np.float64, np.float32
X = NRM = CODES = ART_O = ART_B = TRUTH = VG = RDIR = None    # the current cloud and what the workers read from it


class Infeasible(Exception):
    pass


# ---------- helpers ----------
def pmap(f, jobs):
    """Map in parallel: fork where it exists (workers see the globals as they are now), threads elsewhere."""
    n = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    with (get_context('fork').Pool if hasattr(os, 'fork') else ThreadPool)(n) as p:
        return p.map(f, jobs, chunksize=1)


def unit(Y):
    Y = np.asarray(Y, f32)
    for b in range(0, len(Y), 20000):
        Y[b:b + 20000] /= np.linalg.norm(Y[b:b + 20000], axis=1, keepdims=True)
    return Y


def unitv(v):
    v = np.asarray(v, f64)
    return v / np.linalg.norm(v)


def deg(chord):
    return np.degrees(2 * np.arcsin(np.minimum(chord, 2) / 2))


def angles(x, Y):
    return np.degrees(np.arccos(np.clip(np.asarray(Y, f64) @ x, -1, 1)))


def q3(v):
    v = np.asarray(v, f64)
    v = v[np.isfinite(v)]
    return np.percentile(v, [10, 50, 90]).tolist() if len(v) else None


def boot(v, f=np.median):
    v = np.asarray(v, f64)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return None
    r = np.random.default_rng(17)
    return np.percentile([f(v[r.integers(0, len(v), len(v))]) for _ in range(NBOOT)], [2.5, 97.5]).tolist()


def spread(v):
    v = v[np.isfinite(v)]
    return float(np.subtract(*np.percentile(v, [75, 25])) / 1.349)


def resid(y, x):
    ok = np.isfinite(y) & np.isfinite(x)
    return y - np.polyval(np.polyfit(x[ok], y[ok], 1), x)


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(stats.spearmanr(a[ok], b[ok])[0])


def clean(o):
    """JSON-safe: numpy to Python, NaN and inf to null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in (o.tolist() if isinstance(o, np.ndarray) else o)]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def cdir(cloud, seed=0):
    return f'{BASE}/{cloud.replace(":", "_")}_s{seed}'


def load(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def res(cloud, seed=0):
    return load(f'{cdir(cloud, seed)}/smooth.json')


def merge(path, new):
    """Read-modify-write: replace only the given keys, so one stage never drops another's."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = load(path) or {}
    old.update(clean(new))
    json.dump(old, open(path, 'w'), indent=1)


def R_data():
    return res('real')['t4']['R']


def d_ctrl():
    return load(f'{BASE}/calib.json')['d_ctrl']


# ---------- the current cloud and its neighbour search ----------
def set_cloud(Y, codes=None):
    global X, NRM, CODES, ART_O, ART_B, VG
    X = np.ascontiguousarray(Y, dtype=f32)
    NRM = np.concatenate([(X[b:b + 20000].astype(f64) ** 2).sum(1) for b in range(0, len(X), 20000)])
    CODES = ART if codes is None else codes
    ART_O = np.argsort(CODES, kind='stable')
    ART_B = np.searchsorted(CODES[ART_O], np.arange(CODES.max() + 2))
    VG = None


def masked(Q, allow, excl, selfr):
    """Similarities of a block of query vectors to every row, with disallowed rows, excluded artists and the query itself at -inf."""
    G = Q @ X.T
    if allow is not None:
        G[:, ~allow] = -np.inf
    for i in range(len(Q)):
        for a in excl[i]:
            if a >= 0:
                G[i, ART_O[ART_B[a]:ART_B[a + 1]]] = -np.inf
        if selfr[i] >= 0:
            G[i, selfr[i]] = -np.inf
    return G


def _knn_block(job):
    Q, K, allow, excl, selfr = job
    G = masked(Q, allow, excl, selfr)
    c = min(K + 16, G.shape[1])    # 16 spare candidates for exact duplicates
    I = np.argpartition(G, G.shape[1] - c, axis=1)[:, -c:]
    q64 = Q.astype(f64)
    D = np.stack([np.sqrt(np.maximum(NRM[I[i]] + q64[i] @ q64[i] - 2 * (X[I[i]].astype(f64) @ q64[i]), 0)) for i in range(len(Q))])    # exact chords
    D[np.take_along_axis(G, I, 1) == -np.inf] = np.inf
    dup = D < 1e-4    # an exact duplicate of the query: skipped and counted
    D[dup] = np.inf
    o = np.argsort(D, axis=1)[:, :K]
    D, I = np.take_along_axis(D, o, 1), np.take_along_axis(I, o, 1)
    I[~np.isfinite(D)] = -1
    return D, I, int(dup.sum())


def search(Q, K, allow=None, excl=None, selfr=None, block=500):
    """The K nearest rows to each query by exact chord; allow: bool mask over rows, excl: artist codes to skip, selfr: own row or -1. Returns distances, rows (-1 past the end), duplicates skipped."""
    Q = np.asarray(Q, f32)
    n = len(Q)
    excl = np.full((n, 1), -1) if excl is None else np.asarray(excl).reshape(n, -1)
    selfr = np.full(n, -1) if selfr is None else np.asarray(selfr)
    parts = pmap(_knn_block, [(Q[b:b + block], K, allow, excl[b:b + block], selfr[b:b + block]) for b in range(0, n, block)])
    return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts]), sum(p[2] for p in parts)


def others(r, K, allow=None):
    """Other-artist neighbours of cloud rows r."""
    return search(X[r], K, allow, CODES[r])


def _count_block(job):
    Q, allow, excl, rads = job
    G = masked(Q, allow, excl, np.full(len(Q), -1))
    return np.stack([(G > 1 - r * r / 2).sum(1) for r in rads], 1)


def lb(T, k):
    """Levina-Bickel per row from ascending distances T_1..T_k."""
    return (k - 1) / np.log(T[:, k - 1:k] / T[:, :k - 1]).sum(1)


def spacing():
    r = np.random.default_rng(1).choice(len(X), min(NCAL, len(X)), replace=False)
    return float(np.median(others(r, 10)[0][:, 9]))


def describe():
    r = np.random.default_rng(1).choice(len(X), min(NCAL, len(X)), replace=False)
    D, _, _ = others(r, 1000)
    e = unitv(X.mean(0, dtype=f64))
    return {'d10': float(np.median(D[:, 9])), 'cap_deg': float(np.median(angles(e, X))), 'lb20_median': float(np.median(lb(D, 20))),
            'r_k_deg': {k: float(np.median(deg(D[:, k - 1]))) for k in (1, 10, 100, 1000) if k <= D.shape[1]}, 'D': X.shape[1]}


# ---------- geometry ----------
def logmap(x, Y):
    Y = np.asarray(Y, f64)
    c = np.clip(Y @ x, -1, 1)
    th = np.arccos(c)
    return np.where(th > 1e-8, th / np.maximum(np.sin(th), 1e-12), 1.0)[:, None] * (Y - c[:, None] * x)


def plane(x, Y):
    """Centred tangent offsets of Y at x, their variances (descending) and principal directions as rows."""
    V = logmap(x, Y)
    V -= V.mean(0)
    _, s, Vt = np.linalg.svd(V, full_matrices=False)
    return V, s ** 2, Vt


def share(U, V):
    return float(((V @ U.T) ** 2).sum() / (V ** 2).sum())


def transport(U, x, y):
    """Parallel transport of tangent vectors (rows) at x to y along the great circle."""
    return U - np.outer((U @ y) / (1 + x @ y), x + y)


def overlap(U, V):
    return float(((U @ V.T) ** 2).sum() / len(U))


def cov(Y):
    mu = Y.mean(0, dtype=f64)
    C = np.zeros((Y.shape[1], Y.shape[1]))
    for b in range(0, len(Y), 20000):
        Z = Y[b:b + 20000].astype(f64) - mu
        C += Z.T @ Z
    w, U = np.linalg.eigh(C / len(Y))
    return mu, np.maximum(w[::-1], 0), U[:, ::-1]


def gdirs():
    """The cloud's own global directions (top 300) and 20 random directions, for T1's global share and widths."""
    global VG, RDIR
    if VG is None:
        VG = np.ascontiguousarray(cov(X)[2][:, :300].T)
        RDIR = np.random.default_rng(21).standard_normal((20, X.shape[1]))
        RDIR /= np.linalg.norm(RDIR, axis=1, keepdims=True)


def kmeans(Y, k, seed=16, iters=30):
    """Spherical k-means on unit rows; returns labels."""
    C = Y[np.random.default_rng(seed).choice(len(Y), k, replace=False)].astype(f64)
    for _ in range(iters):
        lab = np.argmax(Y @ C.T.astype(f32), axis=1)
        C = np.stack([Y[lab == j].mean(0, dtype=f64) if (lab == j).any() else C[j] for j in range(k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    return np.argmax(Y @ C.T.astype(f32), axis=1)


# ---------- clouds ----------
def match(make, lo, hi):
    """Bisect one scale (geometric) until the median other-artist d10 equals R within 0.005."""
    R = R_data()
    make(hi)
    if spacing() < R - 0.005:
        raise Infeasible
    for it in range(40):
        mid = np.sqrt(lo * hi)
        make(mid)
        sp = spacing()
        if abs(sp - R) <= 0.005:
            return {'scale': float(mid), 'd10_matched': sp, 'steps': it + 1}
        lo, hi = (mid, hi) if sp < R else (lo, mid)
    raise Infeasible


def lin(axis, W, s, eps=None, sa=0.0):
    Y = np.zeros((len(W), len(axis)), f32) if eps is None else eps * f32(sa)
    Y[:, :W.shape[1]] += f32(s) * W
    Y += axis.astype(f32)
    return unit(Y)


def dim(t):
    return d_ctrl() if t == 'ctrl' else int(t)


def build(name, half=False):
    """Make the named cloud the current one (corpus row order and artist labels) and describe it. half: a sheet of N/2 rows (calib)."""
    global TRUTH
    TRUTH = None
    kind, *a = name.split(':')
    if kind == 'real':
        set_cloud(X0)
        return {'name': name, **describe()}
    rng = lambda tag: np.random.default_rng([SEED, zlib.crc32(f'{name}:{tag}'.encode())])    # noqa: E731
    n, codes = (N // 2, ART[:N // 2]) if half else (N, ART)
    info = {'name': name}
    if kind == 'gauss':    # corpus mean + c * Gaussian with the corpus covariance
        mu, w, U = cov(X0)
        r = rng('z')
        B = np.concatenate([((r.standard_normal((min(20000, N - b), D)) * np.sqrt(w)) @ U.T).astype(f32) for b in range(0, N, 20000)])
        make, lo, hi = (lambda c: set_cloud(unit(mu.astype(f32) + f32(c) * B))), 0.05, 20.0
    elif kind == 'mix':    # the corpus's own k-means lumps, each filled with a Gaussian of its covariance (top 200 + isotropic rest)
        lab, r = kmeans(X0, int(a[0])), rng('z')
        A, B = np.empty_like(X0), np.zeros_like(X0)
        for j in range(int(a[0])):
            idx = np.flatnonzero(lab == j)
            if len(idx) == 0:
                continue
            mu, w, U = cov(X0[idx])
            t = min(200, len(idx) - 1)
            rest = w[t:].mean()
            A[idx] = mu
            B[idx] = (r.standard_normal((len(idx), t)) * np.sqrt(np.maximum(w[:t] - rest, 0))) @ U[:, :t].T + np.sqrt(rest) * r.standard_normal((len(idx), D))
        make, lo, hi = (lambda c: set_cloud(unit(A + f32(c) * B))), 0.05, 20.0
    elif kind in ('sheet', 'curved', 'lumpy'):
        d, r = dim(a[0]), rng('z')
        sig = (sigma_eq_t3() if a[1] == 'eq' else float(a[1])) if kind == 'sheet' else 0.0
        Dc = D if sig > 0 else (2 * d + 1 if kind == 'curved' else d + 1)    # a sigma = 0 cloud is built isometrically in few coordinates
        axis = np.zeros(Dc)
        axis[0] = 1
        W = np.zeros((n, d + 1 if kind != 'curved' else 2 * d + 1), f32)
        eps = r.standard_normal((n, Dc), dtype=f32) if sig > 0 else None
        if kind == 'sheet':
            W[:, 1:] = r.standard_normal((n, d))
        elif kind == 'curved':    # every latent axis bent into an arc of radius f*s; truncated so the arcs do not close
            f = float(a[1])
            z = r.standard_normal((n, d))
            while (bad := np.abs(z) >= 0.95 * np.pi * f).any():
                z[bad] = r.standard_normal(bad.sum())
            W[:, 1::2], W[:, 2::2] = f * np.sin(z / f), f * (1 - np.cos(z / f))
            TRUTH = (z, f)
        else:    # lumpy: a flat sheet whose latent density clumps like the corpus's 64 k-means parts
            wts = np.bincount(kmeans(X0, 64), minlength=64) / N
            comp = r.choice(64, n, p=wts)
            mus, sds = r.standard_normal((64, d)), 0.5 * np.exp(0.3 * r.standard_normal(64))
            W[:, 1:] = mus[comp] + sds[comp, None] * r.standard_normal((n, d))
        R = R_data()
        # sigma_abs sets the RMS off-sheet displacement on the sphere, sigma_abs sqrt(Dc-1) / |y|, to sigma R; |y|^2 = 1 + s^2 d + Dc sigma_abs^2
        sa = lambda s: sig * R * np.sqrt((1 + s * s * d) / ((Dc - 1) - Dc * (sig * R) ** 2)) if sig > 0 else 0.0    # noqa: E731
        make, lo, hi = (lambda s: set_cloud(lin(axis, W, s, eps, sa(s)), codes)), 1e-3, 50.0
        info.update({'d': d, 'sigma_R': sig})
    elif kind == 'union':    # two flat sheets of N/2 rows, axes 40 deg apart, each spacing-matched alone
        cal = load(f'{BASE}/calib.json')
        d1, d2 = (cal['d_u1'], cal['d_u2']) if a[0] == 'ctrl' else [int(round(np.interp(float(m), cal['half_lb'], cal['half_d']))) for m in a]
        h, Dc, r = N // 2, 2 + d1 + d2, rng('z')
        pieces, info['pieces'] = [], []
        for sl, d, c0, ang in ((slice(0, h), d1, 2, 0.0), (slice(h, N), d2, 2 + d1, np.radians(40))):
            axis = np.zeros(Dc)
            axis[:2] = np.cos(ang), np.sin(ang)
            W = np.zeros((sl.stop - sl.start, Dc), f32)
            W[:, c0:c0 + d] = r.standard_normal((len(W), d))
            m = match(lambda s: set_cloud(lin(axis, W, s), ART[sl]), 1e-3, 50.0)
            pieces.append(X.copy())
            info['pieces'].append({'d': d, **m, 'lb': lb_median()})    # what each piece reads alone: t4's expected spread comes from these, not from the grid's d
        set_cloud(np.vstack(pieces))
        info['piece_recovered'] = float(np.mean((np.abs(X[:, 2 + d1:]).sum(1) > 0) == (np.arange(N) >= h)))    # piece 2 lives in its own coordinates
        info.update(describe())
        return info
    else:
        raise ValueError(name)
    info.update(match(make, lo, hi))
    info.update(describe())
    return info


# ---------- samples and ids ----------
def samples():
    """H: 2,000 works, one per artist; E: up to 1,000 per era, one per artist, H's draws first."""
    perm = np.random.default_rng(7).permutation(N)

    def firsts(c):
        return c[np.sort(np.unique(ART[c], return_index=True)[1])]
    H = firsts(perm)[:NH]
    E = {}
    for era, (lo, hi) in ERAS.items():
        inH = H[(YR[H] >= lo) & (YR[H] < hi)]
        c = firsts(perm[(YR[perm] >= lo) & (YR[perm] < hi)])
        E[era] = np.concatenate([inH, c[~np.isin(ART[c], ART[inH])]])[:NE]
    return H, E


def roles():
    """smooth_ids.csv resolved to corpus rows: 26 passers, 2 unfaithful passers, 484 non-passers, 3 starts, 9 oracle (target, start) pairs."""
    rr = list(csv.DictReader(open(os.path.join(os.environ['DUROOM_OUT'], 'inputs', 'smooth_ids.csv'))))
    cid = ids[rows]
    at = {int(i): j for j, i in enumerate(cid)}
    for r in rr:
        assert np.isin(int(r['id']), cid), r    # every id is a corpus work in this file
    out = {k: np.array([at[int(r['id'])] for r in rr if r['role'] == k]) for k in ('passer', 'unfaithful', 'nonpasser', 'start')}
    out['oracle'] = np.array([(at[int(r['id'])], at[int(r['partner'])]) for r in rr if r['role'] == 'oracle'])
    assert [len(out[k]) for k in ('passer', 'unfaithful', 'nonpasser', 'start', 'oracle')] == [26, 2, 484, 3, 9]
    pos = np.concatenate([out['passer'], out['unfaithful'], out['nonpasser']])
    assert ((YR[pos] >= 1900) & (YR[pos] <= 1969)).all() and len(np.unique(pos)) == 512
    return out


def site_key(i):
    return f'{NAMES[i]} {int(YR[i])}'


# ---------- T4 local dimension ----------
def t4():
    real = CLOUD == 'real'
    q = np.arange(N) if real else np.sort(np.random.default_rng(5).choice(N, min(NLB, N), replace=False))
    Do, Io, dup_o = others(q, 40)
    Da, _, dup_a = search(X[q], 40, selfr=q)
    m = {k: {'other': lb(Do, k), 'any': lb(Da, k)} for k in (10, 20, 40)}
    m20 = np.full(N, np.nan)
    m20[q] = m[20]['other']
    nb = Io[:, :10]
    extra = np.setdiff1d(nb, q)
    if len(extra):    # synthetic clouds: the queries' neighbours need their own estimate for coherence
        m20[extra] = lb(others(extra, 20)[0], 20)
    L, ld10 = np.log(m[20]['other']), np.log(Do[:, 9])
    nbm = np.median(np.log(m20[nb]), axis=1)
    sdn = {k: spread(np.log((k - 1) / np.random.default_rng(0).gamma(k - 1, 1.0, 100000))) for k in (10, 20, 40)}
    dec = lambda sd: float(np.sqrt(max(sd ** 2 - sdn[20] ** 2, 0)))    # noqa: E731

    def drop(thr, k=20):    # the estimate with neighbours closer than thr skipped (near-copies)
        j0 = (Do < thr).sum(1)
        idx = j0[:, None] + np.arange(k)
        ok = idx[:, -1] < Do.shape[1]
        T = np.take_along_axis(Do, np.minimum(idx, Do.shape[1] - 1), 1)
        return np.where(ok, lb(T, k), np.nan)
    near = {thr: drop(thr) for thr in (0.15, 0.25)}
    r = np.random.default_rng(17)
    bi = [r.integers(0, len(q), len(q)) for _ in range(NBOOT)]
    xr, yr = resid(L, ld10), resid(nbm, ld10)

    def groups(keys, minn):
        out = {}
        for g in np.unique(keys):
            sel = keys == g
            if sel.sum() >= minn and g not in ('', 'nan'):
                out[str(g)] = {'median': float(np.median(m[20]['other'][sel])), 'ci': boot(m[20]['other'][sel]), 'n': int(sel.sum())}
        return out
    era = np.select([YR[q] < 1900, YR[q] < 1960], ['before_1900', '1900_1959'], '1960_on')
    out = {'mle': {k: {w: {'p10_50_90': q3(v), 'iqr': float(np.subtract(*np.percentile(v, [75, 25])))} for w, v in mk.items()} for k, mk in m.items()},
           's': dec(spread(L)), 's_part': dec(spread(resid(L, ld10))), 'sd_noise': sdn, 'sd_r': spread(L),
           'coherence': {'raw': spearman(L, nbm), 'part': spearman(xr, yr), 'ci_raw': np.percentile([spearman(L[b], nbm[b]) for b in bi], [2.5, 97.5]).tolist(),
                         'ci_part': np.percentile([spearman(xr[b], yr[b]) for b in bi], [2.5, 97.5]).tolist()},
           'scale_ratio': float(np.median(m[40]['other'] / m[10]['other'])), 't1_nn': np.percentile(Do[:, 0], [1, 10, 50]).tolist(),
           'near_copy': {thr: {'p50': float(np.nanmedian(v)), 's': dec(spread(np.log(v))), 'n': int(np.isfinite(v).sum())} for thr, v in near.items()},
           'by_decade': groups((np.floor(YR[q] / 10) * 10).astype(int).astype(str), 200), 'by_region': groups(kmeans(X, 16)[q].astype(str), 1),
           'by_genre': groups(GENRE[q], 200), 'by_era': groups(era, 1), 'duplicates_skipped': {'other': dup_o, 'any': dup_a}, 'n': len(q)}
    if real:
        out['R'] = float(np.median(Do[:, 9]))
        out['R_deg'] = float(deg(out['R']))
        np.savez(f'{OUT}/works.npz', ids=ids[rows], d10_other=Do[:, 9], **{f'mle{k}_{w}': v for k, mk in m.items() for w, v in mk.items()})
    if CLOUD.startswith('union'):
        m1, m2 = (p['lb'] for p in res(CLOUD, SEED)['cloud']['pieces'])
        G = np.random.default_rng(0).gamma(19, 1.0, 100000)
        draws = np.log(19 * np.where(np.arange(100000) < 50000, m1, m2) / G)
        out['union'] = {'s_expected': dec(spread(draws)), 's': out['s'], 'coh_part': out['coherence']['part']}
    return out


# ---------- T1 thickness ----------
def _t1_chunk(job):
    Q, IA, IB, ks, ms = job
    nq, nk, nm = len(Q), len(ks), len(ms)
    o = {n: np.full((nq, nk, nm), np.nan) for n in ('in_share', 'out_share', 'glob_share')}
    o.update({n: np.full((nq, nk), np.nan) for n in ('dim90', 'lopsided', 'rkA', 'rkB')})
    o['width'] = np.full((nq, nk, 4), np.nan)
    for i, q in enumerate(Q):
        x = unitv(X[q])
        if IB is not None:
            G = VG[:max(ms)] - np.outer(VG[:max(ms)] @ x, x)
            Gq = np.linalg.qr(G.T)[0].T    # global top-m directions moved into the tangent space at x
            dirs = [VG[r] if r < len(VG) else None for r in (0, 49, 299)] + list(RDIR)
            dirs = [None if u is None else unitv(u - (u @ x) * x) for u in dirs]
        for a, k in enumerate(ks):
            if IA[i, k - 1] < 0:
                continue
            VA = logmap(x, X[IA[i, :k]])
            mu = VA.mean(0)
            o['lopsided'][i, a] = np.linalg.norm(mu) / np.sqrt((VA ** 2).sum(1).mean())
            VA -= mu
            _, s, Vt = np.linalg.svd(VA, full_matrices=False)
            cs = np.cumsum(s ** 2) / (s ** 2).sum()
            o['dim90'][i, a] = np.searchsorted(cs, 0.9) + 1
            for b, m in enumerate(ms):
                if m < k:
                    o['in_share'][i, a, b] = cs[min(m, len(cs)) - 1]
            if IB is None or IB[i, k - 1] < 0:
                continue
            o['rkA'][i, a], o['rkB'][i, a] = angles(x, X[[IA[i, k - 1], IB[i, k - 1]]])
            VB = logmap(x, X[IB[i, :k]])
            VB -= VB.mean(0)
            for b, m in enumerate(ms):
                if m < k:
                    o['out_share'][i, a, b] = share(Vt[:m], VB)
                    if m <= len(Gq):
                        o['glob_share'][i, a, b] = share(Gq[:m], VB)
            w = [np.nan if u is None else np.degrees(np.sqrt(np.mean((VA @ u) ** 2))) for u in dirs]
            o['width'][i, a] = w[0], w[1], w[2], np.median(w[3:])
    return o


def halves(split):
    """Artist half A for a split seed (a seeded random half of the artists)."""
    return (np.random.default_rng([split, zlib.crc32(b'halves')]).random(ART.max() + 1) < 0.5)[CODES]


def t1_core(Q, ks, ms, split, any_artist=False):
    gdirs()
    kmax = max(ks)
    if any_artist:
        IA, IB = search(X[Q], kmax, selfr=Q)[1], None
    else:
        A = halves(split)
        IA, IB = search(X[Q], kmax, A, CODES[Q])[1], search(X[Q], kmax, ~A, CODES[Q])[1]
    parts = pmap(_t1_chunk, [(Q[i:i + 25], IA[i:i + 25], None if IB is None else IB[i:i + 25], ks, ms) for i in range(0, len(Q), 25)])
    return {n: np.concatenate([p[n] for p in parts]) for n in parts[0]}


def t1():
    dc = d_ctrl()
    ms = sorted(set(MS + [dc]))
    bm = ms.index(dc)
    H, E = samples()
    ro = roles() if CLOUD == 'real' else None
    pos = np.concatenate([ro['passer'], ro['nonpasser']]) if ro else np.array([], int)
    Q = np.unique(np.concatenate([H, *E.values(), pos]))
    at = lambda r: np.searchsorted(Q, r)    # noqa: E731
    h, he = at(H), at(np.unique(np.concatenate([H, *E.values()])))
    out = {'k': KS, 'm_list': ms, 'd_ctrl': dc, 'n_H': len(H), 'n_E': {e: len(v) for e, v in E.items()}, 'split': {}}
    for split in ((0, 1, 2) if CLOUD == 'real' else (0,)):
        r = t1_core(Q, KS, ms, split)
        if split == 0:
            r0 = r
        loc = r['out_share'] / r['glob_share']
        s = {'k': {}}
        for a, k in enumerate(KS):
            s['k'][k] = {**{n: {m: q3(r[n][h, a, b]) for b, m in enumerate(ms)} for n in ('in_share', 'out_share', 'glob_share')},
                         'locality': {m: q3(loc[h, a, b]) for b, m in enumerate(ms)}, 'dim90': q3(r['dim90'][h, a]), 'lopsided': q3(r['lopsided'][h, a]),
                         'width_deg': {n: q3(r['width'][h, a, j]) for j, n in enumerate(('pc1', 'pc50', 'pc300', 'random'))},
                         'r_k_deg': q3(r['rkA'][h, a]), 'rk_ratio': q3(r['rkA'][h, a] / r['rkB'][h, a]), 'out_share_ci': boot(r['out_share'][h, a, bm])}
        s['by_era'] = {e: {'n': len(v), **{k: float(np.nanmedian(r['out_share'][at(v), a, bm])) for a, k in enumerate(KS)}} for e, v in E.items()}
        dec = np.floor(YR[Q[he]] / 10) * 10
        s['by_decade'] = [[int(d_), int((dec == d_).sum()), {k: float(np.nanmedian(r['out_share'][he[dec == d_], KS.index(k), bm])) for k in (100, 300)}]
                          for d_ in np.unique(dec) if (dec == d_).sum() >= 30]
        if ro:
            s['pos'] = {g: {k: float(np.nanmedian(r['out_share'][at(ro[g]), a, bm])) for a, k in enumerate(KS)} for g in ('passer', 'nonpasser')}
        out['split'][split] = s
    ra = t1_core(Q, KS, ms, 0, any_artist=True)
    out['anyartist'] = {k: {'in_share': {m: q3(ra['in_share'][h, a, b]) for b, m in enumerate(ms)}, 'dim90': q3(ra['dim90'][h, a])} for a, k in enumerate(KS)}
    Do = others(Q[he], 20)[0]    # the T4 estimate at the same works, for the dim90 cross-check
    lm, ld, d90 = np.log(lb(Do, 20)), np.log(Do[:, 9]), r0['dim90'][he, KS.index(100)]
    out['dim90_rho'] = {'raw': spearman(lm, d90), 'part': spearman(resid(lm, ld), resid(d90, ld))}
    return out


# ---------- T2 voids (data only) ----------
def lat_path(a, b, e, ts):
    """Latitude-preserving path: the angle to the axis e moves linearly, the direction orthogonal to it is slerped."""
    ta, tb = np.arccos(np.clip(a @ e, -1, 1)), np.arccos(np.clip(b @ e, -1, 1))
    wa, wb = unitv(a - np.cos(ta) * e), unitv(b - np.cos(tb) * e)
    w = slerp(wa, wb, ts)
    th = (1 - ts) * ta + ts * tb
    return np.cos(th)[:, None] * e + np.sin(th)[:, None] * w


def slerp(a, b, ts):
    om = np.arccos(np.clip(a @ b, -1, 1))
    if om < 1e-9:
        return np.repeat(a[None], len(ts), 0)
    return (np.sin((1 - ts) * om)[:, None] * a + np.sin(ts * om)[:, None] * b) / np.sin(om)


def classify(v, t90, t99):
    """INSIDE: no point above p99. HOLE: an interior point above p99 with a point below p90 on each side. EDGE: the end above p99. Else OTHER."""
    hi, lo = v > t99, v < t90
    for i in range(1, len(v) - 1):
        if hi[i] and lo[:i].any() and lo[i + 1:].any():
            return 'HOLE'
    return 'INSIDE' if not hi.any() else 'EDGE' if hi[-1] else 'OTHER'


def d10_kept(Dl, Il, keep):
    """The 10th nearest that survives a subsample, read from a top-50 list."""
    k = (Il >= 0) & keep[np.maximum(Il, 0)]
    hit = k & (np.cumsum(k, 1) == 10)
    return np.where(hit.any(1), Dl[np.arange(len(Dl)), np.argmax(hit, 1)], np.inf)


def _plant(job):
    """One INSIDE segment with a void planted off-grid: HOLE in 4/5 of the usable subsamples (under 10 survivors: skipped) per radius and keep fraction; returns (hole, skipped)."""
    pts, c, ex, seed = job
    Dp = np.sqrt(np.maximum(2 - 2 * (pts @ XPRE.T), 0)).astype(f64)
    dc = np.sqrt(np.maximum(2 - 2 * (XPRE @ c), 0))
    ok = ~np.isin(CODES[PRE], ex)
    u = np.random.default_rng(seed).random(len(dc))
    out, skipped = np.full((len(RHOS), len(KEEPS)), np.nan), np.zeros((len(RHOS), len(KEEPS)), int)
    for a, rho in enumerate(RHOS):
        for b, keep in enumerate(KEEPS):
            alive = ok & ~((dc < rho * RPRE) & (u >= keep))
            use = [j for j in range(20) if (alive & SUBP[j]).sum() >= 10]
            nh = sum(classify(np.partition(Dp[:, alive & SUBP[j]], 9, axis=1)[:, 9], *THR[j]) == 'HOLE' for j in use)
            skipped[a, b] = 20 - len(use)
            if use:
                out[a, b] = 5 * nh >= 4 * len(use)
    return out, skipped


def t2():
    global XPRE, SUBP, THR, RPRE, PRE
    assert CLOUD == 'real'
    ro = roles()
    pre = YR < 1900
    PRE = np.flatnonzero(pre)
    axis = unitv(X[PRE].mean(0, dtype=f64))
    r9 = np.random.default_rng(9)
    ref_pre, ref_all = r9.choice(PRE, min(NREF, len(PRE)), replace=False), r9.choice(N, min(NREF, N), replace=False)
    Dr, Ir, _ = others(ref_pre, 50, pre)
    ref10 = np.sort(Dr[:, 9])
    RPRE = float(np.median(ref10))
    all10 = np.sort(others(ref_all, 10)[0][:, 9])
    pre_ang = np.sort(angles(axis, X[PRE]))
    pct = lambda v, ref: 100 * np.searchsorted(ref, v, 'right') / len(ref)    # noqa: E731
    SUB = np.random.default_rng(11).random((20, N)) < 0.9
    t90, t99 = np.percentile(ref10, [90, 99])
    THR = [np.percentile(d10_kept(Dr, Ir, SUB[j]), [90, 99]) for j in range(20)]
    T = np.concatenate([ro['passer'], ro['unfaithful'], ro['nonpasser']])
    Dt, It, _ = others(T, 50, pre)
    dall = others(T, 10)[0][:, 9]
    rads = [0.9 * RPRE, RPRE, 1.1 * RPRE]
    npre = np.vstack(pmap(_count_block, [(X[T[b:b + 500]], pre, CODES[T[b:b + 500], None], rads) for b in range(0, len(T), 500)]))
    p_pre, p_ang, p_all = pct(Dt[:, 9], ref10), pct(angles(axis, X[T]), pre_ang), pct(dall, all10)
    g = {'passers': np.arange(26), 'unfaithful': np.arange(26, 28), 'all_nonpassers': np.arange(28, 512)}
    na, used, matched = angles(axis, X[T[28:]]), np.zeros(484, bool), []
    for i in range(26):    # 5 non-passers nearest in angle to the PRE axis per passer, without replacement
        o = [j for j in np.argsort(np.abs(na - angles(axis, X[T[i]][None])[0])) if not used[j]][:5]
        used[o] = True
        matched += [28 + j for j in o]
    g['matched'] = np.array(matched)
    read = lambda i: {'density': 'INSIDE' if p_pre[i] <= 90 else 'THIN' if p_pre[i] <= 99 else 'SPARSE',    # noqa: E731
                      'axis': 'within the cap' if p_ang[i] <= 90 else 'beyond the cap', 'filled': bool(p_all[i] < 90)}
    sites = {site_key(T[i]): {'d_pre10_pct': p_pre[i], 'angle_pct': p_ang[i], 'd_all10_pct': p_all[i], 'n_pre': npre[i], 'reading': read(i),
                              'group': 'passer' if i < 26 else 'unfaithful'} for i in range(28)}
    cliff = lambda a, b: float(np.sign(a[:, None] - b[None, :]).mean())    # noqa: E731
    fisher = lambda a, b, j: float(stats.fisher_exact([[(npre[a, j] == 0).sum(), (npre[a, j] > 0).sum()], [(npre[b, j] == 0).sum(), (npre[b, j] > 0).sum()]], 'greater')[1])    # noqa: E731
    tests = {ref: {'mannwhitney_p': float(stats.mannwhitneyu(p_pre[g['passers']], p_pre[g[ref]], alternative='greater')[1]),
                   'cliffs_delta': cliff(p_pre[g['passers']], p_pre[g[ref]]), 'fisher_p': [fisher(g['passers'], g[ref], j) for j in range(3)]} for ref in ('matched', 'all_nonpassers')}
    groups = {k: {'d_pre10_pct': np.percentile(p_pre[v], [10, 25, 50, 75, 90]).tolist(), 'angle_pct': np.percentile(p_ang[v], [10, 50, 90]).tolist(),
                  'n_pre_zero_share': (npre[v] == 0).mean(0).tolist(), 'n': len(v)} for k, v in g.items()}
    # segments: (start row, end row, kind, group, end index); oracle lines by slerp, the rest latitude-preserving
    ends = np.concatenate([g['passers'], g['unfaithful'], g['matched']])
    egrp = ['passers'] * 26 + ['unfaithful'] * 2 + ['matched'] * len(g['matched'])
    perm = r9.permutation(PRE)
    rnd = perm[np.sort(np.unique(ART[perm], return_index=True)[1])][:10]
    segs = [(s, T[e], kind, egrp[n], n) for n, e in enumerate(ends) for kind, ss in (('nearest', It[e, :10]), ('random', rnd)) for s in ss]
    segs += [(s, t, 'oracle', 'oracle', -1) for t, s in ro['oracle']]
    while sum(sg[2] == 'base' for sg in segs) < NPAIR:
        a_, b_ = r9.choice(PRE, 2)
        if ART[a_] != ART[b_]:
            segs.append((a_, b_, 'base', 'pre_pairs', -1))
    ts = np.linspace(0, 1, 13)
    path = lambda a_, b_, kind, t=ts: (slerp if kind == 'oracle' else lambda u, v, w: lat_path(u, v, axis, w))(unitv(X[a_]), unitv(X[b_]), t)    # noqa: E731
    P = np.vstack([path(a_, b_, kind) for a_, b_, kind, _, _ in segs])
    ex = np.repeat(np.array([[CODES[a_], CODES[b_]] for a_, b_, _, _, _ in segs]), 13, 0)
    Dl, Il, _ = search(P, 50, pre, ex)
    full = Dl[:, 9].reshape(len(segs), 13)
    subs = np.stack([d10_kept(Dl, Il, SUB[j]).reshape(len(segs), 13) for j in range(20)])
    nhole = np.array([sum(classify(subs[j, s], *THR[j]) == 'HOLE' for j in range(20)) for s in range(len(segs))])
    cls = [('HOLE' if nhole[s] >= 16 else c if c != 'HOLE' else 'OTHER') for s, c in enumerate(classify(full[s], t90, t99) for s in range(len(segs)))]
    ppath = pct(full, ref10)
    thin = np.array([ts[np.argmax(p > 90)] if (p > 90).any() else np.nan for p in ppath])
    length = np.array([angles(unitv(X[a_]), X[b_][None])[0] for a_, b_, _, _, _ in segs])
    kinds = np.array([sg[2] for sg in segs])
    grps = np.array([sg[3] for sg in segs])
    cls = np.array(cls)
    sel = lambda k, gr: (kinds == k) & (grps == gr)    # noqa: E731
    cells = [(k, gr) for k in ('nearest', 'random') for gr in ('passers', 'unfaithful', 'matched')] + [('base', 'pre_pairs')]
    profiles = {f'{k}|{gr}': {'median': np.median(ppath[sel(k, gr)], 0).tolist(), 'p25': np.percentile(ppath[sel(k, gr)], 25, 0).tolist(),
                              'p75': np.percentile(ppath[sel(k, gr)], 75, 0).tolist(), 'n': int(sel(k, gr).sum())} for k, gr in cells}
    classes = {f'{k}|{gr}': {c: int((cls[sel(k, gr)] == c).sum()) for c in ('INSIDE', 'HOLE', 'EDGE', 'OTHER')} for k, gr in cells}
    t_thin = {f'{k}|{gr}': {'median': float(np.nanmedian(thin[sel(k, gr)])) if np.isfinite(thin[sel(k, gr)]).any() else None,
                            'share_thin': float(np.isfinite(thin[sel(k, gr)]).mean())} for k, gr in cells}
    oracle = [{'line': f'{site_key(s)} -> {site_key(t)}', 'pct': ppath[n].tolist(), 'class': cls[n]} for n, (s, t, kind, _, _) in enumerate(segs) if kind == 'oracle']
    XPRE, SUBP = X[PRE], SUB[:, PRE]
    r13 = np.random.default_rng(13)
    bins = np.digitize(length, BINS) - 1
    jobs, jb = [], []
    for b in range(len(BINS) - 1):
        cand = np.flatnonzero(np.isin(kinds, ['nearest', 'random']) & (cls == 'INSIDE') & (bins == b))
        for s in r13.permutation(cand)[:NPLANT]:
            t = r13.uniform(0.15, 0.85)
            while np.abs(t - ts).min() <= 0.02:
                t = r13.uniform(0.15, 0.85)
            a_, b_ = segs[s][:2]
            jobs.append((P[s * 13:(s + 1) * 13], path(a_, b_, 'nearest', np.array([t]))[0].astype(f32), [CODES[a_], CODES[b_]], int(r13.integers(1 << 31))))
            jb.append(b)
    pl = pmap(_plant, jobs)
    hole, skip = (np.array([q[i] for q in pl]) for i in (0, 1)) if jobs else (np.zeros((0, len(RHOS), len(KEEPS))),) * 2
    jb = np.array(jb)
    blab = [f'{BINS[b]}-{BINS[b + 1]}' for b in range(len(BINS) - 1)]
    usable = lambda b, a, c: hole[jb == b, a, c][np.isfinite(hole[jb == b, a, c])]    # noqa: E731
    power = {blab[b]: {rho: {keep: {'share': float(usable(b, a, c).mean()) if len(usable(b, a, c)) else None, 'n': int((jb == b).sum()), 'n_usable': len(usable(b, a, c)),
                                    'subsets_skipped_mean': float(skip[jb == b, a, c].mean()) if (jb == b).any() else None}
                             for c, keep in enumerate(KEEPS)} for a, rho in enumerate(RHOS)} for b in range(len(BINS) - 1)}
    rho80 = {blab[b]: next((rho for a, rho in enumerate(RHOS) if len(usable(b, a, 0)) and usable(b, a, 0).mean() >= 0.8), None) for b in range(len(BINS) - 1)}
    for i in range(28):    # per target: majority class over its 10 nearest-start segments; HOLE only where the bin has the power
        s = np.flatnonzero((kinds == 'nearest') & (np.array([sg[4] for sg in segs]) == i))
        c, n_ = np.unique(cls[s], return_counts=True)
        b = blab[int(np.digitize(np.median(length[s]), BINS) - 1)]
        sites[site_key(T[i])].update({'class': c[np.argmax(n_)], 'length_bin': b, 'rho80': rho80[b],
                                      'hole_reportable': rho80[b] is not None and rho80[b] <= 1.4})
    def card_of(i):    # a missing image must not cost the numbers above
        try:
            return card(rows[T[i]], OUT)
        except Exception as e:
            return {'id': int(ids[rows[T[i]]]), 'error': repr(e)}
    json.dump(clean({site_key(T[i]): card_of(i) for i in range(28)}), open(f'{OUT}/cards.json', 'w'), indent=1)
    return {'R_pre': RPRE, 'R_all': float(np.median(all10)), 'pre_p90_p99': [t90, t99], 'angle_p90_p99': np.percentile(pre_ang, [90, 99]).tolist(),
            'sites': sites, 'groups': groups, 'tests': tests, 'profiles': profiles, 'classes': classes, 't_thin': t_thin, 'oracle_lines': oracle,
            'base_rate_hole': float((cls[kinds == 'base'] == 'HOLE').mean()), 'power': power, 'rho80': rho80, 'n_segments': len(segs),
            'note': 'matched non-passers share the passers\' angle to the PRE axis'}


# ---------- T3 turning ----------
def true_frame(row, x):
    z, f = TRUTH[0][row], TRUTH[1]
    d = len(z)
    F = np.zeros((d, X.shape[1]))
    F[np.arange(d), 1 + 2 * np.arange(d)], F[np.arange(d), 2 + 2 * np.arange(d)] = np.cos(z / f), np.sin(z / f)
    F -= np.outer(F @ x, x)
    return np.linalg.qr(F.T)[0].T


def _t3_chunk(job):
    Q, IqA, IqB, Pt, IpB, ms = job
    nq, npn, mm = len(Q), Pt.shape[1], max(ms)
    o = {'ov': np.full((nq, npn, len(ms)), np.nan), 'ceil': np.full((nq, len(ms)), np.nan), 'ang': np.full((nq, npn), np.nan),
         'inside': np.full((nq, npn), np.nan), 'rfit': np.full(nq, np.nan), 'ovt': np.full((nq, npn), np.nan)}
    for i, q in enumerate(Q):
        x = unitv(X[q])
        UA, UB = plane(x, X[IqA[i]])[2][:mm], plane(x, X[IqB[i]])[2][:mm]
        o['ceil'][i] = [overlap(UA[:m], UB[:m]) for m in ms]
        o['rfit'][i] = angles(x, X[IqA[i, -1]][None])[0]
        Tq = true_frame(q, x) if TRUTH else None
        for j, p in enumerate(Pt[i]):
            y = unitv(X[p])
            Up = plane(y, X[IpB[i, j]])[2]
            Tr = transport(UA, x, y)
            o['ov'][i, j] = [overlap(Tr[:m], Up[:m]) for m in ms]
            o['ang'][i, j] = angles(x, y[None])[0]
            o['inside'][i, j] = np.mean(angles(x, X[IpB[i, j]]) <= o['rfit'][i])
            if TRUTH:
                o['ovt'][i, j] = overlap(transport(Tq, x, y), true_frame(p, y))
    return o


def delta_star(a, t, tr):
    """The angle at which the median tau has fallen halfway from 1 to tau_rand (JSON only)."""
    h = (1 + tr) / 2
    if t[0] <= h:
        return None
    for j in range(1, len(t)):
        if t[j] <= h:
            return float(a[j - 1] + (h - t[j - 1]) * (a[j] - a[j - 1]) / (t[j] - t[j - 1]))
    return None


def t3():
    dc = d_ctrl()
    ms = sorted(set([dc, 5, 10, 20]))
    kf = max(100, 3 * dc)
    Q = samples()[0][:NT3]
    nq = len(Q)
    out = {'m': ms, 'k': kf, 'f': {m: m / (D - 1) for m in ms}, 'ranks': RANKS + ['random'], 'split': {}}
    for split in ((0, 1, 2) if CLOUD == 'real' else (0,)):
        A = halves(split)
        IqA, IqB = search(X[Q], kf, A, CODES[Q])[1], search(X[Q], kf, ~A, CODES[Q])[1]
        Ip = others(Q, max(RANKS))[1]
        r = np.random.default_rng([split, 3])
        rand = np.stack([(c := r.integers(0, N, 60))[CODES[c] != CODES[q]][:10] for q in Q])
        Pt = np.concatenate([Ip[:, [k - 1 for k in RANKS]], rand], 1)
        IpB = search(X[Pt.ravel()], kf, ~A, np.stack([np.repeat(CODES[Q], Pt.shape[1]), CODES[Pt.ravel()]], 1))[1].reshape(nq, Pt.shape[1], kf)
        parts = pmap(_t3_chunk, [(Q[i:i + 10], IqA[i:i + 10], IqB[i:i + 10], Pt[i:i + 10], IpB[i:i + 10], ms) for i in range(0, nq, 10)])
        o = {n: np.concatenate([p[n] for p in parts]) for n in parts[0]}
        cols = [[j] for j in range(len(RANKS))] + [list(range(len(RANKS), Pt.shape[1]))]    # the random partners are one column
        s = {'m': {}, 'footprint': {'r_fit_deg': float(np.median(o['rfit'])), 'inside_share': [float(np.nanmedian(o['inside'][:, c])) for c in cols]},
             'rank_angle_deg': [float(np.nanmedian(o['ang'][:, c])) for c in cols]}
        for b, m in enumerate(ms):
            f, cm = m / (D - 1), float(np.nanmedian(o['ceil'][:, b]))
            tau = (o['ov'][:, :, b] - f) / (cm - f)
            curve = [[rk, float(np.nanmedian(o['ang'][:, c])), *np.nanpercentile(tau[:, c], [50, 25, 75]).tolist(), int(np.isfinite(tau[:, c]).sum())] for rk, c in zip(out['ranks'], cols)]
            tr = curve[-1][2]
            s['m'][m] = {'ceiling_median': cm, 'curve': curve, 'tau_rand': tr, 'tau_rand_ci': boot(np.nanmedian(tau[:, cols[-1]], 1)),
                         'delta_star_deg': delta_star([c[1] for c in curve[:-1]], [c[2] for c in curve[:-1]], tr)}
        if TRUTH:
            tt = (o['ovt'] - dc / (D - 1)) / (1 - dc / (D - 1))
            s['truth'] = {'curve': [[rk, float(np.nanmedian(tt[:, c]))] for rk, c in zip(out['ranks'], cols)], 'tau_rand': float(np.nanmedian(tt[:, cols[-1]]))}
        out['split'][split] = s
    return out


# ---------- AX: the artist 2x2 ----------
def ax_half(split, by_artist):
    """The fit half of the rows: T1's artist half, or a seeded random half of the rows (work split). The test half is its complement."""
    return halves(split) if by_artist else np.random.default_rng([split, zlib.crc32(b'workhalves')]).random(len(X)) < 0.5


def _ax_chunk(job):
    """T1's out_share for one cell (same log map, centring and share as _t1_chunk), with radii, own-artist shares and fit/test leakage."""
    Q, IA, IB, ks, m = job
    o = {n: np.full((len(Q), len(ks)), np.nan) for n in ('share', 'in_share', 'rfit', 'rtest', 'own_fit', 'own_test', 'same', 'near', 'share_nc')}
    for i, q in enumerate(Q):
        x = unitv(X[q])
        for a, k in enumerate(ks):
            if IA[i, k - 1] < 0 or IB[i, k - 1] < 0:
                continue
            ia, ib = IA[i, :k], IB[i, :k]
            VA = logmap(x, X[ia])
            VA -= VA.mean(0)
            _, s, Vt = np.linalg.svd(VA, full_matrices=False)
            o['in_share'][i, a] = (s[:m] ** 2).sum() / (s ** 2).sum()
            VB = logmap(x, X[ib])
            VB -= VB.mean(0)
            o['share'][i, a] = share(Vt[:m], VB)
            o['rfit'][i, a], o['rtest'][i, a] = angles(x, X[[ia[-1], ib[-1]]])
            o['own_fit'][i, a], o['own_test'][i, a] = (CODES[ia] == CODES[q]).mean(), (CODES[ib] == CODES[q]).mean()
            FA, FB = X[ia].astype(f64), X[ib].astype(f64)
            d = np.sqrt(np.maximum(((FB ** 2).sum(1)[:, None] + (FA ** 2).sum(1)[None] - 2 * FB @ FA.T).min(1), 0))    # each test neighbour's exact chord to its nearest fit neighbour
            o['same'][i, a], o['near'][i, a] = (d < 1e-4).mean(), (d < 0.15).mean()
            if (d >= 0.15).sum() >= 10:    # the share without test neighbours that have a near-copy in the fit set (the fit is fixed, so 10 rows are enough)
                VC = logmap(x, X[ib[d >= 0.15]])
                VC -= VC.mean(0)
                o['share_nc'][i, a] = share(Vt[:m], VC)
    return o


def ax_cell(split, excl, by_artist, H, Q3, rand, kf, dc):
    """One cell: per-anchor arrays for the share (anchors H) and for the orientation (anchors Q3, partners rand)."""
    A = ax_half(split, by_artist)
    IA, IB = (search(X[H], max(AXK), h, CODES[H] if excl else None, H)[1] for h in (A, ~A))    # the anchor's own row is always out (selfr)
    parts = pmap(_ax_chunk, [(H[i:i + 25], IA[i:i + 25], IB[i:i + 25], AXK, dc) for i in range(0, len(H), 25)])
    c = {n: np.concatenate([p[n] for p in parts]) for n in parts[0]}
    c['fit_side'] = A[H]    # the anchor's row is in the fit half (in an artist split: its whole artist is)
    IqA, IqB = (search(X[Q3], kf, h, CODES[Q3] if excl else None, Q3)[1] for h in (A, ~A))
    assert (IqA >= 0).all() and (IqB >= 0).all()
    P, qq = rand.ravel(), np.repeat(Q3, rand.shape[1])
    I = search(X[P], kf + 1, ~A, np.stack([CODES[qq], CODES[P]], 1) if excl else None, P)[1]    # the partner's own row out; one spare for the anchor's
    assert (I >= 0).all()
    IpB = np.stack([row[row != q][:kf] for row, q in zip(I, qq)]).reshape(len(Q3), rand.shape[1], kf)    # the anchor's row out of its partners' fits
    parts = pmap(_t3_chunk, [(Q3[i:i + 10], IqA[i:i + 10], IqB[i:i + 10], rand[i:i + 10], IpB[i:i + 10], [dc]) for i in range(0, len(Q3), 10)])
    c.update(ov=np.concatenate([p['ov'] for p in parts])[:, :, 0], ceil=np.concatenate([p['ceil'] for p in parts])[:, 0],
             ang=np.concatenate([p['ang'] for p in parts]), rfit3=np.concatenate([p['rfit'] for p in parts]))
    if SMOKE:    # the smoke checks read the neighbour rows themselves
        c.update(IA=IA, IB=IB, IqA=IqA, IqB=IqB, IpB=IpB, rand=rand)
    return c


def ax_summary(c, ea, sel, f):
    """A cell's numbers over the anchors in sel (a mask over H; its head is the mask over Q3), with paired differences from EA."""
    pair = lambda d: {'median': float(np.nanmedian(d)) if np.isfinite(d).any() else None, 'ci': boot(d), 'n': int(np.isfinite(d).sum())}    # noqa: E731
    med = lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else None    # noqa: E731
    s3 = sel[:len(c['ceil'])]
    out = {'n': int(sel.sum()), 'n3': int(s3.sum()), 'k': {}}
    lr, sv = np.log(np.nanmedian(c['rfit'][sel], 0)), np.nanmedian(c['share'][sel], 0)
    for a, k in enumerate(AXK):
        rk = float(np.nanmedian(ea['rfit'][sel, a]))    # EA's median fit radius at this k, over the anchors read
        out['k'][k] = {'share': q3(c['share'][sel, a]), 'share_ci': boot(c['share'][sel, a]), 'share_nc': q3(c['share_nc'][sel, a]), 'in_share': med(c['in_share'][sel, a]),
                       'r_fit_deg': med(c['rfit'][sel, a]), 'r_test_deg': med(c['rtest'][sel, a]),
                       'own_fit': float(np.nanmean(c['own_fit'][sel, a])), 'own_test': float(np.nanmean(c['own_test'][sel, a])),
                       'same_image': float(np.nanmean(c['same'][sel, a])), 'near_copy': float(np.nanmean(c['near'][sel, a])),
                       'minus_EA': pair((c['share'] - ea['share'])[sel, a]), 'nc_minus_share': pair((c['share_nc'] - c['share'])[sel, a]),
                       'share_at_EA_radius': float(np.interp(np.log(rk), lr, sv)), 'EA_radius_deg': rk, 'radius_in_ladder': bool(lr[0] <= np.log(rk) <= lr[-1])}
    cm = float(np.nanmedian(c['ceil'][s3]))
    tau = (c['ov'][s3] - f) / (cm - f)    # t3()'s scaling: the median own-site ceiling of the anchors read
    out['orientation'] = {'raw_overlap': float(np.nanmedian(c['ov'][s3])), 'raw_overlap_ci': boot(np.nanmedian(c['ov'][s3], 1)), 'ceiling_median': cm,
                          'tau_rand': float(np.nanmedian(tau)), 'tau_rand_ci': boot(np.nanmedian(tau, 1)), 'partner_angle_deg': float(np.nanmedian(c['ang'][s3])),
                          'r_fit_deg': float(np.nanmedian(c['rfit3'][s3])), 'raw_minus_EA': pair((np.nanmedian(c['ov'], 1) - np.nanmedian(ea['ov'], 1))[s3]),
                          'ceiling_minus_EA': pair((c['ceil'] - ea['ceil'])[s3])}
    return out


def ax():
    dc = d_ctrl()
    H = samples()[0]
    Q3, kf, f = H[:NT3], max(100, 3 * dc), dc / (D - 1)
    out = {'k': AXK, 'm': dc, 'kf': kf, 'n_H': len(H), 'n_Q3': len(Q3), 'cells': {n: {'artist_excluded': e, 'split_by_artist': b} for n, (e, b) in CELLS.items()}, 'split': {}}
    keep = {}
    pair = lambda d: {'median': float(np.nanmedian(d)), 'ci': boot(d)}    # noqa: E731
    for split in ((0, 1, 2) if CLOUD == 'real' else (0,)):
        r = np.random.default_rng([split, 3])
        rand = np.stack([(c := r.integers(0, N, 60))[CODES[c] != CODES[q]][:10] for q in Q3])    # t3()'s random partners, the same in every cell
        cell = {n: ax_cell(split, e, b, H, Q3, rand, kf, dc) for n, (e, b) in CELLS.items()}
        every = np.ones(len(H), bool)
        s = {n: {'all': ax_summary(c, cell['EA'], every, f)} for n, c in cell.items()}
        for side, sel in (('artist_in_fit', cell['IA']['fit_side']), ('artist_in_test', ~cell['IA']['fit_side'])):    # IA is asymmetric: read by side
            s['IA'][side] = ax_summary(cell['IA'], cell['EA'], sel, f)
        ovq = {n: np.nanmedian(c['ov'], 1) for n, c in cell.items()}
        sides = {'': every, '_artist_in_fit': cell['IA']['fit_side'], '_artist_in_test': ~cell['IA']['fit_side']}
        s['effects'] = {name + ('_pooled' if 'IA' in (x, y) and not tag else tag):
                        {'share': {k: pair((cell[x]['share'] - cell[y]['share'])[sel, a]) for a, k in enumerate(AXK)}, 'raw_overlap': pair((ovq[x] - ovq[y])[sel[:len(Q3)]])}
                        for name, (x, y) in {'exclusion_work_split': ('IW', 'EW'), 'exclusion_artist_split': ('IA', 'EA'), 'split_excluded': ('EW', 'EA'),
                                             'split_included': ('IW', 'IA'), 'IW_minus_EA': ('IW', 'EA')}.items()
                        for tag, sel in sides.items() if not tag or 'IA' in (x, y)}    # effects through IA are read by side; the pooled one is secondary
        out['split'][split] = s
        keep.update({f's{split}_{n}_{v}': c[v] for n, c in cell.items() for v in c})
        np.savez_compressed(f'{OUT}/ax.npz', **keep)    # saved after every split, so a later failure keeps the earlier ones
        merge(f'{OUT}/ax.json', {'ax': out})
        print('ax split', split, {n: [(s[n]['all']['k'][k]['share'] or [None] * 3)[1] for k in (100, 300)] + [s[n]['all']['orientation']['tau_rand']] for n in CELLS}, flush=True)
    return out


# ---------- calibration and grid (CLOUD real) ----------
def lb_median():
    q = np.random.default_rng(5).choice(len(X), min(NLB, len(X)), replace=False)
    return float(np.median(lb(others(q, 20)[0], 20)))


def calib():
    target = res('real')['t4']['mle']['20']['other']['p10_50_90'][1]

    def ladder(ds, half):
        rows_ = []
        for d in ds:
            try:
                info = build(f'sheet:{d}:0', half)
                rows_.append({'d': d, 'lb': lb_median(), **info})
            except Infeasible:
                rows_.append({'d': d, 'feasible': False})
        return rows_

    def read(rows_, m):
        ok = [r for r in rows_ if 'lb' in r]
        lbs, ds = np.array([r['lb'] for r in ok]), np.array([r['d'] for r in ok])
        o = np.argsort(lbs)
        if m > lbs.max():    # past the grid: linear extrapolation from the last two rungs, flagged
            return float(ds[o][-1] + (m - lbs[o][-1]) * (ds[o][-1] - ds[o][-2]) / (lbs[o][-1] - lbs[o][-2])), 'extrapolated'
        return float(np.interp(m, lbs[o], ds[o])), 'below_feasible' if m < lbs.min() else ''
    full = ladder(CAL_D, False)
    if target > max([r['lb'] for r in full if 'lb' in r], default=0):
        full += ladder([56, 64], False)
    dval, flag = read(full, target)
    out = {'m_data': target, 'R': R_data(), 'grid': full, 'flags': [flag] if flag else []}
    out['d_ctrl'] = 20 if flag == 'below_feasible' else int(round(dval))
    half = ladder(CAL_DU, True)
    (u1, f1), (u2, f2) = read(half, target), read(half, 2 * target)
    out.update({'half_grid': half, 'd_u1': int(round(u1)), 'd_u2': int(round(u2)), 'half_lb': [r['lb'] for r in half if 'lb' in r], 'half_d': [r['d'] for r in half if 'lb' in r]})
    out['flags'] += [f'd_u1 {f1}'] * bool(f1) + [f'd_u2 {f2}'] * bool(f2)
    merge(f'{BASE}/calib.json', out)
    build('real')
    return out


def grid():
    dc = d_ctrl()
    for d in GRID_D:
        for sg in GRID_S:
            path = f'{BASE}/grid/sheet_{d}_{sg}_s0/smooth.json'
            try:
                info = build(f'sheet:{d}:{sg}')
            except Infeasible:
                merge(path, {'grid': {'d': d, 'sigma_R': sg, 'feasible': False}})
                continue
            Qg = np.random.default_rng(23).choice(N, min(NGRIDQ, N), replace=False)
            r = t1_core(Qg, [100, 300], [dc], 0)
            merge(path, {'cloud': info, 'grid': {'d': d, 'sigma_R': sg, 'feasible': True, 'lb_median': lb_median(),
                                                 's_out': {k: float(np.nanmedian(r['out_share'][:, a, 0])) for a, k in enumerate((100, 300))}}})
    build('real')
    return {'cells': len(GRID_D) * len(GRID_S)}


# ---------- verdict ----------
def so(r, k, split=0):
    return r['t1']['split'][str(split)]['k'][str(k)]['out_share'][str(d_ctrl())][1]


def sigma_eq(k):
    """The data's s_out read on the sheet ladder (sigma = 0 gives 1): an upper bound on off-sheet noise, in units of R."""
    lad = [(0.0, 1.0)] + [(s, so(res(f'sheet:ctrl:{s}'), k)) for s in (0.25, 0.5, 0.65)]
    sig, val = np.array(lad).T
    x = so(res('real'), k)
    return {'value': float(np.interp(x, val[::-1], sig[::-1])), 'beyond_ladder': bool(x < val.min()), 'ladder_s_out': val.tolist(),
            'rung_lb': [None] + [res(f'sheet:ctrl:{s}')['t4']['mle']['20']['other']['p10_50_90'][1] if 't4' in res(f'sheet:ctrl:{s}') else None for s in (0.25, 0.5, 0.65)],
            'rung_cap': [None] + [res(f'sheet:ctrl:{s}')['cloud']['cap_deg'] for s in (0.25, 0.5, 0.65)]}


def sigma_eq_t3():
    """sheet:ctrl:eq for T3: sigma_eq at k = 100 (the k nearest T3's fitting ball), clipped to [0, 0.65], rounded to 0.05."""
    return round(float(np.clip(sigma_eq(100)['value'], 0, 0.65)) / 0.05) * 0.05


def rule(fn):
    try:
        return fn()
    except Exception as e:    # also scipy's QhullError (too few non-collinear grid cells): one rule must not take the others' verdicts with it
        return {'outcome': 'pending', 'missing': repr(e)}


def v_t1():
    real, g0, g1, m64 = res('real'), res('gauss', 0), res('gauss', 1), res('mix:64')
    sg = {k: (so(g0, k) + so(g1, k)) / 2 for k in (100, 300)}
    p = lambda s, k: (s - sg[k]) / (1 - sg[k])    # noqa: E731
    c11 = {k: p(so(res('curved:ctrl:4'), k), k) for k in (100, 300)}
    c13 = {k: p(so(res('sheet:ctrl:0.25'), k), k) for k in (100, 300)}
    c12 = {k: abs(so(g0, k) - so(g1, k)) for k in (100, 300)}
    ctrl = {'C1.1': {'p': c11, 'pass': all(v >= 0.85 for v in c11.values())}, 'C1.2': {'diff': c12, 'pass': all(v <= 0.02 for v in c12.values())},
            'C1.3': {'p': c13, 'pass': all(v >= 0.75 for v in c13.values())}}
    bands = {k: (0.85, 0.15) for k in (100, 300)}
    if not ctrl['C1.3']['pass']:    # re-read of the bands from the ladder
        bands = {k: (c13[k] - 0.03, p(so(res('sheet:ctrl:0.65'), k), k) + 0.15) for k in (100, 300)}
    pos = {sp: {k: p(so(real, k, sp), k) for k in (100, 300)} for sp in (0, 1, 2)}
    band = lambda v, k: 'THIN' if v >= bands[k][0] else 'THICK' if v <= bands[k][1] else 'INTERMEDIATE'    # noqa: E731
    per = {sp: {k: band(v, k) for k, v in pk.items()} for sp, pk in pos.items()}
    per_split = {sp: (b[100] if b[100] == b[300] else 'INTERMEDIATE (k 100 and 300 disagree)') for sp, b in per.items()}
    outcome = per_split[0] if len(set(per_split.values())) == 1 else 'SPLIT-SENSITIVE'
    lumps = {sp: {k: so(real, k, sp) - so(m64, k) for k in (100, 300)} for sp in (0, 1, 2)}
    beyond = all(v >= 0.05 for d_ in lumps.values() for v in d_.values())
    rk = {k: real['t1']['split']['0']['k'][str(k)]['r_k_deg'][1] for k in (100, 300)}

    def at_rk(r):    # a null's s_out interpolated on log r_k at the data's r_k
        kk = [k for k in r['t1']['k'] if r['t1']['split']['0']['k'][str(k)]['out_share'][str(d_ctrl())]]    # k > d_ctrl only
        lr, sv = np.log([r['t1']['split']['0']['k'][str(k)]['r_k_deg'][1] for k in kk]), [so(r, k) for k in kk]
        return {k: float(np.interp(np.log(rk[k]), lr, sv)) for k in (100, 300)}
    return {'outcome': outcome if ctrl['C1.1']['pass'] else 'UNDERPOWERED (C1.1 failed)', 'lumps': 'BEYOND LUMPS' if beyond else 'as thin as the data\'s own lumps give',
            'position_p': pos, 'bands': bands, 'bands_reread': not ctrl['C1.3']['pass'], 'per_split': per, 's_out_data_minus_mix64': lumps, 's_gauss': sg,
            'sigma_eq': {k: rule(lambda k=k: sigma_eq(k)) for k in (100, 300)}, 'out_share_at_data_rk': {n: rule(lambda n=n: at_rk(res(n))) for n in ('gauss', 'mix:64')},
            'controls': ctrl}


def v_t3(t1_outcome):
    dc = str(d_ctrl())
    tr = lambda r, sp=0: r['t3']['split'][str(sp)]['m'][dc]['tau_rand']    # noqa: E731
    g0, g1 = res('gauss', 0), res('gauss', 1)
    tg = (tr(g0) + tr(g1)) / 2
    dt = max(0.02, 3 * abs(tr(g0) - tr(g1)))
    teq, t64, t50 = tr(res('sheet:ctrl:eq')), tr(res('mix:64')), tr(res('sheet:ctrl:0.5'))
    cv = {f: tr(res(f'curved:ctrl:{f}')) for f in (4, 2, 1)}
    ctrl = {'C3.2': {'tau_rand': t50, 'pass': t50 >= 1 - dt}, 'C3.3': {'tau_rand': cv, 'pass': cv[4] - cv[2] >= dt and cv[2] - cv[1] >= dt},
            'C3.4': {'pass': all(cv[f] <= min(tg, teq) - dt for f in (2, 1))}}
    real = res('real')
    per = {}
    for sp in (0, 1, 2):
        t = tr(real, sp)
        per[sp] = ('TURNING' + (' BEYOND LUMPS' if t <= t64 - dt else ' (turning that the data\'s lumps give)')) if t <= min(tg, teq) - dt else 'NO TURNING'
    outcome = per[0] if len(set(per.values())) == 1 else 'SPLIT-SENSITIVE'
    if not (ctrl['C3.3']['pass'] and ctrl['C3.4']['pass']):
        outcome = 'UNDERPOWERED (C3.3 or C3.4 failed)'
    t0 = tr(real)
    fs = np.array([4.0, 2.0, 1.0])
    tv = np.array([cv[4], cv[2], cv[1]])
    a_eq = '> 4s' if t0 > tv.max() else '< 1s' if t0 < tv.min() else float(np.exp(np.interp(t0, tv[np.argsort(tv)], np.log(fs[np.argsort(tv)]))))
    return {'outcome': outcome, 'per_split': per, 'tau_rand': {'data': t0, 'gauss': tg, 'sheet_eq': teq, 'mix64': t64}, 'delta_tau': dt, 'a_eq_in_s': a_eq,
            'gate': 'T1 THICK: read as the turning of the top-m spread, not of a tangent plane' if t1_outcome == 'THICK' else '', 'controls': ctrl}


def v_t4():
    t = lambda c, s=0: res(c, s)['t4']    # noqa: E731
    real, sh0, sh1, lum, uni = t('real'), t('sheet:ctrl:0', 0), t('sheet:ctrl:0', 1), (res('lumpy:ctrl') or {}).get('t4'), t('union:ctrl')
    # The lumpy sheet is spacing-matched like every control, and its clumps can keep the matched spacing out of reach (Infeasible in build, so no
    # smooth.json). Without it C4.4 cannot be read: (a) and (b) are then read against the sheet only, and say so.
    miss = '' if lum else 'lumpy control infeasible at matched spacing'
    c44 = lum['s'] - sh0['s'] if lum else None
    ctrl = {'C4.1': {'s': [sh0['s'], sh1['s']], 'pass': max(sh0['s'], sh1['s']) <= 0.05},
            'C4.2': {'s': uni['union']['s'], 's_expected': uni['union']['s_expected'], 'pass': abs(uni['union']['s'] - uni['union']['s_expected']) <= 0.05},
            'C4.3': {'diff': uni['coherence']['part'] - sh0['coherence']['part'], 'pass': uni['coherence']['part'] - sh0['coherence']['part'] >= 0.1},
            'C4.4': {'lumpy_minus_sheet': c44, 'lumpy_only': c44 >= 0.05} if lum else {'lumpy_minus_sheet': None, 'lumpy_only': None, 'missing': miss}}
    ctl = [sh0, sh1] + [lum] * bool(lum)
    base = [lum] if lum and c44 >= 0.05 else ctl
    a = all(real[key] - max(b[key] for b in base) >= 0.05 for key in ('s', 's_part')) and real['near_copy']['0.15']['s'] - max(b['s'] for b in base) >= 0.05
    bb = real['coherence']['part'] - max(b['coherence']['part'] for b in ctl) >= 0.1
    med = lambda r: r['mle']['20']['other']['p10_50_90'][1]    # noqa: E731
    mg, m64 = (med(t('gauss', 0)) + med(t('gauss', 1))) / 2, med(t('mix:64'))
    return {'a': ('REAL VARIATION' if a else 'no variation beyond the sheet and lumpy controls') + f' ({miss}: read against the sheet control only)' * bool(miss),
            'b': ('REGIONAL' if bb else 'not regional') + f' ({miss}: read against the sheet control only)' * bool(miss), 'missing_controls': [miss] * bool(miss),
            'c_report': {'data': med(real), 'gauss': mg, 'mix64': m64, 'below_mix64_by_10pct': med(real) < 0.9 * m64,
                         'reading': 'lower than the spectrum means lumps or a lower-dimensional set; T4 alone cannot separate them'},
            's': {'data': real['s'], 'sheet': [sh0['s'], sh1['s']], 'lumpy': lum['s'] if lum else None}, 's_part': {'data': real['s_part'], 'sheet': [sh0['s_part'], sh1['s_part']], 'lumpy': lum['s_part'] if lum else None},
            'scale_ratio': {'data': real['scale_ratio'], 'sheet': sh0['scale_ratio']}, 'controls': ctrl}


def v_grid():
    cells = [c['grid'] for c in (load(f'{BASE}/grid/{n}/smooth.json') for n in sorted(os.listdir(f'{BASE}/grid'))) if c and c['grid'].get('feasible')]
    real = res('real')
    m_data = real['t4']['mle']['20']['other']['p10_50_90'][1]
    out = {'n_cells': len(cells)}
    for k in (100, 300):
        pts = np.array([[c['lb_median'], c['s_out'][str(k)]] for c in cells])
        ds = LinearNDInterpolator(pts, np.array([[c['d'], c['sigma_R']] for c in cells], f64))(m_data, so(real, k))[0]
        out[k] = {'d': ds[0], 'sigma_R': ds[1]} if np.isfinite(ds).all() else {'reading': 'no flat sheet with isotropic noise reproduces both'}
    return out


def verdict():
    v1 = rule(v_t1)
    out = {'t1': v1, 't3': rule(lambda: v_t3(v1.get('outcome'))), 't4': rule(v_t4), 'grid': rule(v_grid)}
    old = load(f'{BASE}/verdict.json') or {}
    for k, v in out.items():    # never overwrite a reached verdict with a pending one
        if v.get('outcome') == 'pending' and old.get(k) and old[k].get('outcome') != 'pending':
            out[k] = old[k]
    merge(f'{BASE}/verdict.json', out)
    return {k: v.get('outcome', v.get('a')) for k, v in out.items()}


# ---------- main ----------
def main():
    global CLOUD, SEED, LENS, BASE, OUT, X0, ART, YR, NAMES, GENRE, rows, N, D
    out_root, stages, CLOUD = sys.argv[1], sys.argv[2].split(','), sys.argv[3]
    LENS = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else CLIP
    SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    assert set(stages) <= set(ORDER), stages
    run = [s for s in ORDER if s in stages]
    assert CLOUD == 'real' or set(run) <= {'t1', 't3', 't4', 'ax'}, 'nulls run t1, t3, t4, ax only'
    BASE = f'{out_root}/{os.path.splitext(os.path.basename(LENS))[0]}'
    OUT = cdir(CLOUD, SEED)
    artist = labels['artist']
    rows = np.flatnonzero((artist != '') & (year >= 1000) & (year <= 2026))    # the corpus: named works dated 1000-2026
    X0 = unit(lens(LENS)[rows].astype(f64))
    N, D = X0.shape
    ART, YR, NAMES, GENRE = np.unique(artist[rows], return_inverse=True)[1], year[rows], artist[rows], labels['genre'][rows]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print(CLOUD, run, 'N', N, 'D', D, flush=True)
    if any(s in run for s in ('t1', 't2', 't3', 't4', 'ax')):
        info = build(CLOUD)
        merge(f'{OUT}/smooth.json', {'cloud': info})
        print('cloud', {k: v for k, v in info.items() if k != 'pieces'}, f'{time.time() - t0:.0f}s', flush=True)
    failed = []
    for st in run:    # a failing stage is logged ('FAILED:' in the log) and the later ones still run
        try:
            if st == 'ax':    # its own file: never into smooth.json
                merge(f'{OUT}/ax.json', {'ax': ax()})
                print(st, 'written', f'{time.time() - t0:.0f}s', flush=True)
            elif st in ('t1', 't2', 't3', 't4'):
                merge(f'{OUT}/smooth.json', {st: globals()[st]()})
                print(st, 'written', f'{time.time() - t0:.0f}s', flush=True)
            else:
                r = globals()[st]()
                print(st, r if st == 'verdict' else 'written', f'{time.time() - t0:.0f}s', flush=True)
        except Exception:
            traceback.print_exc()
            print('FAILED:', st, CLOUD, flush=True)
            failed.append(st)
            if st in ('calib', 'grid'):    # these swap the current cloud; the next stage must read the data's
                set_cloud(X0)
    print('done', flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
