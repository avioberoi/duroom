"""Library for t1_v2.py: reuses code of unexp.py, parts.py, basis.py and retro.py by ast extraction (importing them would run their scripts).
Usage: imported by t1_v2.py; loads no corpus at import."""
import ast
import os  # noqa: F401  global of the extracted pmap
import time
import types
from multiprocessing import get_context  # noqa: F401  global of the extracted pmap

import numpy as np
import pandas as pd  # noqa: F401  global of the extracted capped

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCES = {'unexp': f'{BASE}/unexp.py', 'parts': f'{BASE}/parts.py', 'basis': f'{BASE}/basis.py', 'retro': f'{BASE}/retro.py'}
K = 30          # global of sortedK / knn_block: neighbours computed to 30 once; k = 5, 10, 15, 30 are prefixes of the sorted list
G = {}          # global of knn_pass / loner_pass: state shared with the forked workers
AC = None       # global of knn_block / loner_block: code per row of the array being scored
artist = None   # global of capped: artist name per row
EXTRACTED = {}  # name -> exact source text that runs


def _source(name):
    path = SOURCES[name]
    raw = open(path, 'rb').read()
    return raw.decode('utf-8')


def _extract():
    want = {'unexp': ('auc', 'pmap', 'blocks_of', 'sortedK', 'knn_block', 'knn_pass', 'loner_block', 'loner_pass'),
            'parts': ('kmeans',), 'basis': ('capped', 'code')}
    for name, fns in want.items():
        src = _source(name)
        found = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in fns}
        assert sorted(found) == sorted(fns), (name, sorted(found))
        for fn in fns:
            exec(compile(found[fn], f'{SOURCES[name]}:{fn}', 'exec'), globals())
            EXTRACTED[f'{name}.{fn}'] = found[fn]
    src = _source('unexp')
    run = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    stm = {s.lineno: ast.get_source_segment(src, s) for s in run.body}
    codes = {}
    for key, lines in (('local', (125,)), ('centre', (126, 127, 128)), ('leftover', (133, 134, 135, 136, 137))):
        assert all(ln in stm for ln in lines), key
        text = '\n'.join(stm[ln] for ln in lines)
        codes[key] = compile(text, f'unexp.py:run:{lines[0]}-{lines[-1]}', 'exec')
        EXTRACTED[f'unexp.run.{key}'] = text
    src = _source('retro')
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MEDIA')
    text = ast.get_source_segment(src, node)
    ns = {'os': types.SimpleNamespace(environ={})}    # the default list, whatever MEDIA holds in the job's environment
    exec(text, ns)
    EXTRACTED['retro.MEDIA'] = text
    return codes, list(ns['MEDIA'])


STMTS, MEDIA_DEFAULT = _extract()


def describe():
    """One line per extracted piece: name, lines; the three statement groups in full."""
    out = []
    for k, v in EXTRACTED.items():
        out.append(f'  {k} ({v.count(chr(10)) + 1} lines)')
        if k.startswith('unexp.run.') or k == 'retro.MEDIA':
            out += ['      ' + ln for ln in v.splitlines()]
    return '\n'.join(out)


def set_labels(names):
    global artist
    artist = np.asarray(names)


def chord(cos):
    return np.sqrt(np.maximum(2 - 2 * np.asarray(cos, dtype=np.float64), 0))


def unit_rows(Y):
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def full(n, rows, vals, fill=np.nan):
    out = np.full(n, fill, dtype=np.float64)
    out[rows] = vals
    return out


def _check_pool(pool, need):
    pool = np.asarray(pool)
    assert pool.ndim == 1 and len(pool) > need and np.all(np.diff(pool) > 0), 'pool: sorted unique rows, more than K'
    return pool


def knn(XA, pool, query, codes):
    """unexp.knn_pass with AC = codes: sorted chords to the K nearest pool rows (itself out), their rows, and the chords with the query's code also out."""
    global AC
    pool = _check_pool(pool, K)
    assert len(codes) == len(XA)
    AC = np.asarray(codes)
    return knn_pass(XA, pool, np.asarray(query))


def loners(XA, mean, V, pool, flag, query, codes):
    """unexp.loner_pass with AC = codes: how many of each query row's 10 nearest pool rows (itself and its code out) carry flag."""
    global AC
    pool = _check_pool(pool, 10)
    AC = np.asarray(codes)
    query = np.asarray(query)
    if len(query) == 0:
        return np.zeros(0, dtype=np.int64)
    return loner_pass(XA, mean, V, pool, np.asarray(flag), query)


def local_ratio(abs_all, nb, query, n):
    """unexp.run's local-ratio statement as written: abs_all[QUERY] / median(abs_all[nb[QUERY]])."""
    ns = {'np': np, 'S': {'local': np.full(n, np.nan)}, 'QUERY': np.asarray(query), 'abs_all': abs_all, 'nb': nb}
    exec(STMTS['local'], ns)
    return ns['S']['local']


def centre_angle(XA, ref):
    """unexp.run's centre statements as written with past = ref: degrees from the unit mean direction of ref, for every row of XA."""
    ns = {'np': np, 'XA': XA, 'past': np.asarray(ref), 'S': {}}
    exec(STMTS['centre'], ns)
    return ns['S']['centre'], ns['m']


def leftover(XA, fit_rows, ndir):
    """unexp.run's leftover statements as written: each row's share outside the fit sample's first ndir directions, with the mean and directions."""
    ns = {'np': np, 'code': code, 'XA': XA, 'fit_rows': np.asarray(fit_rows), 'KDIR': ndir, 'S': {}}
    exec(STMTS['leftover'], ns)
    return ns['S']['leftover'], ns['mean'], ns['V']


def regions(X, fit, rows, Kr, seed, qs, C=None, mean=None):
    """Spherical k-means regions on the fit rows' directions from mean; for each of rows, r_q = min over centres of angle / radius_k(q) (r > 1 = outside every region)."""
    fit, rows = np.asarray(fit), np.asarray(rows)
    m = X[fit].mean(0) if mean is None else mean
    Yf = unit_rows(X[fit] - m)
    if C is None:
        C = kmeans(Yf, Kr, np.random.default_rng(seed))
    C = np.asarray(C, dtype=np.float64)
    cf = np.clip(Yf @ C.T, -1, 1)
    lab_f = np.argmax(cf, axis=1)
    ang_f = np.degrees(np.arccos(cf[np.arange(len(fit)), lab_f]))
    rad = np.full((len(C), len(qs)), np.nan)
    rawc = np.full(C.shape, np.nan)
    rawrad = np.full((len(C), len(qs)), np.nan)
    for k in range(len(C)):
        sel = lab_f == k
        if sel.any():
            rad[k] = np.percentile(ang_f[sel], qs)
            rc = X[fit[sel]].mean(0)
            rawc[k] = rc / np.linalg.norm(rc)
            rawrad[k] = np.percentile(chord(X[fit[sel]] @ rawc[k]), qs)
    Y = unit_rows(X[rows] - m)
    cr = np.clip(Y @ C.T, -1, 1)
    ang = np.degrees(np.arccos(cr))
    lab = np.argmax(cr, axis=1)
    ok = rad[:, 0] > 0    # NaN (no fit rows) and a zero radius are both skipped
    r = {q: np.min(ang[:, ok] / rad[ok, j][None, :], axis=1) for j, q in enumerate(qs)}
    return {'mean': m, 'C': C, 'rad': rad, 'rawc': rawc, 'rawrad': rawrad, 'n_fit': np.bincount(lab_f, minlength=len(C)), 'lab_fit': lab_f,
            'lab': lab, 'ang_near': ang[np.arange(len(rows)), lab], 'r': r, 'empty': int((~ok).sum()), 'qs': tuple(qs)}


def split(artist_names, usable, year, seed, cut=1900):
    """R = usable rows before cut, later = from cut on; A, B = halves of R's artists (rng seed); later rows whose artist is in A are dropped."""
    R = np.flatnonzero(usable & (year < cut))
    L = np.flatnonzero(usable & (year >= cut))
    names = np.random.default_rng(seed).permutation(np.unique(artist_names[R]))
    inA = np.isin(artist_names, names[:len(names) // 2])
    return R[inA[R]], R[~inA[R]], L[~inA[L]], L[inA[L]], R, L


def h1_scores(X, A, Bq, Lq, codes, ks=(5, 10, 15, 30), Ks=(16, 32, 64), qs=(90, 95, 99), levels=(90, 95, 99), ndir=32, fit_cap=20,
              cap_seed=0, kseed=1, local_k=15, log=print):
    """Every candidate score for rows A (reference), Bq (held-out past), Lq (later), all against A; NaN outside them."""
    n = len(X)
    assert local_k <= K
    A, Bq, Lq = (np.asarray(a) for a in (A, Bq, Lq))
    for a in (A, Bq, Lq):
        assert np.all(np.diff(a) > 0)
    assert len(np.intersect1d(A, Bq)) == 0 and len(np.intersect1d(A, Lq)) == 0 and len(np.intersect1d(Bq, Lq)) == 0
    Q = np.concatenate([A, Bq, Lq])    # A included: the local ratio's denominators are A's own leave-one-out levels
    S = {}
    t0 = time.time()
    da, ia, do = knn(X, A, Q, codes)
    for k in ks:
        assert k <= K
        S[f'S1_k{k}_excl'] = full(n, Q, do[:, :k].mean(1))
        S[f'S1_k{k}_all'] = full(n, Q, da[:, :k].mean(1))
    abs_all = full(n, Q, da[:, :local_k].mean(1))
    nb = np.zeros((n, local_k), dtype=np.int64)
    nb[Q] = ia[:, :local_k]
    S['local'] = local_ratio(abs_all, nb, Q, n)
    del da, ia, do, nb
    log(f'  knn to A ({len(A)} rows) for {len(Q)} rows: {time.time() - t0:.0f}s')
    ang, _ = centre_angle(X, A)
    S['centre_deg'] = full(n, Q, ang[Q])
    S['centre'] = full(n, Q, chord(np.cos(np.radians(ang[Q]))))
    fit_rows = capped(A, fit_cap, np.random.default_rng(cap_seed))
    lo, lmean, V = leftover(X, fit_rows, ndir)
    S['leftover'] = full(n, Q, lo[Q])
    sA = np.sort(lo[A])
    S['loner_rank'] = full(n, Q, np.searchsorted(sA, lo[Q], side='right') / len(A))
    uniq = np.arange(n)
    LON = {}
    for line, ref in (('A', A), ('B', Bq)):
        for L in levels:
            t = float(np.percentile(lo[ref], L))
            over = Q[lo[Q] > t]
            for rule, cds in (('excl', codes), ('all', uniq)):
                cnt = np.full(n, -1, dtype=np.int64)
                cnt[over] = loners(X, lmean, V, A, lo[A] > t, over, cds)
                LON[(line, L, rule)] = (t, cnt)
    log(f'  leftover (fit sample {len(fit_rows)}) and loners: {time.time() - t0:.0f}s')
    REG = {}
    for Kr in Ks:
        reg = regions(X, A, Q, Kr, kseed, qs)
        REG[Kr] = reg
        for q in qs:
            S[f'region_K{Kr}_q{q}'] = full(n, Q, reg['r'][q])
        log(f'  regions K={Kr}: empty {reg["empty"]}, fit rows per region min {reg["n_fit"].min()} median {int(np.median(reg["n_fit"]))} max {reg["n_fit"].max()}')
    for key, s in S.items():
        assert np.isfinite(s[Q]).all(), f'{key}: non-finite score on a scored row'
    return S, LON, REG, {'Q': Q, 'fit_rows': fit_rows, 'lmean': lmean, 'V': V}


def h1_candidates(ks, Ks, qs):
    """(candidate, rule, key of S) for every continuous score; the loner's continuous form is its leftover rank among A's leftovers."""
    c = [(f'S1_k{k}', rule, f'S1_k{k}_{rule}') for k in ks for rule in ('excl', 'all')]
    c += [('local15', 'all', 'local'), ('centre', '-', 'centre'), ('leftover32', '-', 'leftover'), ('loner', '-', 'loner_rank')]
    c += [(f'region_K{Kr}_q{q}', '-', f'region_K{Kr}_q{q}') for Kr in Ks for q in qs]
    return c


def h1_tables(S, LON, groups, later_groups, ks, Ks, qs, levels, keep=None):
    """Shares flagged per group at the A-line (percentile of A's scores) and B-line, loners, regions, and AUC of each later group against B; keep masks the groups."""
    A, Bq = groups['A'], groups['B']
    G_ = {g: (r if keep is None else r[keep[r]]) for g, r in groups.items()}
    rows, aucs = [], []

    def add(cand, rule, line, level, cut, flag_of):
        for g, r in G_.items():
            f = flag_of(r)
            rows.append({'candidate': cand, 'rule': rule, 'line': line, 'level': level, 'cut': cut, 'group': g, 'n': len(r),
                         'flagged': int(f.sum()), 'share': float(f.mean()) if len(r) else np.nan})

    for cand, rule, key in h1_candidates(ks, Ks, qs):
        s = S[key]
        if cand == 'loner' or cand.startswith('region_'):
            continue
        for line, ref in (('A', A), ('B', Bq)):
            for L in levels:
                t = float(np.percentile(s[ref], L))
                add(cand, rule, line, L, t, lambda r, s=s, t=t: s[r] > t)
    lo = S['leftover']
    for (line, L, rule), (t, cnt) in LON.items():
        for g, r in G_.items():
            over = lo[r] > t
            iso = over & (cnt[r] == 0)
            rows.append({'candidate': 'loner', 'rule': rule, 'line': line, 'level': L, 'cut': t, 'group': g, 'n': len(r),
                         'flagged': int(iso.sum()), 'share': float(iso.mean()) if len(r) else np.nan, 'n_over_line': int(over.sum()),
                         'isolated_share_of_over': float(iso.sum() / over.sum()) if over.any() else np.nan})
    for Kr in Ks:
        for q in qs:
            s = S[f'region_K{Kr}_q{q}']
            for line, t in (('def_r>1', 1.0), ('A', float(np.percentile(s[A], q))), ('B', float(np.percentile(s[Bq], q)))):
                add(f'region_K{Kr}_q{q}', '-', line, q, t, lambda r, s=s, t=t: s[r] > t)
    for cand, rule, key in h1_candidates(ks, Ks, qs):
        s = S[key]
        for g in later_groups:
            r = G_[g]
            aucs.append({'candidate': cand, 'rule': rule, 'group': g, 'n': len(r), 'n_B': len(G_['B']),
                         'auc_vs_B': auc(s[r], s[G_['B']]) if len(r) and len(G_['B']) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(aucs)


def _dup_block(idx):
    s = G['DX'][idx] @ G['DP'].T
    i, j = np.nonzero(s > G['thr'])
    return idx[i], G['dpool'][j], s[i, j]


def dup_pairs(X, query, pool, thr):
    """Every (query row, pool row) pair with cosine > thr, exact (blockwise over all pool rows); a row is never paired with itself."""
    query, pool = np.asarray(query), np.asarray(pool)
    if len(query) == 0 or len(pool) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    G.update(DX=X, DP=X[pool], dpool=pool, thr=thr)
    parts = pmap(_dup_block, blocks_of(query))
    a, b, c = (np.concatenate(x) for x in zip(*parts))
    keep = a != b
    return a[keep], b[keep], c[keep]


def c1_scores(X, P, Q, codes, ks):
    """S1 against P's stored vectors for rows Q, both rules (a P row leaves itself out; 'excl' also its artist)."""
    n = len(X)
    da, _, do = knn(X, P, Q, codes)
    out = {}
    for k in ks:
        out[f'k{k}_excl'] = full(n, Q, do[:, :k].mean(1))
        out[f'k{k}_all'] = full(n, Q, da[:, :k].mean(1))
    return out


def c2_scores(X, R, orig_rows, REP, codes, ks):
    """S1 against R's stored vectors of each original and its repaint (REP[i]); 'excl' leaves out the original's artist, 'all' only its own row."""
    n, m = len(X), len(orig_rows)
    orig_rows = np.asarray(orig_rows)
    XA2 = np.vstack([X, REP])
    Q = np.concatenate([orig_rows, n + np.arange(m)])
    out = {}
    for rule, cds in (('excl', np.concatenate([codes, codes[orig_rows]])), ('all', np.concatenate([np.arange(n), orig_rows]))):
        _, _, do = knn(XA2, R, Q, cds)
        for k in ks:
            out[(rule, k)] = (do[:m, :k].mean(1), do[m:, :k].mean(1))
    del XA2
    return out


def own_rank(REP, X, orig_rows, O):
    """Rank of each repaint's own original among the originals O (sorted rows) by chord from the repaint (1 = nearest)."""
    pos = np.searchsorted(O, orig_rows)
    assert np.all(O[pos] == orig_rows)
    ranks = np.empty(len(REP), dtype=np.int64)
    XO = X[O]
    for s in range(0, len(REP), 1000):
        d = chord(REP[s:s + 1000] @ XO.T)
        own = d[np.arange(len(d)), pos[s:s + 1000]]
        ranks[s:s + 1000] = 1 + (d < own[:, None]).sum(1)
    return ranks


def h4_reference(R, cap, size, rng, codes):
    """Sorted reference of `size` R rows with at most `cap` per artist (cap 0 = none); a prefix of basis.capped is a uniform draw of the capped set."""
    ref = capped(R, cap, rng)[:size] if cap else rng.choice(R, size, replace=False)
    assert len(ref) == size, f'cap {cap}: only {len(ref)} rows'
    ref = np.sort(ref)
    if cap:
        assert np.bincount(codes[ref]).max() <= cap
    return ref


def attainable(R, cap, codes):
    if not cap:
        return len(R)
    return int(np.minimum(np.bincount(codes[R]), cap).sum())
