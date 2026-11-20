"""Does the choice of distance (chord, two Mahalanobis, two diffusion maps) change who a work's neighbours are? Per encoder: style, genre, artist and cross-encoder criteria against chord with bootstrap intervals, a reading rule, and neighbour, hub and far-1% measures.
Usage: dist_types.py OUT_DIR   (ROWS_CAP=n keeps n corpus rows after a fixed shuffle, for a quick check; float32 only chooses candidates, reported distances are float64)"""
import json
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

from basis import capped
from data import labels, lens, year

LENS_DIR = os.environ['DUROOM_EMB']
NAMES = ['clip-vit-l-14', 'dinov3-l', 'siglip-so400m']
ALTS = ['MAHA_FULL', 'MAHA_80', 'DIFF_1', 'DIFF_8']
CRITS = ['STYLE', 'GENRE', 'ARTIST', 'XENC']
K, N_QB, MIN_GAIN, BOOT = 10, 20000, 0.01, 2000


def nearest(C, ids, q, k, modes, acode, block=1000):
    """The k nearest rows of C to each query position q, Euclidean, ascending, as corpus positions and distances; modes 'any' (self left out) and 'other' (own artist left out); -1 / nan for a query with no row in C."""
    C32, n2 = np.ascontiguousarray(C, dtype=np.float32), (C ** 2).sum(1).astype(np.float32)
    pos = np.full(len(acode), -1)
    pos[ids] = np.arange(len(ids))
    ac = acode[ids]
    out = {m: (np.full((len(q), k), -1), np.full((len(q), k), np.nan)) for m in modes}
    for b in range(0, len(q), block):
        sel = np.arange(b, min(b + block, len(q)))
        sel = sel[pos[q[sel]] >= 0]
        if len(sel) == 0:
            continue
        p = pos[q[sel]]
        G = C32[p] @ C32.T
        G *= 2
        G -= n2    # higher = nearer (the query's own squared length is the same for every row)
        G[np.arange(len(p)), p] = -np.inf
        for m in ('any', 'other'):    # 'any' first: the 'other' mask is applied in place
            if m not in modes:
                continue
            if m == 'other':
                G[ac[None, :] == acode[q[sel]][:, None]] = -np.inf
            c = min(k + 16, G.shape[1])
            cand = np.argpartition(G, G.shape[1] - c, axis=1)[:, -c:]
            d = np.sqrt(((C[cand] - C[p][:, None, :]) ** 2).sum(2))
            d[np.take_along_axis(G, cand, 1) == -np.inf] = np.inf
            o = np.argsort(d, axis=1)[:, :k]
            d, cand = np.take_along_axis(d, o, 1), np.take_along_axis(cand, o, 1)
            cand = np.where(np.isfinite(d), ids[cand], -1)
            out[m][0][sel], out[m][1][sel] = cand, np.where(np.isfinite(d), d, np.nan)
    return out


def check_cosine_chord(X, q, k=K, block=500):
    """Top-k rows by cosine and by chord (from the squared lengths, not assumed unit) agree for every query q (self left out) whose k-th and (k+1)-th are not tied."""
    n2, checked, tied = (X ** 2).sum(1), 0, 0
    for b in range(0, len(q), block):
        qb = q[b:b + block]
        r = np.arange(len(qb))
        sim = X[qb] @ X.T
        sim[r, qb] = -np.inf
        chord = np.sqrt(np.maximum(n2[None, :] + n2[qb][:, None] - 2 * sim, 0))
        chord[r, qb] = np.inf
        def top(M):    # the k + 1 smallest of each row, in order
            i = np.argpartition(M, k, axis=1)[:, :k + 1]
            return np.take_along_axis(i, np.argsort(np.take_along_axis(M, i, 1), axis=1), 1)
        by_cos, by_chord = top(-sim), top(chord)    # the k + 1 best, in order
        s = np.take_along_axis(sim, by_cos, 1)
        tie = s[:, k - 1] - s[:, k] < 1e-12
        assert (np.sort(by_cos[~tie, :k], axis=1) == np.sort(by_chord[~tie, :k], axis=1)).all(), 'cosine and chord top-k differ'
        checked, tied = checked + int((~tie).sum()), tied + int(tie.sum())
    return {'queries_checked': checked, 'tied_skipped': tied}


def maha(X, sample):
    """Mean of the corpus, and the eigen-decomposition (descending) of the second moment about it of the sampled rows."""
    mean = X.mean(0)
    Z = X[sample] - mean
    lam, U = np.linalg.eigh(Z.T @ Z / len(Z))
    return mean, np.maximum(lam[::-1], 0), U[:, ::-1]


def maha_coords(X, mean, lam, U, kind):
    """Coordinates whose Euclidean distance is the Mahalanobis distance: FULL (eps = 0.01 mean(lambda)) or the leading directions holding 80% of the variance; returns them and the number of directions."""
    if kind == 'FULL':
        return (X - mean) @ (U / np.sqrt(lam + 0.01 * lam.mean())), len(lam)
    m = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.8) + 1)    # fewest leading directions with at least 80%
    return (X - mean) @ (U[:, :m] / np.sqrt(lam[:m])), m


def diffusion(X, acode, n_nb=15, kth=7, n_eig=50):
    """Diffusion map of X (n_nb-nearest kernel, alpha = 1): n_eig + 1 leading eigenpairs, the first trivial, in the largest connected component."""
    N = len(X)
    I, D = nearest(X, np.arange(N), np.arange(N), n_nb, ('any',), acode)['any']
    s = np.maximum(D[:, kth - 1], 1e-9)    # a work with 7+ exact duplicates has s = 0
    A = csr_matrix((np.exp(-D.ravel() ** 2 / (s[np.repeat(np.arange(N), n_nb)] * s[I.ravel()])), (np.repeat(np.arange(N), n_nb), I.ravel())), shape=(N, N))
    A = A.maximum(A.T)
    n_comp, comp = connected_components(A, directed=False)
    big = np.flatnonzero(comp == np.bincount(comp).argmax())
    A = A[big][:, big]
    d = np.asarray(A.sum(1)).ravel()
    A = diags(1 / d) @ A @ diags(1 / d)
    d2 = np.asarray(A.sum(1)).ravel()
    S = diags(d2 ** -0.5) @ A @ diags(d2 ** -0.5)    # same eigenvalues as the Markov matrix D^-1 A
    lam, v = eigsh(S, k=n_eig + 1, which='LA', ncv=min(len(big) - 1, 150), tol=1e-7, v0=np.random.default_rng(0).random(len(big)))
    o = np.argsort(-lam)
    lam, v = lam[o], v[:, o]
    assert abs(lam[0] - 1) < 1e-6, lam[0]
    psi = v / np.sqrt(d2)[:, None] * np.sqrt(d2.sum())    # right eigenvectors of the Markov matrix, unit weighted norm
    return {'big': big, 'lam': lam, 'psi': psi[:, 1:], 'n_components': int(n_comp), 'largest': int(len(big))}


def diffusion_coords(dif, t):
    return dif['psi'] * dif['lam'][1:] ** t


def search_all(C, ids, qa, qb, acode):
    """Everything one distance needs: QA other-artist and any-artist (self left out) neighbours, QB other-artist neighbours with distances."""
    a = nearest(C, ids, qa, K, ('other', 'any'), acode)
    return {'qa_oth': a['other'], 'qa_any': a['any'], 'qb_oth': nearest(C, ids, qb, K, ('other',), acode)['other']}


def share_equal(idx, lab, q):
    """Per query: share of its neighbours with the query's label; nan for an empty label or no neighbours."""
    ok = idx >= 0
    s = ((lab[idx] == lab[q][:, None]) & ok).sum(1) / np.maximum(ok.sum(1), 1)
    return np.where((lab[q] == '') | (ok.sum(1) == 0), np.nan, s)


def overlap(a, b):
    """Per query: how many rows two neighbour lists share; nan where either has none."""
    s = ((a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)).sum((1, 2)).astype(float)
    return np.where((a[:, 0] < 0) | (b[:, 0] < 0), np.nan, s)


def boot_diff(d, rng):
    """Mean of paired differences (nan = query skipped) and its 95% interval from BOOT resamples of the queries."""
    ok = ~np.isnan(d)
    d0, n = np.where(ok, d, 0.0), len(d)
    means = np.concatenate([(lambda ix: d0[ix].sum(1) / ok[ix].sum(1))(rng.integers(0, n, (200, n))) for _ in range(BOOT // 200)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d[ok].mean()), float(lo), float(hi)


def cell(diff, lo, hi, scale):
    """'alt' if the alternative beats chord (difference above 0.01 on a 0-1 scale and the interval excludes 0), 'chord' if chord beats it, else 'none'."""
    if diff > MIN_GAIN * scale and lo > 0:
        return 'alt'
    return 'chord' if diff < -MIN_GAIN * scale and hi < 0 else 'none'


def verdict(cells, crits=CRITS):
    """The reading rule on cells {(alternative, criterion, encoder): 'alt' | 'chord' | 'none'}: another distance is better (beats chord on a criterion in all encoders), cosine is at least as good (no alternative wins more than 2 cells), or mixed."""
    alts, encs = sorted({c[0] for c in cells}), sorted({c[2] for c in cells})
    wins = {a: [(c, e) for c in crits for e in encs if cells[(a, c, e)] == 'alt'] for a in alts}
    losses = {a: [(c, e) for c in crits for e in encs if cells[(a, c, e)] == 'chord'] for a in alts}
    better = [(a, c) for a in alts for c in crits if all(cells[(a, c, e)] == 'alt' for e in encs)]
    out = 'another distance is better' if better else 'cosine is at least as good' if max(len(w) for w in wins.values()) <= 2 else 'mixed'
    return {'verdict': out, 'better_on': better, 'alt_beats_chord': wins, 'chord_beats_alt': losses}


def descriptive(res, ref, qb, yr, n):
    """QB, other-artist 10 nearest: share shared with CHORD, hubness, farthest 1% by 10th-nearest distance against CHORD's, and shares of works from 1960."""
    (I, D), (Ic, Dc) = res['qb_oth'], ref['qb_oth']
    cnt = np.bincount(I[I >= 0], minlength=n).astype(float)
    far = lambda dist: set(np.argsort(np.nan_to_num(dist[:, K - 1], nan=-np.inf))[-max(1, int(round(0.01 * len(qb)))):])    # noqa: E731
    f, fc = far(D), far(Dc)
    late = lambda s: float(np.mean(yr[qb[sorted(s)]] >= 1960))    # noqa: E731
    return {'share_shared_with_chord': float(np.nanmean(overlap(I, Ic)) / K), 'hub_skewness': float((((cnt - cnt.mean()) / cnt.std()) ** 3).mean()),
            'hub_share_never_chosen': float((cnt == 0).mean()), 'hub_max': int(cnt.max()), 'farthest_1pct_size': len(f), 'farthest_1pct_shared_with_chord': len(f & fc) / len(f),
            'farthest_1pct_share_1960_plus': late(f), 'chord_farthest_1pct_share_1960_plus': late(fc), 'qb_share_1960_plus': float(np.mean(yr[qb] >= 1960))}


def main(out):
    cap = int(os.environ['ROWS_CAP']) if os.environ.get('ROWS_CAP') else None
    artist, style, genre = labels['artist'], labels['style'], labels['genre']
    rows = np.flatnonzero((artist != '') & (year >= 1000) & (year <= 2026))
    print('corpus rows', len(rows), 'ROWS_CAP', cap, flush=True)
    if cap:
        rows = np.random.default_rng(99).permutation(rows)[:cap]
    N = len(rows)
    art, sty, gen, yr = artist[rows], style[rows], genre[rows], year[rows]
    acode = np.unique(art, return_inverse=True)[1]
    qa = np.random.default_rng(12).permutation(N)
    qa = qa[np.unique(acode[qa], return_index=True)[1]]    # one random work per artist
    qb = np.random.default_rng(13).choice(N, min(N_QB, N), replace=False)
    where = np.full(len(artist), -1)
    where[rows] = np.arange(N)
    sample = where[capped(rows, 20, np.random.default_rng(11))]    # at most 20 works per artist, as corpus positions
    def chance_of(l):    # the share of corpus rows with the query's label, averaged over the queries that have one
        names, inv, cnt = np.unique(l, return_inverse=True, return_counts=True)
        return float(np.mean((cnt / N)[inv[qa]][l[qa] != '']))
    chance = {'STYLE': chance_of(sty), 'GENRE': chance_of(gen)}
    X = {}
    for name in NAMES:
        X[name] = lens(f'{LENS_DIR}/{name}.npz')[rows].astype('float64')
        X[name] /= np.linalg.norm(X[name], axis=1, keepdims=True)
    res = {'rows_cap': cap, 'lens_dir': LENS_DIR, 'n_corpus': N, 'n_qa': len(qa), 'n_qb': len(qb), 'chance': chance,
           'raw_euclid_note': 'Raw (un-normalised) Euclidean is not rerun here; norm_effect.py measures it for CLIP.', 'encoders': {}}
    allr = np.arange(N)
    chord = {}
    for name in NAMES:
        chord[name] = search_all(X[name], allr, qa, qb, acode)
        res['encoders'][name] = {'cosine_vs_chord': check_cosine_chord(X[name], np.union1d(qa, qb))}
        print(name, 'chord done', flush=True)
    cells = {}
    for name in NAMES:
        r = res['encoders'][name]
        others = [n for n in NAMES if n != name]
        dist_res = {'CHORD': chord[name]}
        mean, lam, U = maha(X[name], sample)
        r['maha'] = {'eigenvalues_top5': lam[:5].tolist(), 'eps': float(0.01 * lam.mean()), 'sample_rows': int(len(sample))}
        for kind, key in (('FULL', 'MAHA_FULL'), ('80', 'MAHA_80')):
            C, m = maha_coords(X[name], mean, lam, U, kind)
            r['maha']['directions_' + kind] = m
            dist_res[key] = search_all(C, allr, qa, qb, acode)
            print(name, key, 'done', flush=True)
        dif = diffusion(X[name], acode)
        r['diffusion'] = {'n_components': dif['n_components'], 'largest_component': dif['largest'], 'eigenvalues_trivial_then_50': dif['lam'].tolist()}
        for t in (1, 8):
            dist_res[f'DIFF_{t}'] = search_all(diffusion_coords(dif, t), dif['big'], qa, qb, acode)
            print(name, f'DIFF_{t}', 'done', flush=True)
        per = {}
        for dname, s in dist_res.items():
            xenc = np.mean([overlap(s['qa_oth'][0], chord[o]['qa_oth'][0]) for o in others], axis=0)
            per[dname] = {'STYLE': share_equal(s['qa_oth'][0], sty, qa), 'GENRE': share_equal(s['qa_oth'][0], gen, qa), 'ARTIST': share_equal(s['qa_any'][0], art, qa), 'XENC': xenc}
        boot = np.random.default_rng(14)
        r['distances'] = {}
        for dname, v in per.items():
            r['distances'][dname] = {'criteria_mean': {c: float(np.nanmean(v[c])) for c in CRITS}, 'queries_used': {c: int((~np.isnan(v[c])).sum()) for c in CRITS},
                                     'descriptive': descriptive(dist_res[dname], chord[name], qb, yr, N)}
            if dname == 'CHORD':
                continue
            r['distances'][dname]['minus_chord'] = {}
            for c in CRITS:
                d = per[dname][c] - per['CHORD'][c]
                diff, lo, hi = boot_diff(d, boot)
                cells[(dname, c, name)] = cell(diff, lo, hi, 10 if c == 'XENC' else 1)    # XENC is out of 10: 0.01 on the 0-1 scale is 0.1
                r['distances'][dname]['minus_chord'][c] = {'diff': diff, 'ci95': [lo, hi], 'cell': cells[(dname, c, name)]}
        del dist_res, X[name]
    res['cells'] = {f'{a}|{c}|{e}': v for (a, c, e), v in cells.items()}
    res['reading'] = verdict(cells)
    res['reading_without_XENC'] = verdict(cells, [c for c in CRITS if c != 'XENC'])
    res['xenc_decides'] = res['reading']['verdict'] != res['reading_without_XENC']['verdict']    # XENC favours chord by construction
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f'{out}/dist_types.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    for name in NAMES:
        print(f'\n{name}: criterion means (alternative minus CHORD, 95% interval) [cell]', flush=True)
        for dname, v in res['encoders'][name]['distances'].items():
            print(f'  {dname:10s}', '  '.join(f"{c} {v['criteria_mean'][c]:.3f}" + ('' if dname == 'CHORD' else f" ({v['minus_chord'][c]['diff']:+.3f} [{v['minus_chord'][c]['ci95'][0]:+.3f},{v['minus_chord'][c]['ci95'][1]:+.3f}] {v['minus_chord'][c]['cell']})") for c in CRITS))
    print('\nchance', chance, '\nverdict:', res['reading']['verdict'], res['reading']['better_on'], '| without XENC:', res['reading_without_XENC']['verdict'], '| XENC decides:', res['xenc_decides'], flush=True)
    return res


if __name__ == '__main__':
    main(sys.argv[1])
