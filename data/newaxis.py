"""The direction later works take that the past's directions do not describe: per cutoff, the shift and new axes of the leftover on the past's first K directions (fit on half the later artists, scored on the other half, over SEEDS splits), against a no-time null, with stability, loadings by decade and label.
Usage: newaxis.py OUT_DIR [CUTOFFS=1900,1950,2000] [K=32] [SEEDS=5] [key=value ...]   (keys: lens=NAME later_window=Y past_window=Y n_later=N; each adds to the folder name)"""
import json
import os
import sys

import numpy as np

from basis import capped, code
from data import CLIP, card, labels, lens, year

CUTS = [int(c) for c in (sys.argv[2] if len(sys.argv) > 2 else '1900,1950,2000').split(',')]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 32
OPT = dict(a.split('=', 1) for a in sys.argv[5:])
assert set(OPT) <= {'lens', 'later_window', 'past_window', 'n_later'}, f'unknown option {OPT}'
LATER_W, PAST_W, N_LATER = (int(OPT[k]) if k in OPT else None for k in ('later_window', 'past_window', 'n_later'))
TAG = ''.join(f'_{k}{v}' for k, v in OPT.items())
OUT = f'{sys.argv[1]}/k{K}{TAG}'    # one folder per K and option set, so a sweep does not overwrite itself
os.makedirs(OUT, exist_ok=True)
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
assert SEEDS >= 2, 'stability and sd need two seeds or more'
N_AXES, CAP = 8, 20
X = lens(f'{os.environ["DUROOM_EMB"]}/{OPT["lens"]}.npz' if 'lens' in OPT else CLIP).astype('float64')
X /= np.linalg.norm(X, axis=1, keepdims=True)
artist, style, genre = labels['artist'], labels['style'], labels['genre']
usable = (artist != '') & (year >= 1000) & (year <= 2026)    # NaN fails both
decade = np.where(usable, np.floor(np.nan_to_num(year) / 10) * 10, np.nan)


def leftover(rows, mean, V):
    c = X[rows] - mean
    return c - (c @ V.T) @ V


def halves(rows, rng):
    """Two artist-disjoint halves of rows, each capped."""
    names = rng.permutation(np.unique(artist[rows]))
    a = np.isin(artist[rows], names[:len(names) // 2])
    return capped(rows[a], CAP, rng), capped(rows[~a], CAP, rng)


def axes(past, later, rng, n=(None, None)):
    """Past basis, shift and new axes fit on half the later artists, and the leftover shares they hold, fit and held-out; n truncates the halves (null)."""
    mean, Vfull = code(X[capped(past, CAP, rng)])
    V = Vfull[:K]
    fit, test = (h[:m] for h, m in zip(halves(later, rng), n))
    Rf, Rt = leftover(fit, mean, V), leftover(test, mean, V)
    u = Rf.mean(0) / np.linalg.norm(Rf.mean(0))
    W = code(Rf)[1][:N_AXES]
    def share(R, rows):    # all shares over the same total: the squared leftover
        return {'leftover_of_total': float((R ** 2).sum() / ((X[rows] - mean) ** 2).sum()), 'shift': float(((R @ u) ** 2).sum() / (R ** 2).sum()),
                'axes': ((((R - R.mean(0)) @ W.T) ** 2).sum(0) / (R ** 2).sum()).tolist()}
    return mean, Vfull, u, W, share(Rf, fit), share(Rt, test), fit, test


def stability(vectors):
    """Mean and min |cosine| over pairs of seeds; for 2-row matrices, the cosine of the largest principal angle between the planes."""
    c = [abs(float(a @ b)) if a.ndim == 1 else float(np.linalg.svd(a @ b.T)[1].min()) for i, a in enumerate(vectors) for b in vectors[i + 1:]]
    return {'mean': float(np.mean(c)), 'min': float(np.min(c))}


def past_rank(v, Vfull):
    """Where a direction sits in the past's own ranking: cumulative squared loadings on the past's directions (about 400 is chance)."""
    sq = (Vfull @ v) ** 2
    cum = np.cumsum(sq)
    return {'top10': [(int(r) + 1, float(sq[r])) for r in np.argsort(-sq)[:10]], 'rank_half': int(np.searchsorted(cum, 0.5)) + 1, 'rank_90pct': int(np.searchsorted(cum, 0.9)) + 1}


def ends(load, rows, n=12):
    """Cards at both ends of a loading, at most two per artist."""
    def pick(order):
        out, seen = [], {}
        for i in rows[order]:
            if seen.get(artist[i], 0) < 2:
                seen[artist[i]] = seen.get(artist[i], 0) + 1
                out.append(card(i, OUT, float(load[i])))
            if len(out) == n:
                return out
        return out
    return {'high': pick(np.argsort(-load[rows])), 'low': pick(np.argsort(load[rows]))}


def by_label(load, v, rows, least=200):
    names, counts = np.unique(v[rows][v[rows] != ''], return_counts=True)
    means = {n: float(load[rows][v[rows] == n].mean()) for n, c in zip(names, counts) if c >= least}
    return sorted(means.items(), key=lambda x: x[1])


def concentration(load, rows, v, rng, top=50, draws=500):
    """Max label share among the top-loading rows vs random draws, at most two works per artist (one artist's works load together)."""
    keep = v[rows] != ''
    lab, art, ld = v[rows][keep], artist[rows][keep], load[rows][keep]
    def pick(order):
        out, seen = [], {}
        for i in order:
            if seen.get(art[i], 0) < 2:
                seen[art[i]] = seen.get(art[i], 0) + 1
                out.append(i)
            if len(out) == top:
                break
        return np.array(out)
    share = lambda idx: float(np.unique(lab[idx], return_counts=True)[1].max() / len(idx))    # noqa: E731
    chosen = pick(np.argsort(-ld))
    observed = share(chosen)
    null = np.array([share(pick(rng.permutation(len(lab)))) for _ in range(draws)])
    names, counts = np.unique(lab[chosen], return_counts=True)
    return {'label': str(names[counts.argmax()]), 'top_max_share': observed, 'null_mean': float(null.mean()), 'p': float((1 + (null >= observed).sum()) / (draws + 1))}


result = {'k': K, 'cap': CAP, 'seeds': SEEDS, 'cutoffs': {}}
first = {}
for cut in CUTS:
    past = np.flatnonzero(usable & (year < cut) & ((year >= cut - PAST_W) if PAST_W else True))
    later = np.flatnonzero(usable & (year >= cut) & ((year < cut + LATER_W) if LATER_W else True))
    runs, dirs, null_dirs = [], {'shift': [], 'axis1': [], 'axis2': [], 'plane': []}, {'shift': [], 'axis1': [], 'axis2': [], 'plane': []}
    for seed in range(SEEDS):
        rng = np.random.default_rng([cut, seed])
        drawn = capped(later, CAP, rng)[:N_LATER] if N_LATER else later
        assert len(drawn) == (N_LATER or len(later)), (cut, len(drawn))
        mean, Vfull, u, W, fit_share, test_share, fit, test = axes(past, drawn, rng)
        perm = rng.permutation(np.union1d(past, drawn))    # the null splits the same works with no time; by default the union is every usable row
        _, _, nu, nW, null_fit, null_test, nfit, ntest = axes(perm[:len(past)], perm[len(past):], rng, (len(fit), len(test)))    # same sizes before and after the cap, no time
        runs.append({'seed': seed, 'n_fit': len(fit), 'n_test': len(test), 'null_n_fit': len(nfit), 'null_n_test': len(ntest), 'fit': fit_share, 'held_out': test_share,
                     'null_fit': null_fit, 'null_held_out': null_test, 'axis1_vs_shift_abs_cosine': float(abs(W[0] @ u)), 'axis2_vs_shift_abs_cosine': float(abs(W[1] @ u)),
                     'shift_vs_null_shift_abs_cosine': float(abs(u @ nu))})    # high means the real shift is partly what a random split also shows, e.g. a cap artefact
        for store, (a, B) in ((dirs, (u, W)), (null_dirs, (nu, nW))):
            for key, v in (('shift', a), ('axis1', B[0]), ('axis2', B[1]), ('plane', B[:2])):
                store[key].append(v)
        if seed == 0:
            ref = (mean, Vfull, u, W)
    mean, Vfull, u, W = ref    # seed 0 draws the pictures and labels; the seeds above say how far they would move
    chance = np.random.default_rng([cut, 7]).standard_normal(X.shape[1])
    chance -= Vfull[:K].T @ (Vfull[:K] @ chance)    # a random direction in the same complement, for reading past_rank
    chance /= np.linalg.norm(chance)
    rows = np.flatnonzero(usable)
    loads = {}
    for key, v in (('shift', u), ('axis1', W[0]), ('axis2', W[1])):
        load = (X - mean) @ v
        loads[key] = load * np.sign(load[later].mean())    # later works load positive; cosmetic for a spread axis
    first[cut] = {k: dirs[k][0] for k in dirs}
    held = lambda key: [r['held_out'][key] for r in runs]    # noqa: E731
    cutoff = {'n_past': int(len(past)), 'n_later': int(len(later)), 'runs': runs,
              'summary': {'shift': {'mean': float(np.mean(held('shift'))), 'sd': float(np.std(held('shift'), ddof=1))},
                          'axes': {'mean': np.mean(held('axes'), axis=0).tolist(), 'sd': np.std(held('axes'), axis=0, ddof=1).tolist()},
                          'null_shift': {'mean': float(np.mean([r['null_held_out']['shift'] for r in runs])), 'sd': float(np.std([r['null_held_out']['shift'] for r in runs], ddof=1))},
                          'null_axes': {'mean': np.mean([r['null_held_out']['axes'] for r in runs], axis=0).tolist(), 'sd': np.std([r['null_held_out']['axes'] for r in runs], axis=0, ddof=1).tolist()}},
              'stability': {k: stability(v) for k, v in dirs.items()}, 'null_stability': {k: stability(v) for k, v in null_dirs.items()},
              'past_rank': {k: past_rank(v, Vfull) for k, v in (('shift', u), ('axis1', W[0]), ('axis2', W[1]), ('random_direction', chance))},
              'by_decade': [(int(dc), float(loads['axis1'][rows][decade[rows] == dc].mean()), float(np.sqrt((loads['axis1'][rows][decade[rows] == dc] ** 2).mean())), int((decade[rows] == dc).sum()))
                            for dc in np.unique(decade[rows]) if (decade[rows] == dc).sum() >= 50],    # decade, mean loading on axis 1, rms, count
              'by_decade_plane': [(int(dc), *[float(loads[k][rows][decade[rows] == dc].mean()) for k in ('shift', 'axis1', 'axis2')], int((decade[rows] == dc).sum()))
                                  for dc in np.unique(decade[rows]) if (decade[rows] == dc).sum() >= 50],    # decade, mean on shift, axis 1, axis 2, count: a line or an arc
              'by_style': by_label(loads['axis1'], style, rows), 'by_genre': by_label(loads['axis1'], genre, rows),
              'by_style_axis2': by_label(loads['axis2'], style, rows), 'by_genre_axis2': by_label(loads['axis2'], genre, rows),
              'axis1_later': ends(loads['axis1'], later), 'axis1_past': ends(loads['axis1'], past), 'axis2_later': ends(loads['axis2'], later), 'axis2_past': ends(loads['axis2'], past),
              'shift_later': ends(loads['shift'], later), 'shift_past': ends(loads['shift'], past),
              'precursors': {k: {lab: concentration(loads[k], past, v, np.random.default_rng([cut, 99])) for lab, v in (('style', style), ('genre', genre))} for k in ('shift', 'axis1')}}
    P = np.array([x[1:4] for x in cutoff['by_decade_plane'] if x[0] >= cut])    # the post-cutoff decade means in the shift, axis 1, axis 2 coordinates
    P = P @ np.linalg.inv(np.linalg.qr(np.array([u, W[0], W[1]]).T)[1])    # the same means in an orthonormal basis of that plane: the shift is not orthogonal to the axes
    ev = np.clip(np.linalg.eigvalsh(np.cov((P - P.mean(0)).T))[::-1], 0, None) if len(P) >= 4 else None    # three points always span a plane: uninformative below four
    cutoff['path_variance_shares'] = (ev / ev.sum()).tolist() if ev is not None else None    # all on the first: a straight line; a second share: a bend
    cutoff['n_post_cutoff_decades'] = int(len(P))
    cutoff['directions'] = {'shift': u.tolist(), 'axis1': W[0].tolist(), 'axis2': W[1].tolist()}    # seed 0, so runs at other K can be compared by cosine
    bins = np.linspace(-0.4, 0.8, 61)
    cutoff['by_decade_distribution'] = {k: [(int(dc), np.percentile(loads[k][rows][decade[rows] == dc], [5, 10, 25, 50, 75, 90, 95]).tolist(),
                                             np.histogram(loads[k][rows][decade[rows] == dc], bins)[0].tolist()) for dc in np.unique(decade[rows]) if (decade[rows] == dc).sum() >= 50]
                                        for k in ('shift', 'axis1')}    # decade, percentiles 5 to 95, counts in the bins below
    cutoff['distribution_bins'] = bins.tolist()
    cutoff['distribution_outside_bins'] = {k: [int((loads[k][rows] < bins[0]).sum()), int((loads[k][rows] > bins[-1]).sum())] for k in ('shift', 'axis1')}    # below, above: the bins suit CLIP
    cutoff['options'] = OPT
    cutoff['n_later_used'] = int(N_LATER or len(later))
    np.savez_compressed(f'{OUT}/loads_{cut}.npz', rows=rows, year=year[rows], **{k: v[rows].astype('float32') for k, v in loads.items()})    # every work, seed 0's directions
    result['cutoffs'][cut] = cutoff
    result['abs_cosines_between_cutoffs'] = {k: {f'{a}_{b}': float(abs(first[a][k] @ first[b][k])) for a in first for b in first if a < b} for k in ('shift', 'axis1', 'axis2')}
    json.dump(result, open(f'{OUT}/newaxis.json', 'w'), indent=1)    # after each cutoff, so a failure keeps the earlier ones
    s = cutoff['summary']
    st, ns = cutoff['stability'], cutoff['null_stability']
    print(cut, 'held out over %d seeds: shift %.3f±%.3f (null %.3f), axis 1 %.3f±%.3f, axis 2 %.3f±%.3f (null %.3f); |cos| across seeds, real vs null: shift %.2f/%.2f, axis 1 %.2f/%.2f, axis 2 %.2f/%.2f, plane %.2f/%.2f; past rank of axis 1 %d, random %d' % (
        SEEDS, s['shift']['mean'], s['shift']['sd'], s['null_shift']['mean'], s['axes']['mean'][0], s['axes']['sd'][0], s['axes']['mean'][1], s['axes']['sd'][1], s['null_axes']['mean'][0],
        st['shift']['mean'], ns['shift']['mean'], st['axis1']['mean'], ns['axis1']['mean'], st['axis2']['mean'], ns['axis2']['mean'], st['plane']['mean'], ns['plane']['mean'],
        cutoff['past_rank']['axis1']['rank_half'], cutoff['past_rank']['random_direction']['rank_half']), flush=True)
print('done', flush=True)
