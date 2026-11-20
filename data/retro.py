"""Retrodiction: would the search's own score have flagged what the record did after a cutoff? Every named dated work is scored against R = works before CUT (abs, others, local, leftover), with bars at p95 and p99, by era, decade, genre and style.
Usage: retro.py OUT_DIR [CUT=1900] [LENS file, default CLIP] [REF_N]   MEDIA env: comma list of genres counted as other media"""
import json
import os
import sys
from multiprocessing import get_context

for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np    # noqa: E402

from basis import capped, code    # noqa: E402
from data import CLIP, ids, labels, lens, year    # noqa: E402

CUT = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 1900
LENS = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else CLIP
REF_N = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 0    # optional: a random subsample of R of this size (seed 0), so cutoffs with different record sizes compare fairly
OUT = f'{sys.argv[1]}/{os.path.splitext(os.path.basename(LENS))[0]}_cut{CUT}' + (f'_ref{REF_N}' if REF_N else '')
os.makedirs(OUT, exist_ok=True)
K, KDIR, CAP = 15, 32, 20
MEDIA = os.environ.get('MEDIA', 'photo,poster,installation,architecture,design,sculpture').split(',')
X = lens(LENS).astype('float64')
assert np.isfinite(X).all() and (np.linalg.norm(X, axis=1) > 1e-8).all(), 'a vector is not finite or is zero'
X /= np.linalg.norm(X, axis=1, keepdims=True)
artist, style, genre = labels['artist'], labels['style'], labels['genre']
usable = (artist != '') & (year >= 1000) & (year <= 2026)    # NaN fails both
rows = np.flatnonzero(usable)
past, later = rows[year[rows] < CUT], rows[year[rows] >= CUT]
if REF_N:
    past = np.sort(np.random.default_rng(0).choice(past, REF_N, replace=False))    # sorted: chunk() relies on searchsorted
Xp, art_p = X[past], artist[past]    # the reference, shared with the workers by fork
decade = np.floor(np.nan_to_num(year) / 10) * 10
SCORES = ('abs_all', 'abs_others', 'local', 'leftover')
media = np.isin(genre, MEDIA)
assert media[later].any(), f'no later work carries a genre in {MEDIA}'


def nearest(d):
    """Mean distance to the K nearest reference works, and which rows they are (unsorted)."""
    part = np.argpartition(d, K, axis=1)[:, :K]
    return np.take_along_axis(d, part, axis=1).mean(1), past[part]


def chunk(idx):
    d = np.sqrt(np.maximum(2 - 2 * (X[idx] @ Xp.T), 0))
    pos = np.searchsorted(past, idx)
    own = (pos < len(past)) & (past[np.minimum(pos, len(past) - 1)] == idx)    # rows of idx that are reference works
    d[np.flatnonzero(own), pos[own]] = np.inf    # not itself
    over_all = nearest(d)
    for i, q in enumerate(idx):
        d[i, art_p == artist[q]] = np.inf    # not its own artist either
    return over_all, nearest(d)


def auc(a, b):
    """P(a > b) with ties split: the rank form of the Mann-Whitney statistic."""
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


if __name__ == '__main__':
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', len(os.sched_getaffinity(0))))
    blocks = [rows[i:i + 500] for i in range(0, len(rows), 500)]
    print(len(rows), 'works,', len(past), 'before', CUT, 'in', len(blocks), 'blocks on', workers, 'processes', flush=True)
    with get_context('fork').Pool(workers) as p:
        parts = p.map(chunk, blocks, chunksize=1)
    S, NB = {}, {}
    for way, w in (('all', 0), ('others', 1)):
        s, nb = np.full(len(X), np.nan), np.full((len(X), K), -1, dtype=int)
        for b, pair in zip(blocks, parts):
            s[b], nb[b] = pair[w]
        S[f'abs_{way}'], NB[way] = s, nb
    S['local'] = S['abs_all'] / np.median(S['abs_all'][np.where(NB['all'] < 0, 0, NB['all'])], axis=1)    # the neighbours' own leave-one-out level
    rng = np.random.default_rng(0)
    names = rng.permutation(np.unique(artist[past]))
    fit_half = np.isin(artist[past], names[:len(names) // 2])    # the basis is fit on half of R's artists, scored on the other half
    fit_rows = capped(past[fit_half], CAP, rng)
    mean, V = code(X[fit_rows])
    V = V[:KDIR]
    c = X - mean
    proj = c @ V.T
    S['leftover'] = 1 - (proj ** 2).sum(1) / (c ** 2).sum(1)
    u = (c[later] - proj[later] @ V).mean(0)
    u /= np.linalg.norm(u)
    shift = c @ u    # loading on the later works' mean leftover direction
    shift[~usable] = np.nan
    for k in S:
        S[k][~usable] = np.nan
        assert np.isfinite(S[k][rows]).all(), f'{k}: a usable work has a non-finite score'
    REF = {s: past for s in SCORES}
    REF['leftover'] = past[~fit_half]    # the other half of R's artists: never in the fit
    bars = {s: {str(p): float(np.percentile(S[s][REF[s]], p)) for p in (95, 99)} for s in SCORES}

    def table(sel):
        sel = np.flatnonzero(sel)
        out = {'n': int(len(sel))}
        if len(sel) < 50:
            return out
        for s in SCORES:
            rows_s = np.setdiff1d(sel, past[fit_half]) if s == 'leftover' else sel    # pre-cutoff works of the fit half never score the leftover
            if s == 'leftover' and len(rows_s) != len(sel):
                out['n_leftover'] = int(len(rows_s))
            if len(rows_s) < 50:
                continue
            v = S[s][rows_s]
            out[f'{s}_above_q99'], out[f'{s}_above_q95'] = float((v > bars[s]['99']).mean()), float((v > bars[s]['95']).mean())
            ref = np.setdiff1d(REF[s], rows_s)    # the decade's own works leave the reference pool
            out[f'{s}_auc_vs_past'] = auc(v, S[s][ref]) if len(ref) else None
        passed = S['abs_all'][sel] > bars['abs_all']['99']
        out['shift_median'] = float(np.median(shift[sel]))
        out['shift_median_abs_passers'] = float(np.median(shift[sel][passed])) if passed.any() else None
        out['shift_median_abs_nonpassers'] = float(np.median(shift[sel][~passed])) if (~passed).any() else None
        return out

    big = [dc for dc in np.unique(decade[rows]) if (decade[rows] == dc).sum() >= 50]
    result = {'lens': LENS, 'cut': CUT, 'k': K, 'directions': KDIR, 'cap_for_fit': CAP, 'n_past': int(len(past)), 'n_later': int(len(later)), 'ref_subsample': REF_N,
              'same_era_holdout': table(usable & (year < CUT) & ~np.isin(np.arange(len(X)), past)) if REF_N else None,    # pre-cutoff works left out of R: the contemporaries' pass rate
              'n_fit_sample': int(len(fit_rows)), 'n_leftover_reference': int(len(REF['leftover'])), 'media_genres': MEDIA, 'bars': bars,
              'reference_share_above_own_q99': {s: float((S[s][REF[s]] > bars[s]['99']).mean()) for s in SCORES},    # 0.01 by construction
              'eras': {era: table(usable & (year >= lo) & (year < hi)) for era, (lo, hi) in (('before_cut', (0, CUT)), ('cut_to_1959', (CUT, 1960)), ('1960_on', (1960, 3000)))},
              'later_not_media': table(usable & (year >= CUT) & ~media), 'later_media': table(usable & (year >= CUT) & media),
              'by_decade': {int(dc): table(usable & (decade == dc)) for dc in big},
              'by_decade_not_media': {int(dc): table(usable & (decade == dc) & ~media) for dc in big}}

    def by_label(v):
        out = {}
        for name in np.unique(v[later]):
            sel = later[v[later] == name]
            if name == '' or len(sel) < 200:
                continue
            out[name] = {'n': int(len(sel)), **{f'{s}_above_q99': float((S[s][sel] > bars[s]['99']).mean()) for s in SCORES}, 'shift_median': float(np.median(shift[sel]))}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]['abs_all_above_q99']))

    def top(v, sel):    # the ten most frequent labels among sel, with their share of sel and of all later works
        names, counts = np.unique(v[sel], return_counts=True)
        return [{'label': str(n), 'n': int(k), 'share_of_passers': float(k / len(sel)), 'share_of_later': float((v[later] == n).mean())}
                for n, k in sorted(zip(names, counts), key=lambda t: -t[1]) if n != ''][:10]

    result['later_by_genre'], result['later_by_style'] = by_label(genre), by_label(style)
    passers = {s: later[S[s][later] > bars[s]['99']] for s in SCORES}
    result['passers_abs_all'] = {'n': int(len(passers['abs_all'])), 'media_share': float(media[passers['abs_all']].mean()) if len(passers['abs_all']) else None,
                                 'media_share_of_later': float(media[later].mean()), 'top_genres': top(genre, passers['abs_all']), 'top_styles': top(style, passers['abs_all'])}
    result['passer_jaccard'] = {f'{a}_{b}': float(len(np.intersect1d(passers[a], passers[b])) / max(1, len(np.union1d(passers[a], passers[b]))))
                                for a, b in (('abs_all', 'leftover'), ('abs_all', 'local'), ('abs_all', 'abs_others'))}
    np.savez_compressed(f'{OUT}/scores.npz', ids=ids, shift=shift.astype('float32'), **{k: v.astype('float32') for k, v in S.items()})
    with open(f'{OUT}/retro.json', 'w') as f:
        json.dump(result, f, indent=1)
    fmt = lambda t, k: ('%.3f' % t[k]) if t.get(k) is not None else 'na'    # noqa: E731
    print('bars q99', {s: round(bars[s]['99'], 4) for s in SCORES}, '| reference above own q99', {s: round(v, 4) for s, v in result['reference_share_above_own_q99'].items()})
    print('eras (above q99: abs_all / leftover / local; auc abs_all):', {e: '/'.join(fmt(t, k) for k in ('abs_all_above_q99', 'leftover_above_q99', 'local_above_q99', 'abs_all_auc_vs_past')) for e, t in result['eras'].items()})
    print('later not media / media, abs_all above q99:', fmt(result['later_not_media'], 'abs_all_above_q99'), fmt(result['later_media'], 'abs_all_above_q99'))
    print('by decade (abs_all above q99 / leftover above q99 / shift median):', ' '.join('%d:%s/%s/%s' % (dc, fmt(t, 'abs_all_above_q99'), fmt(t, 'leftover_above_q99'), fmt(t, 'shift_median')) for dc, t in result['by_decade'].items() if dc >= CUT - 50))
    print('done', flush=True)
