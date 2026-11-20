"""What is left over for each artwork, and to whom is it rare? Leftover share and vector under the record's first K directions; kind or isolate among the top 1% by leftover neighbours; shares under three references (record, past, random half).
Usage: leftover.py OUT_DIR [LENS file, default CLIP] [K=32] [CAP=0] [SCORES npz from distance.py, optional]"""
import json
import os
import sys
from multiprocessing import get_context

for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np    # noqa: E402

from basis import capped, code    # noqa: E402
from data import CLIP, card, ids, labels, lens, year    # noqa: E402

LENS = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else CLIP
K = int(sys.argv[3]) if len(sys.argv) > 3 else 32
CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 0
SCORES = sys.argv[5] if len(sys.argv) > 5 else ''
OUT = f'{sys.argv[1]}/{os.path.splitext(os.path.basename(LENS))[0]}_k{K}_cap{CAP}'
X = lens(LENS).astype('float64')
X /= np.linalg.norm(X, axis=1, keepdims=True)
os.makedirs(OUT, exist_ok=True)
artist, style, genre = labels['artist'], labels['style'], labels['genre']
usable = (artist != '') & (year >= 1000) & (year <= 2026)
rows = np.flatnonzero(usable)
decade = np.floor(year / 10) * 10
rng = np.random.default_rng(2)
KN = 10


def fit(sample):
    """Mean and first K directions of a sample (capped if CAP), as the reference."""
    mean, V = code(X[capped(sample, CAP, rng) if CAP else sample])
    return mean, V[:K]


def leftover(mean, V, idx):
    """Leftover vectors and shares for works idx under a reference."""
    c = X[idx] - mean
    r = c - (c @ V.T) @ V
    return r, (r ** 2).sum(1) / (c ** 2).sum(1)


mean, V = fit(rows)
R, share = leftover(mean, V, rows)    # the whole record as the reference
Rn = R / np.linalg.norm(R, axis=1, keepdims=True)


def nearest(s):
    part = np.argpartition(-s, KN, axis=1)[:, :KN]
    sk = np.take_along_axis(s, part, axis=1)
    order = np.argsort(-sk, axis=1)
    return np.take_along_axis(sk, order, axis=1), np.take_along_axis(part, order, axis=1)


art_r = artist[rows]


def chunk(block):
    """Nearest works in leftover space, by cosine of the leftover vectors: over all works (itself excluded), and over works by other artists."""
    s = Rn[block] @ Rn.T
    s[np.arange(len(block)), block] = -np.inf
    all_k = nearest(s)
    for i, q in enumerate(block):
        s[i, art_r == art_r[q]] = -np.inf
    return all_k, nearest(s)


def concentration(top_idx, v, draws=500, n=50):
    """Max label share among the top n (at most two per artist) vs random draws of the labelled works, one-sided permutation p."""
    lab_all = v[rows]
    def pick(order):
        out, seen = [], {}
        for i in order:
            if lab_all[i] and seen.get(artist[rows[i]], 0) < 2:
                seen[artist[rows[i]]] = seen.get(artist[rows[i]], 0) + 1
                out.append(i)
            if len(out) == n:
                break
        return np.array(out)
    sh = lambda idx: float(np.unique(lab_all[idx], return_counts=True)[1].max() / len(idx))    # noqa: E731
    chosen = pick(top_idx)
    obs = sh(chosen)
    null = np.array([sh(pick(rng.permutation(len(rows)))) for _ in range(draws)])
    names, counts = np.unique(lab_all[chosen], return_counts=True)
    return {'label': str(names[counts.argmax()]), 'top_share': obs, 'null_mean': float(null.mean()), 'p': float((1 + (null >= obs).sum()) / (draws + 1))}


if __name__ == '__main__':
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', len(os.sched_getaffinity(0))))
    blocks = [np.arange(i, min(i + 500, len(rows))) for i in range(0, len(rows), 500)]
    with get_context('fork').Pool(workers) as p:
        parts = p.map(chunk, blocks, chunksize=1)
    sim = np.vstack([a[0] for a, _ in parts])    # cosine to the 10 nearest leftovers, positions into rows
    nb = np.vstack([a[1] for a, _ in parts])
    nb_others = np.vstack([b[1] for _, b in parts])    # the same over works by other artists
    past = np.flatnonzero(usable & (year < 1900))
    _, share_past = leftover(*fit(past), rows)
    half = rng.permutation(len(rows)) < len(rows) // 2
    share_half = np.full(len(rows), np.nan)
    for h in (half, ~half):    # code on one half, scored on the other
        _, share_half[~h] = leftover(*fit(rows[h]), rows[~h])
    refs = {'record': share, 'past': share_past, 'half': share_half}
    rank = lambda a: np.argsort(np.argsort(a))    # noqa: E731
    spearman = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])    # noqa: E731
    n1 = len(rows) // 100
    top = {k: set(np.argsort(-v)[:n1]) for k, v in refs.items()}
    result = {'lens': LENS, 'k': K, 'cap': CAP, 'n': int(len(rows)), 'kept_by_record_code_median': float(1 - np.median(share)),
              'share_percentiles_1_10_50_90_99': np.percentile(share, [1, 10, 50, 90, 99]).tolist(),
              'spearman': {f'{a}_{b}': spearman(refs[a], refs[b]) for a, b in (('record', 'past'), ('record', 'half'), ('past', 'half'))},
              'top1pct_overlap': {f'{a}_{b}': len(top[a] & top[b]) / n1 for a, b in (('record', 'past'), ('record', 'half'), ('past', 'half'))}}
    # null for the kin counts: the top flag shuffled over works (same number of flags), 200 draws
    top_mask = np.zeros(len(rows), bool); top_mask[list(top['record'])] = True
    top10 = np.zeros(len(rows), bool); top10[np.argsort(-share)[:len(rows) // 10]] = True
    kin = top_mask[nb[top_mask]].sum(1)
    def kin_stats(flag, neighbours):
        k = flag[neighbours[top_mask]].sum(1)
        return {'mean_of_10': float(k.mean()), 'share_with_none': float((k == 0).mean()), 'share_with_5_or_more': float((k >= 5).mean())}
    nulls = []
    for _ in range(200):
        f = np.zeros(len(rows), bool); f[rng.choice(len(rows), n1, replace=False)] = True
        nulls.append(f[nb[top_mask]].sum(1).mean())
    result['top1pct_leftover_neighbours_also_top'] = {'all_works': kin_stats(top_mask, nb), 'other_artists': kin_stats(top_mask, nb_others),
                                                       'vs_top10pct_all_works': kin_stats(top10, nb), 'vs_top10pct_other_artists': kin_stats(top10, nb_others),
                                                       'null_mean_of_10': {'mean': float(np.mean(nulls)), 'sd': float(np.std(nulls, ddof=1))},
                                                       'leftover_cosine_to_10th_median_top': float(np.median(sim[top_mask][:, -1])), 'leftover_cosine_to_10th_median_all': float(np.median(sim[:, -1]))}
    order = np.argsort(-share)
    result['concentration'] = {lab: concentration(order, v) for lab, v in (('style', style), ('genre', genre))}
    result['by_decade'] = [(int(dc), float(np.median(share[decade[rows] == dc])), float(np.median(share_past[decade[rows] == dc])), int((decade[rows] == dc).sum())) for dc in np.unique(decade[rows]) if (decade[rows] == dc).sum() >= 50]
    by_label = lambda v: sorted(((n, float(np.median(share[v[rows] == n])), int((v[rows] == n).sum())) for n in np.unique(v[rows][v[rows] != '']) if (v[rows] == n).sum() >= 200), key=lambda x: x[1])    # noqa: E731
    result['by_style'] = {'low': by_label(style)[:8], 'high': by_label(style)[-10:]}
    result['by_genre'] = {'low': by_label(genre)[:6], 'high': by_label(genre)[-8:]}
    result['top1pct_by_era'] = {era: int(sum(1 for i in top['record'] if lo <= year[rows[i]] < hi)) for era, (lo, hi) in (('before_1900', (0, 1900)), ('1900_1959', (1900, 1960)), ('1960_on', (1960, 3000)))}
    if SCORES:
        z = np.load(SCORES)
        assert (z['ids'] == ids).all()
        for s in ('d10_all', 'local_all', 'centre'):
            t = set(np.argsort(-z[s][rows])[:n1])
            result[f'top1pct_overlap_with_{s}'] = len(top['record'] & t) / n1
    def pick(cands, key, n=12):
        out, seen = [], {}
        for i in cands:
            a = artist[rows[i]]
            if seen.get(a, 0) < 2:
                seen[a] = seen.get(a, 0) + 1
                out.append({**card(rows[i], OUT, float(share[i])), 'share': float(share[i]), 'share_past': float(share_past[i]), 'kin_of_10': int(top_mask[nb[i]].sum()), 'leftover_cosine_to_10th': float(sim[i, -1])})
            if len(out) == n:
                break
        return out
    top_sorted = [i for i in order if top_mask[i]]
    result['cards'] = {'largest_leftover': pick(order, 'share'), 'largest_leftover_kinds': pick([i for i in top_sorted if top_mask[nb[i]].sum() >= 5], 'share'),
                       'largest_leftover_isolates': pick([i for i in top_sorted if top_mask[nb[i]].sum() == 0], 'share'), 'smallest_leftover': pick(order[::-1], 'share'),
                       'largest_under_past_not_record': pick(sorted(top['past'] - top['record'], key=lambda i: -share_past[i]), 'share_past')}
    print('NaN shares:', int(np.isnan(share).sum()), int(np.isnan(share_past).sum()), int(np.isnan(share_half).sum()), '| NaN cosines:', int(np.isnan(sim).sum()), flush=True)
    np.savez(f'{OUT}/leftover.npz', ids=ids[rows], share=share, share_past=share_past, share_half=share_half, neighbour_ids=ids[rows[nb]], neighbour_ids_other_artists=ids[rows[nb_others]], neighbour_cosine=sim)
    json.dump(result, open(f'{OUT}/leftover.json', 'w'), indent=1)
    print('median kept by the record code at k=%d: %.3f; leftover share percentiles 1/10/50/90/99: %s' % (K, result['kept_by_record_code_median'], np.round(result['share_percentiles_1_10_50_90_99'], 3)))
    print('rank agreement', {k: round(v, 3) for k, v in result['spearman'].items()}, '| top 1% overlap', {k: round(v, 3) for k, v in result['top1pct_overlap'].items()})
    t = result['top1pct_leftover_neighbours_also_top']
    print(f"top 1% leftovers: neighbours also top 1% (mean of 10) {t['all_works']['mean_of_10']:.2f} (other artists {t['other_artists']['mean_of_10']:.2f}, null {t['null_mean_of_10']['mean']:.2f}), none {100 * t['all_works']['share_with_none']:.0f}%; "
          f"also top 10%: {t['vs_top10pct_all_works']['mean_of_10']:.2f}; top-50 concentration {result['concentration']}")
    print('top 1% by era', result['top1pct_by_era'], '| genres high', [(n, round(v, 3)) for n, v, c in result['by_genre']['high'][-4:]])
    print('done', flush=True)
