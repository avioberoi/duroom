"""Does the same distance from the record mean the same thing everywhere? Per named dated work: d10 (distance to the 10th nearest), local (d10 over its neighbours' median d10) and distance to the centre, each over all works and over other artists' works; by decade and label, rank agreement and top-1% overlap.
Usage: distance.py OUT_DIR [LENS file, default CLIP]"""
import json
import os
import sys
from multiprocessing import get_context

for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np    # noqa: E402

from data import CLIP, card, ids, labels, lens, year    # noqa: E402

LENS = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else CLIP
OUT = f'{sys.argv[1]}/{os.path.splitext(os.path.basename(LENS))[0]}'
X = lens(LENS).astype('float64')
X /= np.linalg.norm(X, axis=1, keepdims=True)
os.makedirs(OUT, exist_ok=True)
artist, style, genre = labels['artist'], labels['style'], labels['genre']
usable = (artist != '') & (year >= 1000) & (year <= 2026)
rows = np.flatnonzero(usable)
Xr, art_r = X[rows], artist[rows]    # the record, shared with the workers by fork
decade = np.floor(year / 10) * 10
K = 10


def nearest(d):
    part = np.argpartition(d, K, axis=1)[:, :K]
    dk = np.take_along_axis(d, part, axis=1)
    order = np.argsort(dk, axis=1)
    return np.take_along_axis(dk, order, axis=1)[:, K - 1], rows[np.take_along_axis(part, order, axis=1)]


def chunk(idx):
    """For a block of works: distance to the K nearest works, over all works (itself excluded) and over works by other artists, and who they are."""
    d = np.sqrt(np.maximum(2 - 2 * (X[idx] @ Xr.T), 0))    # against the record (named dated works)
    d[np.arange(len(idx)), np.searchsorted(rows, idx)] = np.inf    # not itself
    all_k = nearest(d)
    for i, q in enumerate(idx):
        d[i, art_r == artist[q]] = np.inf    # not its own artist either
    return all_k, nearest(d)


if __name__ == '__main__':
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', len(os.sched_getaffinity(0))))
    blocks = [rows[i:i + 500] for i in range(0, len(rows), 500)]
    print(len(rows), 'works in', len(blocks), 'blocks on', workers, 'processes', flush=True)
    with get_context('fork').Pool(workers) as p:
        parts = p.map(chunk, blocks, chunksize=1)
    scores, neighbours = {}, {}
    for way, w in (('all', 0), ('others', 1)):    # over all works, and over works by other artists
        d10 = np.full(len(X), np.nan)
        neigh = np.full((len(X), K), -1, dtype=int)
        for b, pair in zip(blocks, parts):
            d10[b], neigh[b] = pair[w]
        local = d10 / np.median(d10[np.where(neigh < 0, 0, neigh)], axis=1)    # how far, for where it sits
        local[~usable] = np.nan
        scores[f'd10_{way}'], scores[f'local_{way}'], neighbours[way] = d10, local, neigh
    centre_vec = X[rows].mean(0)
    centre_vec /= np.linalg.norm(centre_vec)
    scores['centre'] = np.sqrt(np.maximum(2 - 2 * (X @ centre_vec), 0))
    print('NaN local:', {k: int(np.isnan(v[rows]).sum()) for k, v in scores.items() if k.startswith('local')}, flush=True)
    np.savez(f'{OUT}/scores.npz', ids=ids, **scores, **{f'neighbours_{w}': n for w, n in neighbours.items()})    # per-work scores, corpus row order, for later use
    spearman = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])    # noqa: E731  # rank correlation, no scipy needed
    result = {'lens': LENS, 'n': int(len(rows)), 'k': K}
    result['by_decade'] = [(int(dc), *[float(np.median(scores[s][rows][decade[rows] == dc])) for s in scores], int((decade[rows] == dc).sum())) for dc in np.unique(decade[rows]) if (decade[rows] == dc).sum() >= 50]
    PAIRS = (('d10_all', 'local_all'), ('d10_others', 'local_others'), ('d10_all', 'd10_others'), ('local_all', 'local_others'), ('d10_all', 'centre'), ('local_all', 'centre'))
    result['spearman'] = {f'{a}_{b}': spearman(scores[a][rows], scores[b][rows]) for a, b in PAIRS}
    top = {s: set(rows[np.argsort(-scores[s][rows])[:len(rows) // 100]]) for s in scores}
    result['top1pct_overlap'] = {f'{a}_{b}': len(top[a] & top[b]) / len(top[a]) for a, b in PAIRS}
    result['top1pct_by_era'] = {s: {era: int(sum(1 for i in top[s] if lo <= year[i] < hi)) for era, (lo, hi) in (('before_1900', (0, 1900)), ('1900_1959', (1900, 1960)), ('1960_on', (1960, 3000)))} for s in scores}
    result['era_sizes'] = {era: int(((year[rows] >= lo) & (year[rows] < hi)).sum()) for era, (lo, hi) in (('before_1900', (0, 1900)), ('1900_1959', (1900, 1960)), ('1960_on', (1960, 3000)))}
    by_label = lambda s, v: sorted(((n, float(np.median(scores[s][rows][v[rows] == n]))) for n in np.unique(v[rows][v[rows] != '']) if (v[rows] == n).sum() >= 200), key=lambda x: x[1])    # noqa: E731
    result['by_style'] = {s: {'low': by_label(s, style)[:8], 'high': by_label(s, style)[-8:]} for s in scores}
    result['by_genre'] = {s: {'low': by_label(s, genre)[:6], 'high': by_label(s, genre)[-6:]} for s in scores}
    def pick(cands, s, n=12):
        out, seen = [], {}
        for i in sorted(cands, key=lambda i: -scores[s][i]):
            if seen.get(artist[i], 0) < 2:
                seen[artist[i]] = seen.get(artist[i], 0) + 1
                out.append({**card(i, OUT, float(scores[s][i])), **{k: float(v[i]) for k, v in scores.items()}})
            if len(out) == n:
                break
        return out
    result['cards'] = {'top_d10_not_local': pick(top['d10_all'] - top['local_all'], 'd10_all'), 'top_local_not_d10': pick(top['local_all'] - top['d10_all'], 'local_all'), 'top_both': pick(top['d10_all'] & top['local_all'], 'd10_all'),
                       'top_centre_not_local': pick(top['centre'] - top['local_all'], 'centre'), 'top_others_not_all': pick(top['d10_others'] - top['d10_all'], 'd10_others')}
    # the record's local scale by decade: how spread works are where they sit, which is what the absolute score cannot see
    nb = neighbours['all']
    result['neighbour_scale_by_decade'] = [(int(dc), float(np.median(np.median(scores['d10_all'][nb[rows][decade[rows] == dc]], axis=1)))) for dc in np.unique(decade[rows]) if (decade[rows] == dc).sum() >= 50]
    json.dump(result, open(f'{OUT}/distance.json', 'w'), indent=1)
    print('spearman', {k: round(v, 3) for k, v in result['spearman'].items()}, '| top 1% overlap', {k: round(v, 3) for k, v in result['top1pct_overlap'].items()})
    print('top 1% by era:', result['top1pct_by_era'], '| era sizes', result['era_sizes'])
    print('medians by decade (decade: d10_all, local_all, d10_others, local_others, centre):', ' '.join('%d:' % x[0] + '/'.join('%.2f' % v for v in x[1:-1]) for x in result['by_decade'] if x[0] >= 1700 and x[0] % 20 == 0))
    print('done', flush=True)
