"""How well does a code learned on one group compress another? Held-out works T under codes A (learned without T's group), B and B2 (two disjoint random halves of the other works); gap = A - B, null = B2 - B.
Usage: common.py OUT_DIR [SEEDS=5] [GROUPS, comma-separated, default all; a prefix like cluster selects all of that kind] [LENS file, default CLIP]"""
import json
import os
import sys
import zlib
from multiprocessing import get_context

for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'    # one BLAS thread per process; the processes are the parallelism
import numpy as np    # noqa: E402

from basis import KS, capped, code, each_kept
from data import CLIP, card, ids, labels, lens, year

SEEDS, ALL = int(sys.argv[2]) if len(sys.argv) > 2 else 5, 10 ** 9
LENS = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else CLIP    # an empty argument means the default, so GROUPS can be skipped with "" or all
OUT = f'{sys.argv[1]}/{os.path.splitext(os.path.basename(LENS))[0]}'
X = lens(LENS).astype('float64')    # a wrong file fails here, before any folder is made
os.makedirs(OUT, exist_ok=True)
X /= np.linalg.norm(X, axis=1, keepdims=True)    # stored norms run 0.999 to 1.001 for all three observer files
artist, style, genre = labels['artist'], labels['style'], labels['genre']
named = artist != ''


def kept(Y, mean, V):
    cum, total = each_kept(Y, mean, V)
    return (cum.sum(0) / total.sum()).tolist()


def artists_held(rng):
    """T from half of each held artist's works, so B can see their other works and A cannot see the artist."""
    names = rng.permutation(np.unique(artist[named]))
    held = names[:len(names) // 2]
    half = [rng.permutation(np.flatnonzero(artist == a)) for a in held]
    return np.concatenate([h[:len(h) // 2] for h in half]), np.flatnonzero(named & ~np.isin(artist, held))


def by_mask(t, a):
    return lambda rng: (np.flatnonzero(t & named), np.flatnonzero(a & named))


dated = ~np.isnan(year)
later, earlier = dated & (year >= 1900), dated & (year < 1900)
straddlers = np.intersect1d(artist[later & named], artist[earlier & named])
groups = {'artist': artists_held, 'after_1900': by_mask(later, earlier), 'before_1900': by_mask(earlier, later),
          'after_1900_no_shared_artists': by_mask(later, earlier & ~np.isin(artist, straddlers))}
sweep = list(groups)
for prefix, v in (('style', style), ('genre', genre)):
    names, counts = np.unique(v[v != ''], return_counts=True)
    for n in names[counts >= 500]:
        groups[f'{prefix}:{n}'] = by_mask(v == n, (v != n) & (v != ''))


def kmeans(rows, k, rng, iters=25):
    """Spherical k-means on unit vectors: centroids from a capped sample, then every row assigned by cosine. Returns labels for all of X."""
    fit = X[capped(rows, 20, rng)]
    C = fit[rng.choice(len(fit), k, replace=False)]
    for _ in range(iters):
        lab = np.argmax(fit @ C.T, axis=1)
        C = np.stack([fit[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    return np.argmax(X @ C.T, axis=1), C


cluster, centroids = kmeans(np.flatnonzero(named), 16, np.random.default_rng(16))
cluster_cards = {}
for j in range(16):
    members = np.flatnonzero(named & (cluster == j))
    if len(members) >= 500:
        groups[f'cluster:{j}'] = by_mask(cluster == j, cluster != j)
        top = lambda v: [(n, int(c)) for n, c in sorted(zip(*np.unique(v[members][v[members] != ''], return_counts=True)), key=lambda x: -x[1])[:5]]    # noqa: E731
        cluster_cards[j] = {'size': int(len(members)), 'styles': top(style), 'genres': top(genre), 'year_quartiles': np.nanpercentile(year[members], [25, 50, 75]).tolist(),
                            'nearest_centroid': [card(i, OUT) for i in members[np.argsort(-(X[members] @ centroids[j]))[:8]]]}

cells = [(g, c, ALL) for g in sweep for c in (5, 10, 20, 50)] + [(g, 20, f) for g in sweep for f in (1000, 5000, 10000)] + [(g, 20, ALL) for g in groups if g not in sweep]
if len(sys.argv) > 3 and sys.argv[3] not in ('', 'all'):    # "" or all means all groups
    want = sys.argv[3].split(',')
    assert set(want) <= set(groups) | {g.split(':')[0] for g in groups}, 'unknown group'    # a typo would otherwise run nothing and still print done
    cells = [c for c in cells if c[0] in want or c[0].split(':')[0] in want]    # a small test: one group, one seed; or one kind of group


def cell(job):
    """One row: group, cap, fit size, seed. Seeded by its own name, so the result does not depend on which process runs it."""
    name, cap, fit, seed = job
    rng = np.random.default_rng([seed, zlib.crc32(f'{name}{cap}{fit}'.encode())])
    t_pool, a_pool = groups[name](rng)
    T = capped(t_pool, cap, rng)
    A = capped(np.setdiff1d(a_pool, T), cap, rng)
    pool = rng.permutation(np.setdiff1d(np.flatnonzero(named), T))
    B, B2 = capped(pool[:len(pool) // 2], cap, rng), capped(pool[len(pool) // 2:], cap, rng)
    n = min(len(A), len(B), len(B2), fit)
    A, B, B2 = A[:n], B[:n], B2[:n]
    null = np.column_stack([rng.permutation(X[np.r_[A, T], d]) for d in range(X.shape[1])])
    (mA, VA), (mB, VB) = code(X[A]), code(X[B])
    mT = X[T].mean(0)
    offset = lambda m: float(len(T) * ((mT - m) ** 2).sum() / ((X[T] - m) ** 2).sum())    # share of T's squared distance from a code's mean that is the gap between the two means  # noqa: E731
    return {'group': name, 'cap': cap, 'fit': int(n), 'seed': seed, 'n_test': len(T), 'artists_A_B_B2': [len(np.unique(artist[Y])) for Y in (A, B, B2)],    # A can hold fewer artists than B at the same n
            'unseen': kept(X[T], mA, VA), 'random': kept(X[T], mB, VB),
            'random2': kept(X[T], *code(X[B2])), 'null': kept(null[n:], *code(null[:n])),
            # Move or reordering: the same directions, measured around T's own mean, so a shift of the mean no longer counts.

            'unseen_own_mean': kept(X[T], mT, VA), 'random_own_mean': kept(X[T], mT, VB), 'offset_share_A': offset(mA), 'offset_share_B': offset(mB)}


if __name__ == '__main__':
    jobs = [(name, cap, fit, seed) for name, cap, fit in cells for seed in range(SEEDS)]
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', len(os.sched_getaffinity(0))))
    print(len(jobs), 'cells on', workers, 'processes', flush=True)
    with get_context('fork').Pool(workers) as p:    # fork, stated: X and the groups are shared, not copied or re-imported
        rows = list(p.imap(cell, jobs, chunksize=1))    # same order as jobs
    for i in range(0, len(rows), SEEDS):
        r = rows[i:i + SEEDS]
        print(r[0]['group'], r[0]['cap'], cells[i // SEEDS][2], 'k=32 unseen %.3f random %.3f random2 %.3f null %.3f' % tuple(np.mean([x[key][5] for x in r]) for key in ('unseen', 'random', 'random2', 'null')), flush=True)
    json.dump({'lens': LENS, 'd': int(X.shape[1]), 'ks': KS, 'rows': rows, 'clusters': cluster_cards}, open(f'{OUT}/heldout.json', 'w'), indent=1)    # rows survive if the picture section fails

    rng = np.random.default_rng(1900)
    past = np.flatnonzero(earlier & named)
    T, A = capped(np.flatnonzero(later & named), 20, rng), capped(past, 20, rng)
    mean, Vt = code(X[A])
    cum, total = each_kept(X[T], mean, Vt)
    share = (cum / total)[:, KS.index(32)]
    order = np.argsort(share)
    worst, typical = order[:8], order[len(order) // 2 - 4:len(order) // 2 + 4]

    def strip(j):
        q = T[j]
        frames = []
        for k in (4, 16, 32, 64, 128, 256, 512):
            V = Vt[:k]
            rebuilt = mean + (X[q] - mean) @ V.T @ V
            frames.append({'k': k, **card(past[np.argmax(X[past] @ rebuilt)], OUT)})
        return {'query': card(q, OUT), 'kept_at_32': float(share[j]), 'frames': frames}

    result = {'lens': LENS, 'd': int(X.shape[1]), 'ks': KS, 'rows': rows, 'clusters': cluster_cards, 'worst': [strip(j) for j in worst], 'typical': [strip(j) for j in typical],
              'share_at_32_capped_later_works': {'ids': [int(i) for i in ids[T[order[:500]]]], 'worst_500': share[order[:500]].tolist(), 'percentiles_10_50_90': np.percentile(share, [10, 50, 90]).tolist()}}
    json.dump(result, open(f'{OUT}/heldout.json', 'w'), indent=1, default=str)
    print('done', flush=True)
