"""Is an artwork one position, or a sum of parts? Per work, the share of squared length explained by its best k-means centre against the best non-negative combination of up to three, with a random-directions null, stability across part counts and seeds, and labels at both ends.
Usage: parts.py OUT_DIR [LENS file, default CLIP] [SEEDS, comma-separated, e.g. 7,8] [NULL: gauss]"""
import json
import os
import sys
from multiprocessing import get_context

for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np    # noqa: E402

from basis import code    # noqa: E402
from data import CLIP, card, ids, labels, lens, year    # noqa: E402

LENS = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else CLIP
NULL = sys.argv[4] if len(sys.argv) > 4 else ''
assert NULL in ('', 'gauss'), NULL
OUT =f'{sys.argv[1]}/{os.path.splitext(os.path.basename(LENS))[0]}' + ('_gauss' if NULL == 'gauss' else '')
X = lens(LENS).astype('float64')
X /= np.linalg.norm(X, axis=1, keepdims=True)
os.makedirs(OUT, exist_ok=True)
artist, style, genre = labels['artist'], labels['style'], labels['genre']
usable = (artist != '') & (year >= 1000) & (year <= 2026)
rows = np.flatnonzero(usable)
decade = np.floor(year / 10) * 10
rng = np.random.default_rng(6)
SEEDS = [int(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 and sys.argv[3] else []
K, MAXP = 32, 3


def kmeans(Y, k, r, iters=30):
    """Spherical k-means on unit vectors; centres as unit rows; r draws the start."""
    C = Y[r.choice(len(Y), k, replace=False)]
    for _ in range(iters):
        lab = np.argmax(Y @ C.T, axis=1)
        C = np.stack([Y[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C


def nnls(A, y):
    """Exact non-negative least squares for a few columns: the best all-non-negative least-squares fit over subsets of columns."""
    best, bw = np.inf, np.zeros(A.shape[1])
    for m in range(1, 2 ** A.shape[1]):
        cols = [j for j in range(A.shape[1]) if m >> j & 1]
        w = np.linalg.lstsq(A[:, cols], y, rcond=None)[0]
        if (w >= 0).all():
            r = y - A[:, cols] @ w
            if r @ r < best:
                best, bw = r @ r, np.zeros(A.shape[1]); bw[cols] = w
    return bw


def explain(Y, C):
    """Per row of Y: share explained by its best single part and by up to MAXP parts (greedy, refit by NNLS), and how many carry a fifth of the largest weight."""
    out = np.zeros((len(Y), 3))
    for i, y in enumerate(Y):
        n2 = y @ y
        proj = C @ y
        one = max(proj.max(), 0) ** 2 / n2    # best single part, non-negative direction
        chosen, r = [], y.copy()
        for _ in range(MAXP):
            cand = C @ r
            cand[chosen] = -np.inf
            j = int(np.argmax(cand))
            if cand[j] <= 0:
                break
            chosen.append(j)
            w = nnls(C[chosen].T, y)
            r = y - C[chosen].T @ w
        many = 1 - (r @ r) / n2
        out[i] = one, many, int((w >= 0.2 * w.max()).sum()) if chosen else 0
    return out


def run(Y, C, name, P=None, r=None):
    """Explained shares under the parts and under random directions (projected onto the subspace P spans if given), with summaries; per-work arrays too."""
    r = rng if r is None else r
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', len(os.sched_getaffinity(0))))
    blocks = [Y[i:i + 2000] for i in range(0, len(Y), 2000)]
    with get_context('fork').Pool(workers) as p:
        real = np.vstack(p.starmap(explain, [(b, C) for b in blocks]))
        Rand = r.standard_normal(C.shape)
        if P is not None:
            Rand -= (Rand @ P.T) @ P    # the null lives where the leftover lives
        Rand /= np.linalg.norm(Rand, axis=1, keepdims=True)
        null = np.vstack(p.starmap(explain, [(b, Rand) for b in blocks[:5]]))    # 10,000 works are enough for the chance level
    one, many, n_parts = real.T
    gain = many - one
    summary = {'n_parts_fit': int(len(C)), 'one_part_share_pct_10_50_90': np.percentile(one, [10, 50, 90]).tolist(), 'three_parts_share_pct_10_50_90': np.percentile(many, [10, 50, 90]).tolist(),
               'gain_pct_10_50_90': np.percentile(gain, [10, 50, 90]).tolist(), 'gain_relative_median': float(np.median(gain / np.maximum(one, 1e-9))),
               'share_needing_2_or_more_parts': float((n_parts >= 2).mean()), 'share_needing_3': float((n_parts >= 3).mean()),
               'null_one_median': float(np.median(null[:, 0])), 'null_three_median': float(np.median(null[:, 1])), 'null_gain_median': float(np.median(null[:, 1] - null[:, 0])),
               'by_decade': [(int(dc), float(np.median(one[decade[rows] == dc])), float(np.median(gain[decade[rows] == dc])), float((n_parts[decade[rows] == dc] >= 2).mean())) for dc in np.unique(decade[rows]) if (decade[rows] == dc).sum() >= 50]}
    lab = lambda v, key: sorted(((n, float(np.median(key[v[rows] == n]))) for n in np.unique(v[rows][v[rows] != '']) if (v[rows] == n).sum() >= 200), key=lambda x: x[1])    # noqa: E731
    summary['gain_by_style'] = {'low': lab(style, gain)[:6], 'high': lab(style, gain)[-8:]}
    summary['gain_by_genre'] = {'low': lab(genre, gain)[:5], 'high': lab(genre, gain)[-6:]}
    def pick(order, n=12):
        out, seen = [], {}
        for i in order:
            a = artist[rows[i]]
            if seen.get(a, 0) < 2:
                seen[a] = seen.get(a, 0) + 1
                out.append({**card(rows[i], OUT, float(gain[i])), 'one': float(one[i]), 'three': float(many[i]), 'n_parts': int(n_parts[i])})
            if len(out) == n:
                break
        return out
    summary['cards'] = {'one_position': pick(np.argsort(gain)), 'sum_of_parts': pick(np.argsort(-gain))}
    assign = np.argmax(Y @ C.T, axis=1)
    summary['parts'] = [{'part': j, 'works': int((assign == j).sum()), 'styles': [(n, int(c)) for n, c in sorted(zip(*np.unique(style[rows][(assign == j) & (style[rows] != '')], return_counts=True)), key=lambda x: -x[1])[:3]],
                         'median_year': (float(np.median(year[rows][assign == j])) if (assign == j).any() else None)} for j in range(len(C))]
    print(f"{name}: one part median {np.median(one):.3f}, three parts {np.median(many):.3f}, gain {np.median(gain):.3f} (null one {np.median(null[:, 0]):.4f}, three {np.median(null[:, 1]):.4f}); works needing 2+ parts {100 * (n_parts >= 2).mean():.0f}%, 3 {100 * (n_parts >= 3).mean():.0f}%", flush=True)
    return summary, real


if __name__ == '__main__':
    if NULL == 'gauss':    # a continuous cloud of the record's shape: Gaussian with the record's mean and covariance spectrum, on the sphere
        Z = X[rows] - X[rows].mean(0)
        w, U = np.linalg.eigh(Z.T @ Z / len(Z))
        X[rows] = X[rows].mean(0) + (rng.standard_normal(Z.shape) * np.sqrt(np.maximum(w, 0))) @ U.T
        X[rows] /= np.linalg.norm(X[rows], axis=1, keepdims=True)
        del Z, w, U
    mean, V = code(X[rows])
    c = X[rows] - mean
    Cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    fit = rng.choice(len(rows), min(40000, len(rows)), replace=False)
    res = {'lens': LENS, 'n': int(len(rows)), 'max_parts': MAXP, 'whole': {}, 'leftover': {}}
    R = c - (c @ V[:K].T) @ V[:K]
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    per = {}
    for name, Y, ks, P in (('whole', Cn, (8, 32, 64), None), ('leftover', Rn, (8, 16, 32), V[:K])):
        for k in ks:
            res[name][str(k)], per[(name, k)] = run(Y, kmeans(Y[fit], k, rng), f'{name}, {k} parts', P)
        gains = {k: per[(name, k)][:, 1] - per[(name, k)][:, 0] for k in ks}
        needs = {k: per[(name, k)][:, 2] >= 2 for k in ks}
        rank = lambda a: np.argsort(np.argsort(a))    # noqa: E731
        top = lambda g: set(np.argsort(-g)[:len(rows) // 10])    # noqa: E731
        res[name]['across_k'] = {'gain_spearman': {f'{a}_{b}': float(np.corrcoef(rank(gains[a]), rank(gains[b]))[0, 1]) for a in ks for b in ks if a < b},
                                 'top10pct_gain_shared': {f'{a}_{b}': float(len(top(gains[a]) & top(gains[b])) / (len(rows) // 10)) for a in ks for b in ks if a < b},
                                 'needs_2_at_every_k': float(np.all([needs[k] for k in ks], axis=0).mean()), 'needs_2_at_no_k': float((~np.any([needs[k] for k in ks], axis=0)).mean())}
        print(name, 'across k:', {kk: {k2: round(v2, 2) for k2, v2 in vv.items()} if isinstance(vv, dict) else round(vv, 3) for kk, vv in res[name]['across_k'].items()}, flush=True)
        # the same against other seeds of the centres (new fit sample, new start, new null directions) at one number of parts
        k0 = ks[1]
        res[name]['across_seeds'] = {'k': k0, 'main_seed': 6}
        for s in SEEDS:
            r2 = np.random.default_rng(s)
            fit2 = r2.choice(len(rows), min(40000, len(rows)), replace=False)
            s2, per2 = run(Y, kmeans(Y[fit2], k0, r2), f'{name}, {k0} parts, seed {s}', P, r2)
            g2, n2 = per2[:, 1] - per2[:, 0], per2[:, 2] >= 2
            res[name]['across_seeds'][str(s)] = {'gain_spearman': float(np.corrcoef(rank(gains[k0]), rank(g2))[0, 1]), 'top10pct_gain_shared': float(len(top(gains[k0]) & top(g2)) / (len(rows) // 10)),
                                                 'needs_2_both': float((needs[k0] & n2).mean()), 'needs_2_neither': float((~needs[k0] & ~n2).mean()),
                                                 'one_part_median': float(np.median(per2[:, 0])), 'gain_median': float(np.median(g2)), 'null_gain_median': s2['null_gain_median']}
            print(name, f'seed {s} vs 6 at {k0} parts:', {kk: round(vv, 3) for kk, vv in res[name]['across_seeds'][str(s)].items()}, flush=True)
    np.savez(f'{OUT}/parts.npz', ids=np.full(len(rows), -1) if NULL else ids[rows], **{f'{n}_{k}': a for (n, k), a in per.items()})    # per work: one-part share, three-part share, parts needed; no ids for synthetic rows
    json.dump(res, open(f'{OUT}/parts.json', 'w'), indent=1)
    print('done', flush=True)
