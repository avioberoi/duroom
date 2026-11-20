"""Band test: in which directions, defined from works before 1900 only, does a later work's offset from its nearest earlier works lie? Shares in local and global bands, later works against held-out earlier works of matched offset length, bootstrap 95 percent interval. CLIP unit vectors of real works.
Usage: python band.py EDA_DIR (EDA_DIR holds data.py; prints one block per seed and K)"""
import sys

import numpy as np

M_LOCAL, G_CUTS = 39, (0, 32, 75, 296, 768)
G_NAMES = ('global 1-32', 'global 33-75', 'global 76-296', 'global 297-768')


def one_per_artist(rows, artist, rng, cap=1):
    """Shuffled rows, at most cap per artist."""
    rows = rng.permutation(rows)
    seen, out = {}, []
    for r in rows:
        a = artist[r]
        if seen.get(a, 0) < cap:
            seen[a] = seen.get(a, 0) + 1
            out.append(r)
    return np.array(out, dtype=int)


def shares(x, F, G, gmean, K):
    """local shares (3), global shares (4), offset length, for one test vector x against the fit matrix F (rows unit vectors)."""
    nn = np.argsort(((F - x) ** 2).sum(1))[:K]
    N = F[nn]
    mu = N.mean(0)
    o = x - mu
    tot = float(o @ o)
    Vt = np.linalg.svd(N - mu, full_matrices=False)[2]          # rows: the neighbours' directions, sorted by spread; K rows, the last has no spread
    e = (Vt[:K - 1] @ o) ** 2
    loc = (e[:M_LOCAL].sum() / tot, e[M_LOCAL:].sum() / tot, 1 - e.sum() / tot)
    g = (G @ o) ** 2
    glo = tuple(g[a:b].sum() / tot for a, b in zip(G_CUTS[:-1], G_CUTS[1:]))
    return loc + glo + (np.sqrt(tot),)


def run(X, year, artist, seed, K, n_test=1000, cap=20, n_boot=2000):
    rng = np.random.default_rng(seed)
    dated = ~np.isnan(year) & (year >= 1000) & (year <= 2026) & (artist != '')      # the corpus: a readable year and a named artist
    arts = rng.permutation(np.unique(artist[dated]))
    inA = np.isin(artist, arts[:len(arts) // 2])
    past = dated & (year < 1900)
    fit = one_per_artist(np.flatnonzero(past & inA), artist, rng, cap)
    F = X[fit].astype(np.float64)
    gmean = F.mean(0)
    c = F - gmean
    G = np.ascontiguousarray(np.linalg.eigh(c.T @ c)[1][:, ::-1].T)   # FIT's directions sorted by spread, as rows
    groups = {'held-out earlier works': past & ~inA, 'later works 1900-1969': dated & (year >= 1900) & (year < 1970) & ~inA,
              'later works 1970 on': dated & (year >= 1970) & ~inA}
    names = (f'local 1-{M_LOCAL}', f'local {M_LOCAL + 1}-{K - 1}', 'local rest') + G_NAMES
    width = (M_LOCAL, K - 1 - M_LOCAL, 768 - (K - 1)) + tuple(b - a for a, b in zip(G_CUTS[:-1], G_CUTS[1:]))
    res = {}
    for name, mask in groups.items():
        rows = one_per_artist(np.flatnonzero(mask), artist, rng)[:n_test]
        res[name] = np.array([shares(X[r].astype(np.float64), F, G, gmean, K) for r in rows])
    print(f'\nseed {seed} K {K}: fit works {len(fit)} ({len(np.unique(artist[fit]))} artists); ' + '; '.join(f'{k} n {len(v)} median offset length {np.median(v[:, -1]):.3f}' for k, v in res.items()))
    print('   band (evenly random share)        ' + ' | '.join(f'{k:>24s}' for k in res))
    for j, (nm, w) in enumerate(zip(names, width)):
        print(f'   {nm:16s} ({w / 768:.3f})          ' + ' | '.join(f'median {np.median(v[:, j]):.3f} enrich {np.median(v[:, j]) / (w / 768):5.2f}' for v in res.values()))
    print(f'   inside the neighbours\' own directions (local ranks 1 to {K - 1}; evenly random {(K - 1) / 768:.3f}): ' + ' | '.join(f'{k}: median {np.median(v[:, 0] + v[:, 1]):.3f}' for k, v in res.items()))
    E = res['held-out earlier works']
    order = np.argsort(E[:, -1])
    for name in ('later works 1900-1969', 'later works 1970 on'):
        L = res[name]
        pos = np.clip(np.searchsorted(E[order, -1], L[:, -1]), 1, len(E) - 1)
        pick = np.where(np.abs(E[order[pos], -1] - L[:, -1]) < np.abs(E[order[pos - 1], -1] - L[:, -1]), order[pos], order[pos - 1])   # closest length
        Mt = E[pick]
        print(f'   {name} against length-matched held-out earlier works (matched median length {np.median(Mt[:, -1]):.3f} against {np.median(L[:, -1]):.3f}; distinct earlier works used {len(np.unique(pick))}):')
        idx = rng.integers(0, len(L), (n_boot, len(L)))
        for j, nm in enumerate(names):
            d = np.median(L[:, j]) - np.median(Mt[:, j])
            b = np.median(L[idx, j], axis=1) - np.median(Mt[idx, j], axis=1)
            lo, hi = np.percentile(b, [2.5, 97.5])
            print(f'      {nm:16s} later {np.median(L[:, j]):.3f} matched earlier {np.median(Mt[:, j]):.3f} difference {d:+.3f} [{lo:+.3f}, {hi:+.3f}]')
    return res


if __name__ == '__main__':
    sys.path.insert(0, sys.argv[1])
    import data
    X = data.lens(data.CLIP)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    for seed in (0, 1, 2):
        for K in (100, 300):
            run(X, data.year, data.labels['artist'], seed, K)
