"""Geometry and neighbourhood homophily of the WikiArt corpus in the three encoders: spectrum, dimension, share of the 15 nearest works with the same artist, style, genre or century against chance, k-means agreement with the labels, neighbour overlap between encoders.
Usage: wikiart_geom.py --out DIR [--quick]   (--quick subsamples to a few thousand rows)"""
import argparse
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

ART = os.environ['WIKIART_DIR'] + '/artwork_data'
META = os.environ['WIKIART_DIR'] + '/artist_data'
OBS = os.environ['DUROOM_EMB']
K = 15
QUICK_N = 4000


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=conv)


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def knn(X, k=K, Q=None):
    """k nearest neighbours by L2 for every row of Q (default X, self excluded): (mean distance over the k neighbours, neighbour indices)."""
    X = np.ascontiguousarray(X, dtype='float32')
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    if Q is not None:
        D, I = index.search(np.ascontiguousarray(Q, dtype='float32'), k)
        return np.sqrt(np.maximum(D, 0)).mean(1), I
    D, I = index.search(X, k + 1)
    self_hit = I == np.arange(len(X))[:, None]
    keep = ~self_hit
    keep[~self_hit.any(1), -1] = False  # no self hit (exact duplicate came first): drop the last
    D = np.sqrt(np.maximum(D[keep].reshape(len(X), k), 0))
    return D.mean(1), I[keep].reshape(len(X), k)


def geometry(X, rng, n_id=10000):
    """Spectrum, effective dimension, anisotropy and two intrinsic-dimension estimates."""
    X = X.astype('float64')
    Xc = X - X.mean(0)
    ev = np.linalg.eigvalsh(Xc.T @ Xc / (len(X) - 1))[::-1].clip(min=0)
    share = np.cumsum(ev) / ev.sum()
    U = unit(X)
    a, b = rng.integers(0, len(X), 20000), rng.integers(0, len(X), 20000)
    ok = a != b
    Uc = unit(Xc)
    q = rng.choice(len(X), min(n_id, len(X)), replace=False)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(np.ascontiguousarray(X, dtype='float32'))
    D, I = index.search(np.ascontiguousarray(X[q], dtype='float32'), 21)
    D = np.sqrt(np.maximum(D, 0))
    D = np.where(I == q[:, None], np.nan, D)
    D = np.array([r[~np.isnan(r)][:20] for r in D])  # 20 neighbours, self removed
    good = D.min(1) > 0  # rows with a zero distance (duplicates) are left out
    mu = D[good, 1] / D[good, 0]
    twonn = len(mu) / np.log(mu).sum()  # Facco et al. 2017, maximum-likelihood form
    lb = np.log(D[good, 19:20] / D[good, :19]).mean(1)  # Levina-Bickel, k = 20
    mle = 1.0 / np.mean(lb)  # MacKay-Ghahramani average of inverses
    return {'n': len(X), 'dim': X.shape[1],
            'participation_ratio': float(ev.sum() ** 2 / (ev ** 2).sum()),
            'components_for_80_90_95_pct': [int(np.searchsorted(share, s) + 1) for s in (.8, .9, .95)],
            'top1_variance_share': float(ev[0] / ev.sum()),
            'mean_cosine_random_pairs_raw': float((U[a[ok]] * U[b[ok]]).sum(1).mean()),
            'mean_cosine_random_pairs_centered': float((Uc[a[ok]] * Uc[b[ok]]).sum(1).mean()),
            'intrinsic_dim_twonn': float(twonn), 'intrinsic_dim_mle_k20': float(mle),
            'id_points_used': int(good.sum())}, ev / ev.sum()


def homophily(I, labels):
    """Share of each item's neighbours that carry its label, against the share expected at random."""
    lab = np.asarray(labels, dtype=object)
    valid = np.array([v not in ('', None) and v == v for v in lab])
    li = lab[I]
    vi = valid[I]
    same = (li == lab[:, None]) & vi
    rows = valid & vi.any(1)
    obs = (same[rows].sum(1) / vi[rows].sum(1)).mean()
    p = pd.Series(lab[valid]).value_counts(normalize=True).values
    return {'observed': float(obs), 'random_baseline': float((p ** 2).sum()),
            'lift': float(obs / (p ** 2).sum()), 'n_items': int(rows.sum())}


def kmeans(X, k):
    return MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2000).fit_predict(X)


def nmi_table(labels_by_k, fields):
    out = {}
    for k, lab in labels_by_k.items():
        out[f'k{k}'] = {}
        for name, v in fields.items():
            v = np.asarray(v, dtype=object)
            ok = np.array([x not in ('', None) and x == x for x in v])
            out[f'k{k}'][name] = float(nmi(v[ok].astype(str), lab[ok])) if ok.sum() > 10 else None
    return out


def load_wikiart(quick, rng):
    d = np.load(f'{ART}/artwork_style_embeddings.npz')
    ids = d['ids'].astype(int)
    E = {}
    for n in ['clip-vit-l-14', 'dinov3-l', 'siglip-so400m']:
        o = np.load(f'{OBS}/{n}.npz')
        assert (o['ids'].astype(int) == ids).all(), n
        E[n] = o['embeddings']
    dels = {int(v) for v in open(f'{ART}/delete_ids.txt').read().split()}
    keep = ~np.isin(ids, list(dels))
    df = pd.read_csv(f'{META}/artwork_data_merged.csv', low_memory=False).dropna(subset=['image_n'])
    df['image_n'] = df.image_n.astype(int)
    m = df.drop_duplicates('image_n', keep='first').set_index('image_n').reindex(ids)
    first = lambda s: s.fillna('').astype(str).str.split('|').str[0].str.strip()  # noqa: E731
    meta = pd.DataFrame({'id': ids, 'title': m['Artwork Name'].fillna('').values,
                         'artist': m.Artist_name.fillna('').values,
                         'year': pd.to_numeric(m.Year, errors='coerce').values,
                         'style': first(m.Style).values, 'genre': first(m.Genre).values})
    meta['century'] = (meta.year // 100 * 100).map(lambda v: '' if v != v else str(int(v)))
    rows = np.flatnonzero(keep)  # the corpus without the flagged duplicates
    if quick:
        rows = np.sort(rng.choice(rows, QUICK_N, replace=False))
    E = {k: v[rows].astype('float32') for k, v in E.items()}
    return E, meta.iloc[rows].reset_index(drop=True), rows, keep


def task_wikiart_geom(out, quick, rng):
    E, meta, rows, keep = load_wikiart(quick, rng)
    res = {'rows_used': len(rows), 'rows_dropped_as_duplicates': int((~keep).sum())}
    fields = {'style': meta['style'].values, 'genre': meta['genre'].values,
              'artist': meta['artist'].values, 'century': meta['century'].values}
    spectra, knn_mean = {}, {}
    res['geometry'], res['homophily'], res['clusters_nmi'] = {}, {}, {}
    neighbours = {}
    for name, X in E.items():
        t = time.time()
        g, spec = geometry(X, rng)
        res['geometry'][name] = g
        spectra[name] = spec[:200]
        knn_mean[name], I = knn(X)
        neighbours[name] = I
        res['homophily'][name] = {f: homophily(I, v) for f, v in fields.items()}
        labs = {k: kmeans(X, k) for k in (10, 20)}
        res['clusters_nmi'][name] = nmi_table(labs, fields)
        res.setdefault('seconds', {})[name] = round(time.time() - t, 1)
        print(name, json.dumps(g), flush=True)
        jdump(res, out / 'wikiart_geom.json')
    # neighbour overlap between spaces (same rows, 15-NN sets)
    names = list(neighbours)
    q = rng.choice(len(rows), min(20000, len(rows)), replace=False)
    res['neighbour_overlap_k15'] = {
        f'{a}|{b}': float(np.mean([len(set(neighbours[a][i]) & set(neighbours[b][i])) / K for i in q]))
        for ai, a in enumerate(names) for b in names[ai + 1:]}
    np.savez_compressed(out / 'wikiart_knn15.npz', rows=rows, **{k.replace('-', '_'): v for k, v in knn_mean.items()})
    np.savez_compressed(out / 'wikiart_spectra.npz', **{k.replace('-', '_'): v for k, v in spectra.items()})
    jdump(res, out / 'wikiart_geom.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    out = Path(a.out) / 'wikiart_geom'
    out.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    t0 = time.time()
    print('start wikiart_geom', 'quick' if a.quick else 'full', time.ctime(), flush=True)
    task_wikiart_geom(out, a.quick, np.random.default_rng(0))
    print('done wikiart_geom', round(time.time() - t0, 1), 's', flush=True)
