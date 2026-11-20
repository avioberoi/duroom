"""Chord from sets of renders to the nearest corpus work of any era, of the pre-1900 era and of 1900 on (DINOv3, SigLIP), with the source work or its artist masked.
Usage (CPU): python on_support.py OUT_NPZ"""
import glob
import sys
import os
import numpy as np

O = os.environ['DUROOM_OUT'] + '/sphere'
OBS = os.environ['DUROOM_EMB']
sys.path.insert(0, os.environ['DUROOM_REPO'] + '/data')
from data import ids, labels, lens, year  # noqa: E402

LENS = ('dinov3-l', 'siglip-so400m')


def unit(X):
    X = np.asarray(X, np.float32)
    U = X / np.linalg.norm(X, axis=-1, keepdims=True)
    assert np.isfinite(U).all(), 'zero-norm or non-finite embedding'
    return U


corpus = {n: unit(lens(f'{OBS}/{n}.npz')) for n in LENS}
yr = np.asarray(year, dtype=np.float64)
dated = np.isfinite(yr) & (yr > 0)
pre, post = dated & (yr < 1900), dated & (yr >= 1900)
cid = np.array([str(i) for i in ids])
row_of = {i: r for r, i in enumerate(cid)}
artist = np.asarray(labels['artist']).astype(str)
rows_of_artist = {}
for r, a in enumerate(artist):
    if a:
        rows_of_artist.setdefault(a, []).append(r)
print('corpus', {n: corpus[n].shape for n in LENS}, 'pre-1900', int(pre.sum()), '1900 on', int(post.sum()), 'undated', int((~dated).sum()), flush=True)
raw_ids = np.array([str(i) for i in np.load(f'{OBS}/dinov3-l.npz', allow_pickle=True)['ids']])   # unfiltered observer order, which the runs' start_idx indexes


def src_rows(folder, paths):
    m = {}
    for f in sorted(glob.glob(f'{O}/{folder}/shard*.npz')):
        s = np.load(f, allow_pickle=True)
        m.update({str(p): str(i) for p, i in zip(s['paths'], s['ids'])})
    src = np.array([row_of.get(m.get(str(p), ''), -1) for p in paths])
    return src


def nearest(E, masks):
    """masks: one list of corpus rows to exclude per item (may be empty). Returns chords to the nearest row overall / pre / post."""
    out = {}
    for n in LENS:
        X = unit(E[n]); C = corpus[n]
        best = {k: np.empty(len(X), np.float32) for k in ('all', 'pre', 'post')}
        for i in range(0, len(X), 256):
            S = X[i:i + 256] @ C.T
            for j, rows in enumerate(masks[i:i + 256]):
                if len(rows):
                    S[j, rows] = -2.0
            best['all'][i:i + 256] = S[:, dated].max(1)
            best['pre'][i:i + 256] = S[:, pre].max(1)
            best['post'][i:i + 256] = S[:, post].max(1)
        for k, v in best.items():
            out[f'{n}__{k}'] = np.sqrt(np.clip(2 - 2 * v, 0, None)).astype(np.float32)
    return out


res = {}
def store(name, r):  # noqa: E302
    for k, v in r.items():
        res[f'{name}__{k}'] = v
    print(f"{name}: n {len(r['dinov3-l__all'])}; nearest dated work DINOv3 median {np.median(r['dinov3-l__all']):.3f}, SigLIP {np.median(r['siglip-so400m__all']):.3f}; "
          f"nearest pre-1900 {np.median(r['dinov3-l__pre']):.3f} / {np.median(r['siglip-so400m__pre']):.3f}; nearest 1900-on {np.median(r['dinov3-l__post']):.3f} / {np.median(r['siglip-so400m__post']):.3f}", flush=True)


for name, file, folder in (('P', 'push_emb.npz', 'push'), ('P2', 'push2_emb.npz', 'push2'), ('pos', 'pos_emb.npz', 'pos')):
    z = np.load(f'{O}/{file}', allow_pickle=True)
    src = src_rows(folder, z['paths'])
    share = float((src >= 0).mean())
    print(name, 'rows', len(src), 'source found', round(share, 4), flush=True)
    assert share > 0.99, f'{name}: source rows found for only {share:.3f}'
    keep = src >= 0
    E = {n: np.asarray(z[f'emb_{n}'])[keep] for n in LENS}
    s = src[keep]
    store(f'{name}_srcout', nearest(E, [[r] for r in s]))
    store(f'{name}_artistout', nearest(E, [rows_of_artist.get(artist[r], [r]) for r in s]))
names = sorted({k.split('__')[0] for k in res})
for n in LENS:
    for bname in ('srcout', 'artistout'):
        base = np.concatenate([res[f'P_{bname}__{n}__all'], res[f'P2_{bname}__{n}__all']])
        q = np.nanpercentile(base, [50, 95, 99])
        res[f'baseline_{bname}__{n}'] = q.astype(np.float32)
        print(f'baseline {bname} {n} (P + P2 renders to the nearest other dated real work): p50 {q[0]:.3f} p95 {q[1]:.3f} p99 {q[2]:.3f}', flush=True)
        for name in names:
            v = res[f'{name}__{n}__all']
            print(f'   {name}: within baseline p95 {np.mean(v <= q[1]):.2f}, within p99 {np.mean(v <= q[2]):.2f}', flush=True)
np.savez_compressed(sys.argv[1], **res)
print('saved', sys.argv[1])
