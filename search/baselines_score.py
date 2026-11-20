"""Scores runs, baselines, the positive control and calibration against P: exact fidelity, mean chord to the k=10/15/20 nearest P renders in DINOv3, SigLIP, AIMv2, and P's q95-q99.5 bars (CPU).
Usage: python baselines_score.py OUT_NPZ [--runs TAG ...] [--baselines DIR ...]"""
import argparse
import os
import sys

import numpy as np

O = os.environ['DUROOM_OUT'] + '/sphere'
OBS = ('dinov3-l', 'siglip-so400m', 'aimv2-huge')
KS, QS = (10, 15, 20), (95, 98, 99, 99.5)
unit = lambda X: np.asarray(X, np.float64) / np.linalg.norm(np.asarray(X, np.float64), axis=-1, keepdims=True)  # noqa: E731
chord = lambda c: np.sqrt(np.clip(2 - 2 * c, 0, None))  # noqa: E731
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--runs', nargs='*', default=[])
ap.add_argument('--baselines', nargs='*', default=[])
ap.add_argument('--data-dir', default=os.path.join(os.environ['DUROOM_REPO'], 'data'))
a = ap.parse_args()

P = np.load(f'{O}/push_emb.npz', allow_pickle=True)
Pe = {n: unit(P[f'emb_{n}']) for n in OBS}
same = P['artist'][:, None] == P['artist'][None, :]
res = {}
for n in OBS:
    Dm = chord(Pe[n] @ Pe[n].T); np.fill_diagonal(Dm, np.inf); Dm[same] = np.inf
    S = np.sort(Dm, axis=1)
    for k in KS:
        for q in QS:
            res[f'bar__{n}__k{k}__q{q}'] = float(np.percentile(S[:, :k].mean(1), q))
res['bar__fid_exact'] = float(np.percentile(P['fidelity_clip'], 99))
sys.path.insert(0, a.data_dir)
from data import lens, year  # noqa: E402
corpus = unit(lens(os.environ['DUROOM_EMB'] + '/dinov3-l.npz').astype(np.float32)).astype(np.float32)


def score(prefix, E, V):
    """E: embedding npz-like with emb_* rows aligned to V (requested unit vectors, or None)."""
    for n in OBS:
        S = np.sort(chord(unit(E[f'emb_{n}']) @ Pe[n].T), axis=1)
        for k in KS:
            res[f'{prefix}__dP__{n}__k{k}'] = S[:, :k].mean(1).astype(np.float32)
    if V is not None:
        res[f'{prefix}__fid_exact'] = chord((unit(V) * unit(E['emb_clip-vit-l-14'])).sum(1)).astype(np.float32)
    Rj = unit(E['emb_dinov3-l']).astype(np.float32)
    near = np.concatenate([np.argmax(Rj[i:i + 256] @ corpus.T, axis=1) for i in range(0, len(Rj), 256)])   # chunked: the corpus is 172k rows
    res[f'{prefix}__near_year'] = np.asarray(year)[near].astype(np.float32)


for tag in a.runs:
    d = f'{O}/runs/{tag}_committee_euclidean_min_seedrow'
    f = f'{d}/all_candidates.npz' if os.path.exists(f'{d}/all_candidates.npz') else f'{d}/all_draws.npz'
    z, e = np.load(f, allow_pickle=True), np.load(f'{O}/eval/{tag}_emb.npz', allow_pickle=True)
    assert len(z['paths']) == len(e['paths']) and all(str(x) == str(y) for x, y in zip(z['paths'], e['paths'])), tag
    score(f'run_{tag}', e, z['style_embs'])
    res[f'run_{tag}__iteration'] = np.asarray(z['iteration']); res[f'run_{tag}__admitted'] = np.asarray(z['admitted']).astype(bool)
    print('scored run', tag, flush=True)
for name, file in (('pos', f'{O}/pos_emb.npz'), ('P2', f'{O}/push2_emb.npz')):
    z = np.load(file, allow_pickle=True)
    score(name, z, z['vec'])
    print('scored', name, flush=True)
for d in a.baselines:
    m, e = np.load(f'{d}/manifest.npz', allow_pickle=True), np.load(f'{d}/emb.npz', allow_pickle=True)
    assert len(m['paths']) == len(e['paths']) and all(str(x) == str(y) for x, y in zip(m['paths'], e['paths'])), d
    key = 'base_' + os.path.basename(d.rstrip('/'))
    score(key, e, m['v'])
    for k in ('kind', 'tag', 'row', 'seed', 'step'):
        res[f'{key}__{k}'] = m[k]
    print('scored baseline', key, flush=True)
np.savez_compressed(a.out, **res)
q = {n: res[f'bar__{n}__k15__q99'] for n in OBS}
print('bars k15 q99', q, 'fid', res['bar__fid_exact'])
for key in sorted({k.split('__')[0] for k in res if not k.startswith('bar')}):
    dn, ds, da = (res[f'{key}__dP__{n}__k15'] for n in OBS)
    both = (dn > q['dinov3-l']) & (ds > q['siglip-so400m'])
    fid = res.get(f'{key}__fid_exact')
    msg = f'{key}: n {len(dn)}, pass both {both.mean():.3f}, pass AIMv2 {(da > q["aimv2-huge"]).mean():.3f}, pass all three {(both & (da > q["aimv2-huge"])).mean():.3f}'
    if fid is not None:
        msg += f', faithful {(fid <= res["bar__fid_exact"]).mean():.3f}, faithful and pass both {(both & (fid <= res["bar__fid_exact"])).mean():.3f}'
    print(msg, flush=True)
