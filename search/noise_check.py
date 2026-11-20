"""Scores the 160 noise embeddings against the 1900 reference with today_rescore.reference / score / far and prints how many count as far.
Usage: python noise_check.py   (CPU; paths fixed below)"""
import os
import sys

import numpy as np

B = os.environ['DUROOM_OUT']
C = os.path.join(os.environ['DUROOM_REPO'], 'measure')
O = f'{B}/sphere'
sys.path.insert(0, C)
import today_rescore as tr  # noqa: E402
from local_rescore import OBS  # noqa: E402

Pz = np.load(f'{O}/push_emb.npz', allow_pickle=True)
P = tr.reference(Pz)
E = np.load(f'{O}/eval/bound/noise_check/noise_emb.npz', allow_pickle=True)
print('thresholds: ' + ', '.join(f'{o} {P[o]["bar"]:.4f}' for o in OBS))
paths = [str(p) for p in E['paths']]
kind = np.array([p.rsplit('/', 1)[1][0] for p in paths])
assert len(paths) == 160 and all((kind == k).sum() == 32 for k in 'abcde')
NAMES = {'a': 'a uniform pixel noise', 'b': 'b gaussian pixel noise', 'c': 'c solid colour', 'd': 'd smooth low-freq', 'e': 'e 8x8 colour blocks'}
sc = tr.score(P, {f'emb_{o}': E[f'emb_{o}'] for o in OBS})
far = tr.far(sc, P)
d = {o: sc[o][0] / P[o]['bar'] for o in OBS}


def row(name, k):
    over = {o: d[o][k] > 1 for o in OBS}
    s = f'{name:24s} n {int(k.sum()):4d} | far (both) {int(far[k].sum()):4d} | over dinov3 {int(over[OBS[0]].sum()):4d} (only dinov3 {int((over[OBS[0]] & ~over[OBS[1]]).sum()):4d}) | over siglip {int(over[OBS[1]].sum()):4d} (only siglip {int((over[OBS[1]] & ~over[OBS[0]]).sum()):4d})'
    for o in OBS:
        v = d[o][k]
        s += f' | {o} med {np.median(v):.3f} min {v.min():.3f} max {v.max():.3f}'
    print(s)


print('distance / threshold (1.0 = at the threshold); "over dinov3/siglip" = over that observer (the other either way), "only" = over it and not over the other')
for kk in 'abcde':
    row(NAMES[kk], kind == kk)
row('all 160', np.ones(160, bool))

# sanity: the reference's own first 200 pictures; each is its own nearest neighbour (chord 0), so its artist's rows are excluded via `artist`
n = 200
own = {f'emb_{o}': Pz[f'emb_{o}'][:n] for o in OBS}
s2 = tr.score(P, own, artist=P['artist'][:n])
f2 = tr.far(s2, P)
d2 = {o: s2[o][0] / P[o]['bar'] for o in OBS}
print('sanity: reference first 200 pictures, same-artist rows excluded (leave-one-out as in reference())')
print(f'{"reference first 200":24s} n {n:4d} | far (both) {int(f2.sum()):4d} | over dinov3 {int((d2[OBS[0]] > 1).sum()):4d} | over siglip {int((d2[OBS[1]] > 1).sum()):4d}' + ''.join(
    f' | {o} med {np.median(d2[o]):.3f} min {d2[o].min():.3f} max {d2[o].max():.3f}' for o in OBS))
print('sanity check: first-200 distances equal reference()["own"] per observer:', {o: bool(np.allclose(s2[o][0], P[o]['own'][:n], atol=1e-6)) for o in OBS})
