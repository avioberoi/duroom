"""Are the 160 noise images covered (not far against today's reference T at the 1900 bars in at least one observer, and later share > 0.5)? Per noise kind: far, covered, far and covered, not far from T, median later share.
Usage (CPU): python -B noise_covered.py"""
import os

import numpy as np

O = os.environ['DUROOM_OUT'] + '/sphere'
import controls_evaluate as ev  # noqa: E402
from local_rescore import OBS  # noqa: E402
from today_rescore import far, score  # noqa: E402

P, T, _ = ev.references(O)
E = np.load(f'{O}/eval/bound/noise_check/noise_emb.npz', allow_pickle=True)
paths = [str(p) for p in E['paths']]
kind = np.array([p.rsplit('/', 1)[1][0] for p in paths])
assert len(paths) == 160 and all((kind == k).sum() == 32 for k in 'abcde')
emb = {f'emb_{o}': E[f'emb_{o}'] for o in OBS}
sP, sT = score(P, emb), score(T, emb)
F = far(sP, P)
notT = ~far(sT, P)
later = np.mean([(T['year'][sT[o][1]] >= 1900).mean(1) for o in OBS], axis=0)
Cv = notT & (later > 0.5)
print('thresholds: ' + ', '.join(f'{o} {P[o]["bar"]:.4f}' for o in OBS) + f' | today reference rows {len(T["year"])}, from 1900 on {int((T["year"] >= 1900).sum())}')
print('far = over the 1900 thresholds from P in both observers; covered = C (not far from T at the 1900 thresholds in at least one observer, and later share > 0.5); '
      'later share = mean over the two observers of the share of the 15 nearest T rows dated 1900 or later')
NAMES = {'a': 'a uniform pixel noise', 'b': 'b gaussian pixel noise', 'c': 'c solid colour', 'd': 'd smooth low-freq', 'e': 'e 8x8 colour blocks'}


def row(name, k):
    c = Cv[k]
    med = f'{np.median(later[k][c]):.3f}' if c.any() else 'n/a'
    print(f'{name:24s} n {int(k.sum()):4d} | far {int(F[k].sum()):4d} | covered {int(c.sum()):4d} | far AND covered {int((F[k] & c).sum()):4d} | '
          f'not far from T {int(notT[k].sum()):4d} | median later share, covered {med} | median later share, all {np.median(later[k]):.3f} (min {later[k].min():.3f} max {later[k].max():.3f})')


for kk in 'abcde':
    row(NAMES[kk], kind == kk)
row('all 160', np.ones(160, bool))
