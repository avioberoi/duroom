"""Bar-level sweep: far, far-and-faithful and AIMv2-far shares per prefix at k 10, 15, 20 and bar level q 95 to 99.5, on all candidates and on admitted rows, from sens_runs.npz and sens_all.npz (baselines_score.py).
Usage: python sb_grid.py (reads $DUROOM_OUT/sphere/eval/sens_all.npz and sens_runs.npz when present; writes eval/sb_grid.json)"""
import json
import os

import numpy as np

EV = os.environ['DUROOM_OUT'] + '/sphere/eval'
files = [f'{EV}/sens_runs.npz', f'{EV}/sens_all.npz']
Z = {}
for f in files:
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            Z.setdefault(k, z[k])
KS, QS = (10, 15, 20), ('95', '98', '99', '99.5')
DN, SG, AI = 'dinov3-l', 'siglip-so400m', 'aimv2-huge'
FID = float(Z['bar__fid_exact'])
prefixes = sorted({k.split('__')[0] for k in Z if not k.startswith('bar__') and not k.startswith('base_')})


def bar(o, k, q):
    return float(Z[f'bar__{o}__k{k}__q{q}'])


def masks(p, k, q):
    far = (Z[f'{p}__dP__{DN}__k{k}'] > bar(DN, k, q)) & (Z[f'{p}__dP__{SG}__k{k}'] > bar(SG, k, q))
    ff = far & (Z[f'{p}__fid_exact'] <= FID) if f'{p}__fid_exact' in Z else None
    ai = Z[f'{p}__dP__{AI}__k{k}'] > bar(AI, k, q)
    return far, ff, ai


out = {'fid_bar': FID, 'cells': {}}
for p in prefixes:
    sel = np.asarray(Z[f'{p}__iteration']) >= 0 if f'{p}__iteration' in Z else np.ones(len(Z[f'{p}__dP__{DN}__k15']), bool)
    adm = (np.asarray(Z[f'{p}__admitted']).astype(bool) & sel) if f'{p}__admitted' in Z else None
    for k in KS:
        for q in QS:
            far, ff, ai = masks(p, k, q)
            c = {'n': int(sel.sum()), 'far': float(far[sel].mean()), 'ff': None if ff is None else float(ff[sel].mean()), 'ai': float(ai[sel].mean())}
            if adm is not None and adm.sum():
                c.update(n_adm=int(adm.sum()), adm_far=float(far[adm].mean()), adm_ff=None if ff is None else float(ff[adm].mean()),
                         adm_ai=float(ai[adm].mean()))
            out['cells'][f'{p}|k{k}|q{q}'] = c

C = out['cells']
cells = [(k, q) for k in KS for q in QS]
json.dump(out, open(f'{EV}/sb_grid.json', 'w'), indent=1)

print('fid bar', round(FID, 4))
print(f"{'prefix':16s} " + ' '.join(f'k{k}q{q}'.rjust(16) for k, q in cells))
for m in ('far', 'ff', 'ai'):
    print(f'--- {m} (all candidates; {m}) ---')
    for p in prefixes:
        vals = [C[f'{p}|k{k}|q{q}'][m] for k, q in cells]
        print(f'{p:16s} ' + ' '.join(('   -' if v is None else f'{100 * v:5.1f}').rjust(16) for v in vals))
print('--- admitted rows: far / ff / ai at k15 q99 ---')
for p in prefixes:
    c = C[f'{p}|k15|q99']
    if 'n_adm' in c:
        print(f"{p:16s} n_adm {c['n_adm']:4d} far {100 * c['adm_far']:5.1f} ff {100 * c['adm_ff']:5.1f} ai {100 * c['adm_ai']:5.1f}")
