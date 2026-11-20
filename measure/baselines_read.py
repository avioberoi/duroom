"""Reads eval/sens_all.npz (baselines_score.py output): the decoder's output for random requests (cone and sphere) and round trips of real P works, against P's leave-one-out q99 bars at k15 and the exact fidelity bar, with the positive control.
Usage: python baselines_read.py sens_all.npz"""
import sys

import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
OBS = ('dinov3-l', 'siglip-so400m', 'aimv2-huge')
bar = {n: float(z[f'bar__{n}__k15__q99']) for n in OBS}
fbar = float(z['bar__fid_exact'])
print('bars k15 q99', {n: round(b, 3) for n, b in bar.items()}, 'fidelity bar', round(fbar, 3))


def passes(prefix, idx=slice(None)):
    d = {n: np.asarray(z[f'{prefix}__dP__{n}__k15'])[idx] for n in OBS}
    both = (d['dinov3-l'] > bar['dinov3-l']) & (d['siglip-so400m'] > bar['siglip-so400m'])
    return d, both, np.asarray(z[f'{prefix}__fid_exact'])[idx] <= fbar, np.asarray(z[f'{prefix}__near_year'])[idx]


print('\n== prior: the decoder painting random requests (cone = around P\'s mean at P\'s median angle, sphere = uniform); pass vs P alone')
tagp = np.asarray(z['base_prior__tag'])
for t in ('cone', 'sphere'):
    m = tagp == t
    d, pb_, pf_, py_ = passes('base_prior', m)
    yrs = py_[np.isfinite(py_)]
    print(f'{t}: n {m.sum()}; pass both {pb_.mean():.3f} (DINOv3 {(d["dinov3-l"] > bar["dinov3-l"]).mean():.3f}, SigLIP {(d["siglip-so400m"] > bar["siglip-so400m"]).mean():.3f}, '
          f'AIMv2 {(d["aimv2-huge"] > bar["aimv2-huge"]).mean():.3f}); faithful to the random request {pf_.mean():.3f}; far and faithful {(pb_ & pf_).mean():.3f}; '
          f'median dP DINOv3 {np.median(d["dinov3-l"]):.3f} SigLIP {np.median(d["siglip-so400m"]):.3f}; nearest work year median {np.median(yrs):.0f}, share 1900 on {(yrs >= 1900).mean():.2f}')

print('\n== chains: 48 real P works, re-encoded and repainted 7 times with no selection (step 0 = the work\'s own render)')
step = np.asarray(z['base_chains_h200_qg__step'])
for s in sorted(set(step.tolist())):
    m = step == s
    d, pb_, pf_, py_ = passes('base_chains_h200_qg', m)
    yrs = py_[np.isfinite(py_)]
    print(f'step {s}: n {m.sum()}; median dP DINOv3 {np.median(d["dinov3-l"]):.3f} SigLIP {np.median(d["siglip-so400m"]):.3f} AIMv2 {np.median(d["aimv2-huge"]):.3f}; '
          f'pass both {pb_.mean():.3f}; faithful to its request {pf_.mean():.2f}; nearest year median {np.median(yrs):.0f}, share 1900 on {(yrs >= 1900).mean():.2f}')

print('\n== the positive control')
d, pb_, pf_, py_ = passes('pos')
yrs = py_[np.isfinite(py_)]
print(f'pos (512 renders of 1900-69 works): pass both {pb_.mean():.3f}, faithful {pf_.mean():.2f}, far and faithful {(pb_ & pf_).mean():.3f}; nearest year median {np.median(yrs):.0f}, share 1900 on {(yrs >= 1900).mean():.2f}')
