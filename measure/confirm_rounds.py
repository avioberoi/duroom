"""Per-round counts for the 15 main runs from the evaluator's own load(), stats() and quality(): far, H, distinct hits (DH) and quality hits (QH) per round of 32 pictures; prints the JSON.
Usage (CPU): python -B confirm_rounds.py > confirm_rounds.json"""
import json
import os
import numpy as np
import controls_evaluate as ce

B = os.environ['DUROOM_OUT']
O = f'{B}/sphere'
R, E, L = f'{O}/runs', f'{O}/eval', f'{B}/logs'


ARM = {'ra': 'search', 'pin': "random scores with the search's step sizes", 'c2': 'round trips with no search'}
CACHE = f'{O}/eval/bound/ev_safety_conf.json'
cache = json.load(open(CACHE))
P, T, Pz = ce.references(O)
ids, q = ce.startids(O), ce.qfun(O)
sup = ce.load_support()


def cell(arm, N):
    tag = f'qgc_{arm}_s{N}'
    return ce.load(f'{tag}={R}/{tag}_committee_euclidean_min_seedrow,{E}/{tag}_emb.npz,{L}/{tag}.out', P, T, ids, q, conf=True)


def per_round(it, flag):
    """counts of `flag` (length 1920, loop order) per iteration value 0..59; asserts 32 pictures in every round."""
    assert it.shape == flag.shape == (1920,)
    assert all(int((it == k).sum()) == 32 for k in range(60)), 'a round does not have 32 pictures'
    return [int(flag[it == k].sum()) for k in range(60)]


cells = {}
for arm in ('ra', 'pin', 'c2'):
    for N in (6, 7, 8, 9, 10):
        tag = f'qgc_{arm}_s{N}'
        r = cell(arm, N)
        s = ce.quality(r, ce.stats(r, 'exact'), sup, cache, None)
        z = r['z']
        it = z['iteration'][z['iteration'] >= 0].astype(int)
        far = per_round(it, r['F'])
        h = per_round(it, r['F'] & r['ok_exact'] & r['C'])
        dh = per_round(it, s['DH'])
        qhflag = np.zeros(1920, bool)
        qhflag[np.flatnonzero(s['DH'])[s['QH']]] = True
        assert int(qhflag.sum()) == s['nQH'] and set((1 + np.flatnonzero(qhflag)).tolist()) == set(int(x) for x in s['qrows']), tag
        qh = per_round(it, qhflag)
        assert sum(h) == s['n_H']
        assert sum(far) == int(r['F'].sum()) and sum(dh) == s['nDH'] and sum(qh) == s['nQH']
        cells[tag] = dict(arm=ARM[arm], start=int(r['wid']), seed=N, far=far, h=h, dh_new=dh, qh_new=qh)
out = dict(rounds=60, per_round=32, cells=cells)
print(json.dumps(out), flush=True)
