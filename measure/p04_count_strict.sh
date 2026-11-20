#!/usr/bin/env bash
# Per loop and start, counts of a OFF (onoff_today_v3), b F (t3 measure cell F; also recomputed as f <= 0.7785), c OFF and F, d c and far from today's reference at today's bar in both observers, e eligible (E and F and H); files lined up by picture path.
# Needs onoff_today3_qgt.npz and measure_qgt.npz from run/07_readouts.sh; run in the DUROOM_PY environment.
set -eo pipefail
unset PYTHONPATH
$DUROOM_PY -B - <<'PY'
import os

import numpy as np

B = os.environ['DUROOM_OUT'] + '/tests12'
O = np.load(f'{B}/onoff/out/onoff_today3_qgt.npz', allow_pickle=False)
M = np.load(f'{B}/t3/out/measure_qgt.npz', allow_pickle=False)
print('onoff bars', O['bars'], 'line_T', O['line_T'])
TODAY_BAR = np.array([1.044886, 0.957734])
for arm in ('V', 'C1', 'C2'):
    tags = list(M[f'{arm}__tags'])
    unit, paths = M[f'{arm}__unit'], M[f'{arm}__paths']
    cF, cE, cH, el = M[f'{arm}__cell_F'], M[f'{arm}__cell_E'], M[f'{arm}__cell_H'], M[f'{arm}__eligible']
    pos = {p: i for i, p in enumerate(paths)}
    assert len(pos) == len(paths) == 9600, (arm, len(pos))
    out = {k: [] for k in ('a', 'b', 'b_recomp', 'c', 'd', 'e', 'E', 'E_eq_OFF', 'E_eq_FARm', 'd_avail')}
    for u, tag in enumerate(tags):
        off = O[f'{tag}__OFF']
        farm = O[f'{tag}__FAR_masked']
        op = O[f'{tag}__paths']
        assert len(op) == 1920 and len(set(op)) == 1920 and off.dtype == bool, (arm, tag, 'run length / paths / dtype')
        idx = np.array([pos[p] for p in op])
        assert np.all(unit[idx] == u), (arm, tag, 'unit mismatch')
        f_run = M[f'{arm}__{tag}__f']
        e15 = M[f'{arm}__{tag}__e15']
        mrows = np.where(unit == u)[0]
        mp = paths[mrows]
        assert len(mrows) == 1920
        F_m = cF[idx]
        mpos = {p: i for i, p in enumerate(mp)}
        jdx = np.array([mpos[p] for p in op])
        F_re = f_run[jdx] <= 0.7785
        assert np.array_equal(F_re, F_m), (arm, tag, 'F recomputed differs row-wise from cell_F')
        assert e15.shape == (1920, 2), (arm, tag, e15.shape)
        far_today = np.all(e15[jdx] > TODAY_BAR, axis=1)
        out['a'].append(int(off.sum()))
        out['b'].append(int(F_m.sum()))
        out['b_recomp'].append(int(F_re.sum()))
        out['c'].append(int((off & F_m).sum()))
        out['d'].append(int((off & F_m & far_today).sum()))
        out['e'].append(int(el[idx].sum()))
        out['E'].append(int(cE[idx].sum()))
        out['E_eq_OFF'].append(int((cE[idx] == off).sum()))
        out['E_eq_FARm'].append(int((cE[idx] == farm).sum()))
        out['d_avail'].append(int(far_today.sum()))
        assert np.all(el == (cE & cF & cH)), (arm, 'eligible != E&F&H')
    print(f'== {arm} tags {tags}')
    for k in ('a', 'b', 'b_recomp', 'c', 'd', 'e', 'E', 'E_eq_OFF', 'E_eq_FARm', 'd_avail'):
        print(f'   {k:10s} {out[k]}  pooled {sum(out[k])}')
PY
