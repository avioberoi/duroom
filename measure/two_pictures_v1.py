"""Far scores and the faithful reading of two search pictures: the picture painted in round 15 of qgc_ra_s10 (row 499) and in round 7 of qgc_ra_s6 (row 246), against the reference alone.
Usage: python -B two_pictures_v1.py (prints, per picture and observer, the far score against the bar, the far flag of the evaluator, and the exact-reading chord against the faithful line)"""
import importlib.util
import os

import numpy as np

B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
f = f'{C_}/controls_evaluate.py'
spec = importlib.util.spec_from_file_location('ev', f); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
P, T, Pz = ev.references(O); ids, q = ev.startids(O), ev.qfun(O)
for pid, tag, row in (('026', 'qgc_ra_s10', 499), ('f_s6_0245', 'qgc_ra_s6', 246)):
    d = f'{O}/runs/{tag}_committee_euclidean_min_seedrow'
    r = ev.load(f'{tag}={d},{O}/eval/{tag}_emb.npz,{B}/logs/{tag}.out', P, T, ids, q, conf=True)
    z = r['z']; m = z['iteration'] >= 0; i = row - 1   # position among the 1,920 (row 0 = start)
    assert m[row] and int(z['iteration'][row]) == (15 if pid == '026' else 7)
    emb = {f'emb_{o}': z[f'emb_{o}'][m] for o in ev.OBS}
    sP = ev.score(P, emb)
    print(pid, tag, 'row', row, 'round', int(z['iteration'][row]), 'path', str(z['paths'][row]).split('/')[-1])
    for o in ev.OBS:
        print(f'   {o}: far score (mean chord to 15 nearest P, reference alone) {sP[o][0][i]:.4f} vs bar {P[o]["bar"]:.4f} -> {"far" if sP[o][0][i] > P[o]["bar"] else "near"}')
    print(f'   far in both (evaluator F) {bool(r["F"][i])}; faithful (exact CLIP, chord {float(ev.chord((ev.unit(z["style_embs"][row]) * ev.unit(np.load(f"{O}/eval/{tag}_emb.npz")["emb_clip-vit-l-14"][row])).sum())):.4f} vs {ev.FID:.4f}) {bool(r["ok_exact"][i])}')
