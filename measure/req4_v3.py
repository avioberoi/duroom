"""Shift score q, coverage against today's reference and the per-round CSV for the 15 main runs: q on the exact CLIP reading per arm and start (repainted real works by date as the scale), notT / later / covered for far-and-faithful candidates and far repaints of later works.
Usage: python -B req4_v3.py OUTPREFIX ROUNDS_CSV LF_CMA.npz LF_MAIN.npz (the lf_cma_v2.py and lf_main_v2.py outputs)"""
import csv
import importlib.util
import io
import os
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C, O, L = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere', f'{B}/logs'
EV_FILE = f'{C}/controls_evaluate.py'
ARMS_NPZ = f'{B}/tests12/arms/arms_barfree_v1.npz'
ARMS = (('V', 'ra'), ('C1', 'pin'), ('C2', 'c2'))
STARTS = (6, 7, 8, 9, 10)
pid = lambda paths: np.array([int(re.search(r'push_(\d+)\.png$', str(p)).group(1)) for p in paths])  # noqa: E731
OUT = []


def say(s=''):
    OUT.append(s)


def q3(v):
    v = np.asarray(v, float)
    if not len(v):
        return 'n     0'
    p = np.percentile(v, (10, 50, 90))
    return f'n {len(v):5d} | p10 {p[0]:+.4f} median {p[1]:+.4f} p90 {p[2]:+.4f}'


def main(prefix, rounds_csv, LFA, LFB):
    for f in (prefix + '.txt', prefix + '_STOP.txt', rounds_csv):
        assert not os.path.exists(f), f
    spec = importlib.util.spec_from_file_location('controls_evaluate', EV_FILE)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    from local_rescore import auc
    OBS, K, unit, chord, FID = ev.OBS, ev.K, ev.unit, ev.chord, float(ev.FID)
    P, T, Pz = ev.references(O)
    ids, qf = ev.startids(O), ev.qfun(O)
    A, LA, LB = np.load(ARMS_NPZ), np.load(LFA), np.load(LFB)
    say('req4_v3.py. Counts and distributions only.')
    say('Hits = DH (the evaluator\'s distinct hits: admitted by the replay, faithful, covered). Real works by date: a three-way split only (before 1900, 1900-1969, 1970 on; pool years).')
    say('Runs: the 15 main runs (V = qgc_ra_s6..10, C1 = qgc_pin_s6..10, C2 = qgc_c2_s6..10). Reference for far: P (push_emb.npz); today reference T = push_today_emb.npz, years from pools/today_cap3.npz.')
    say()

    # the 15 cells through the evaluator's own load (confirmation mode)
    R = {}
    for arm, tg in ARMS:
        for n in STARTS:
            cell = f'qgc_{tg}_s{n}'
            spec_ = f'{cell}={O}/runs/{cell}_committee_euclidean_min_seedrow,{O}/eval/{cell}_emb.npz,{L}/{cell}.out'
            r = ev.load(spec_, P, T, ids, qf, conf=True)
            assert (r['F'] == A[f'{cell}__F'].astype(bool)).all() and (r['ev'] == A[f'{cell}__EV'].astype(bool)).all()
            DF, DH = r['ev'] & r['ok_exact'], r['ev'] & r['ok_exact'] & r['C']
            assert (DH == A[f'{cell}__DH'].astype(bool)).all()
            R[cell] = dict(arm=arm, n=n, q=np.asarray(r['q_exact'], float), FF=r['F'] & r['ok_exact'], DF=DF, DH=DH, notT=r['notT'], later=np.asarray(r['later'], float), C=r['C'],
                           z=r['z'], wid=r['wid'])
    say('anchors: per cell the evaluator\'s F, admitted and DH flags equal the arms npz')
    say()

    say('TABLE 4.2a shift score q on the exact CLIP reading; per arm and start: all candidates | far-and-faithful (far from P at P99 in both observers, exact gap <= FID) | DH hits')
    for arm, tg in ARMS:
        for n in list(STARTS) + ['pool']:
            cs = [f'qgc_{tg}_s{k}' for k in (STARTS if n == 'pool' else [n])]
            cat = lambda key, sel=None: np.concatenate([R[c]['q'][R[c][sel]] if sel else R[c]['q'] for c in cs])  # noqa: E731
            say(f'   {arm:2s} {("s" + str(n)) if n != "pool" else "pool":4s} all {q3(cat("q"))} || far-and-faithful {q3(cat("q", "FF"))} || DH hits {q3(cat("q", "DH"))}')
    say()
    say('TABLE 4.2b per start: P(a random V candidate has a larger q than a random control candidate of the same start), ties half; 0.5 = no difference')
    for sel, lab in ((None, 'all candidates'), ('FF', 'far-and-faithful candidates'), ('DH', 'DH hits')):
        for x, tx in (('C1', 'pin'), ('C2', 'c2')):
            v = []
            for n in STARTS:
                a_, b_ = R[f'qgc_ra_s{n}'], R[f'qgc_{tx}_s{n}']
                v.append(auc(a_['q'][a_[sel]] if sel else a_['q'], b_['q'][b_[sel]] if sel else b_['q']))
            say(f'   {lab:45s} V vs {x}: ' + ', '.join(f's{n} {u:.4f}' for n, u in zip(STARTS, v)) + f' | mean {np.mean(v):.4f}')
    say()
    say('TABLE 4.2c the same q for repainted real works (scale; three-way date split): on the exact CLIP reading of the repaint (emb_clip-vit-l-14) | on the request (vec = the original\'s stored CLIP vector)')
    pools = {k: np.load(f'{O}/pools/{k}.npz', allow_pickle=True) for k in ('pre1900_cap3', 'pos_1900_1969_cap3', 'today_cap3')}
    yr = {k: dict(zip(p['ids'].astype(int).tolist(), p['year'].astype(float).tolist())) for k, p in pools.items()}
    REP = {}
    for name, f, pool in (('P (reference)', 'push_emb.npz', 'pre1900_cap3'), ('P2 (before 1900)', 'push2_emb.npz', 'pre1900_cap3'), ('pos (1900-1969)', 'pos_emb.npz', 'pos_1900_1969_cap3'),
                          ('today', 'push_today_emb.npz', 'today_cap3')):
        z = np.load(f'{O}/{f}', allow_pickle=True)
        idv = pid(z['paths'])
        year = np.array([yr[pool][int(i)] for i in idv])
        sels = [(name, np.ones(len(idv), bool))] if name != 'today' else [('today, before 1900', year < 1900), ('today, 1900-1969', (year >= 1900) & (year < 1970)), ('today, 1970 on', year >= 1970)]
        for lab, s in sels:
            REP[lab] = dict(z=z, sel=s, ids=idv[s])
            say(f'   {lab:22s} reading {q3(qf(z["emb_clip-vit-l-14"][s]))} || request {q3(qf(z["vec"][s]))}')
    say()

    def t_score(emb, own_ids):
        """per observer, mean chord to the 15 nearest T pictures, the row whose work id equals own_ids[i] left out (the picture itself, for a today repaint);
        returns notT (not far against T at P's bars, both observers) and later (mean over observers of the share of the 15 dated 1900 or later)"""
        far_T, later = np.ones(len(own_ids), bool), np.zeros(len(own_ids))
        for o in OBS:
            D = chord(unit(emb[o]) @ T[o]['E'].T)
            for i, w in enumerate(own_ids):
                D[i, T['id'] == w] = np.inf
            nn = np.argsort(D, axis=1)[:, :K]
            far_T &= np.take_along_axis(D, nn, 1).mean(1) > P[o]['bar']
            later += (T['year'][nn] >= 1900).mean(1) / len(OBS)
        return ~far_T, later
    for c in list(R)[:3] + list(R)[-1:]:   # check: t_score reproduces the evaluator's notT and later (no own row: the start is not in T)
        z = R[c]['z']
        m = z['iteration'] >= 0
        nt, la = t_score({o: z[f'emb_{o}'][m] for o in OBS}, np.full(1920, -1))
        assert (nt == R[c]['notT']).all() and np.allclose(la, R[c]['later'], rtol=0, atol=1e-12), c
    say('check: the T score written here reproduces the evaluator\'s notT and later on 4 cells (exact / 1e-12)')
    say('TABLE 4.3a candidates; per arm and start. far-and-faithful (F and exact gap <= FID): n, not far against T, median later-share, covered (not far against T and later > 0.5) | '
        'evaluator-admitted and faithful (DF): n, covered (= DH)')
    for arm, tg in ARMS:
        tot = np.zeros(5, int)
        for n in STARTS:
            r = R[f'qgc_{tg}_s{n}']
            ff, df = r['FF'], r['DF']
            row = [int(ff.sum()), int((ff & r['notT']).sum()), int((ff & r['C']).sum()), int(df.sum()), int((df & r['C']).sum())]
            tot += row
            say(f'   {arm:2s} s{n:<2d} far-and-faithful {row[0]:4d}, not far against T {row[1]:4d}, median later {np.median(r["later"][ff]) if ff.any() else float("nan"):.3f}, covered {row[2]:4d} | DF {row[3]:3d} -> covered {row[4]:3d}')
        lat = np.concatenate([R[f'qgc_{tg}_s{n}']['later'][R[f'qgc_{tg}_s{n}']['FF']] for n in STARTS])
        say(f'   {arm:2s} pool far-and-faithful {tot[0]:4d}, not far against T {tot[1]:4d}, median later {np.median(lat):.3f}, covered {tot[2]:4d} | DF {tot[3]:3d} -> covered {tot[4]:3d}')
    say()
    say('TABLE 4.3b repaints of real later works, far from P (P99 in both observers, against all of P; none of these sources is a row of P). Against T the picture\'s own T row '
        '(same work id) is left out; pos works are not rows of T. Per group: far n, not far against T, median later-share, covered | the same for far-and-faithful (gap <= FID)')
    for lab in ('pos (1900-1969)', 'today, 1900-1969', 'today, 1970 on'):
        rp = REP[lab]
        z, s = rp['z'], rp['sel']
        assert not np.isin(rp['ids'], P['id']).any()
        emb = {o: np.asarray(z[f'emb_{o}'])[s] for o in OBS}
        sP = ev.score(P, {f'emb_{o}': emb[o] for o in OBS})
        far = ev.far(sP, P)
        gap = chord((unit(z['vec'][s]) * unit(z['emb_clip-vit-l-14'][s])).sum(1))
        nt, la = t_score(emb, rp['ids'])
        inT = int(np.isin(rp['ids'], T['id']).sum())
        out = []
        for sel in (far, far & (gap <= FID)):
            out.append(f'n {int(sel.sum()):4d}, not far against T {int((sel & nt).sum()):4d}, median later {np.median(la[sel]) if sel.any() else float("nan"):.3f}, covered {int((sel & nt & (la > 0.5)).sum()):4d}')
        say(f'   {lab:18s} (rows {int(s.sum())}, own row in T {inT}) far: {out[0]} || far-and-faithful: {out[1]}')
    say()

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['cell', 'arm', 'start', 'round', 'sigma_asked', 'centre_angle_from_start_deg', 'move_after_this_round_deg', 'move_after_previous_round_deg', 'n_infeasible',
                'median_far_dino', 'median_far_siglip', 'median_minrank'])
    for ia, (arm, tg) in enumerate(ARMS):
        for j, n in enumerate(STARTS):
            cell = f'qgc_{tg}_s{n}'
            z = R[cell]['z']
            sa = np.asarray(z['log_sigma_asked'], float)
            prev = LA[f'{cell}__prev']
            cen = np.vstack([prev, LA[f'{cell}__rea'][-1:]])          # centres of rounds 0..59 and the one after round 59
            uu = unit(cen)
            move = np.degrees(2 * np.arctan2(np.linalg.norm(uu[1:] - uu[:-1], axis=1), np.linalg.norm(uu[1:] + uu[:-1], axis=1)))   # move[r] = centre r -> centre r+1, made after round r
            drift = LB['drift_from_start'][ia, j]
            inf = LB[f'{cell}__infeasible_per_round']
            ser = A['series_median_per_round'][ia, j]
            assert len(sa) == 60 and prev.shape[0] == 60 and len(inf) == 60 and ser.shape == (60, 3)
            for rr in range(60):
                w.writerow([cell, arm, f's{n}', rr, f'{sa[rr]:.6g}', f'{drift[rr]:.6f}', f'{move[rr]:.6f}', '' if rr == 0 else f'{move[rr - 1]:.6f}', int(inf[rr]),
                            f'{ser[rr, 0]:.6f}', f'{ser[rr, 1]:.6f}', f'{ser[rr, 2]:.2f}'])
    say(f'rounds CSV: {rounds_csv} (900 rows = 15 cells x 60 rounds). Sources: sigma_asked = run file log_sigma_asked; centre angles from the pycma replay (lf_cma_v2); '
        'n_infeasible = run file feasible flag per round; medians of the far scores and the smaller rank per round = arms npz series_median_per_round. '
        'centre_angle_from_start_deg = angle of the centre round r\'s requests were drawn around from the start. move_after_this_round_deg = angle from round r\'s centre to the next one, '
        'i.e. the move made after round r (round 59: to the centre after the last round). move_after_previous_round_deg = the same for round r-1; blank at round 0. '
        'Angles by 2 arctan2(|a - b|, |a + b|).')
    open(rounds_csv, 'x').write(buf.getvalue())
    open(prefix + '.txt', 'x').write('\n'.join(OUT) + '\n')
    print('\n'.join(OUT), flush=True)


if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    except AssertionError as e:
        open(sys.argv[1] + '_STOP.txt', 'x').write('\n'.join(OUT) + f'\nSTOP: a check failed: {e!r}\n')
        raise
