"""r5_v4.py's sections 1 to 3 on the CLIP-scored search and its random-score loop (qgcclip_ra, qgcclip_pin, starts 6 to 10): the centre's path through the 32 CLIP regions, far-and-faithful share by region class, score by round.
Usage (cwd = the search folder): python -B r5b_v1.py OUTPREFIX -> OUTPREFIX.txt (refused if it exists)"""
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
R5, R1B, EV_FILE = f'{C_}/r5_v4.py', f'{C_}/r1b_v1.py', f'{C_}/controls_evaluate.py'
STARTS = (6, 7, 8, 9, 10)
ARMS = (('VC', 'ra', 'CLIP-scored search'), ('C1C', 'pin', 'its random-score loop'))
CLS = ('under 1%', '1-10%', '10-50%', '50% and over')
FH, OUT, CHK = [None], [], []


def say(s=''):
    OUT.append(str(s))
    print(s, flush=True)
    FH[0].write(str(s) + '\n')
    FH[0].flush()


def want(label, ok, got, exp):
    CHK.append(bool(ok))
    say(f'   GUARD {label}: got {got} | expected {exp} | {"same" if ok else "DIFFERENT"}')


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(prefix):
    t0 = time.time()
    for ext in ('.txt', '_STOP.txt'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    r5 = load_mod('r5', R5)
    r1b = load_mod('r1b', R1B)
    ev = load_mod('controls_evaluate', EV_FILE)
    say(f'numpy {np.__version__}')
    orig = ev.tag_role

    def tag_role(fl, d):
        t, role, n = orig(fl, d)
        mm = re.fullmatch(r'qgcclip_(ra|pin)_s([0-9]+)', t or '')
        return (t, ev.ROLE_OF[mm.group(1)], int(mm.group(2))) if mm else (t, role, n)
    ev.tag_role = tag_role   # the evaluator's tag rule knows only qgc_*; the CLIP-scored runs get the roles V and C1 by the same pattern
    unit = r5.unit
    af = np.load(r5.ALLFIT, allow_pickle=True)
    pw = np.load(r5.PERWORK, allow_pickle=True)
    vz = np.load(r5.VEC, allow_pickle=True)
    idk = [k for k in ('ids', 'id', 'work_ids') if k in vz.files][0]
    vid = np.asarray(vz[idk]).astype(np.int64)
    Vall = unit(vz['embeddings'])
    row = {int(i): j for j, i in enumerate(vid)}
    Qids, Qreg = np.asarray(af['Q_ids']).astype(np.int64), np.asarray(af['Q_region'])
    XQ = Vall[[row[int(i)] for i in Qids]]
    mean, cen_n = np.asarray(af['mean'], np.float64), unit(af['centres'])
    region = lambda X: np.argmax(unit(unit(X) - mean) @ cen_n.T, axis=1)  # noqa: E731
    want('G1 region rule reproduces Q_region on all works', np.array_equal(region(XQ), Qreg), f'{int(np.sum(region(XQ) == Qreg))}/{len(Qreg)}', len(Qreg))
    g = json.load(open(r5.GEOM))['mapA']['regions']
    grp, yr = np.asarray(pw['group_0A_1B_2later']), np.asarray(pw['year'], np.float64)
    got = [[int(np.sum((Qreg == k) & (grp == 0))), int(np.sum((Qreg == k) & (grp == 1))), int(np.sum((Qreg == k) & (grp == 2) & (yr < 1960))),
            int(np.sum((Qreg == k) & (grp == 2) & (yr >= 1960)))] for k in range(32)]
    want('G2 per-region counts = the map json', got == [[r['n_A'], r['n_B'], r['n_1900_1959'], r['n_1960_on']] for r in g], '32 regions', '32/32')
    share = np.array([r['pre1900_share_of_A_plus_later'] for r in g])
    cls = np.digitize(share, [0.01, 0.10, 0.50])
    Wold = XQ[yr < 1900]
    P, T, Pz = ev.references(O)
    ids, q = ev.startids(O), ev.qfun(O)
    res = {}
    for arm, tg, w in ARMS:
        for n in STARTS:
            tag = f'qgcclip_{tg}_s{n}'
            d = f'{O}/runs/{tag}_committee_euclidean_min_seedrow'
            r = ev.load(f'{tag}={d},{O}/eval/{tag}_emb.npz,{B}/logs/{tag}.out', P, T, ids, q, conf=True)
            z = r['z']
            rounds, picked, cen, cen_next, gd, Rq, Cr, it = r1b.replay(z, n)
            CHK.append(bool(gd[0] >= 0.98 and gd[1] >= 0.95))
            m = it >= 0
            ex = np.load(f'{O}/eval/{tag}_emb.npz', allow_pickle=True)
            assert [str(a) for a in ex['paths']] == [str(b) for b in z['paths']]
            FF = np.asarray(r['F'], bool) & np.asarray(r['ok_exact'], bool)
            fz = np.asarray(z['feasible']).astype(bool)
            zo, zc = np.asarray(z['z_agg'], np.float64), np.asarray(z['z_clip-vit-l-14'], np.float64)
            med = lambda v: np.array([np.nanmedian(v[rounds[t][fz[rounds[t]]]]) if fz[rounds[t]].any() else np.nan for t in range(60)])  # noqa: E731
            res[tag] = dict(creg=region(np.array(cen)), reg=region(np.asarray(ex['emb_clip-vit-l-14'], np.float64)[m]), it=it[m], FF=FF,
                            so=med(zo), sc=med(zc), loc=np.array([r5.local_share(cen[t], cen_next[t] - cen[t], Wold) for t in range(59)]))
            say(f'      {tag}: read; replay guard {"same" if CHK[-1] else "DIFFERENT"}; {time.time() - t0:.0f} s')
    if not all(CHK):
        say('STOP: a guard failed.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x', encoding='utf-8').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say(f'   GUARD PASSED ({len(CHK)} checks)')
    say()
    say("== 1 THE CENTRE'S PATH (region x rounds; class 0 under 1% ... 3 50% and over)")
    for arm, tg, w in ARMS:
        say(f'  {w}')
        ends = []
        for n in STARTS:
            c = res[f'qgcclip_{tg}_s{n}']['creg']
            vs = r5.visits(c.tolist())
            seen, ret = set(), 0
            for rg, _ in vs:
                ret += rg in seen
                seen.add(rg)
            first = next((t for t in range(60) if cls[c[t]] <= 1), None)
            ends.append(int(c[59]))
            say(f'     s{n}: {"-".join(f"{rg}x{ln}" for rg, ln in vs)} | distinct {len(seen)} | returns {ret} | first round under 10%: {first} | end region {c[59]} (class {cls[c[59]]})')
        say(f'     end regions s6..s10: {ends}')
    say()
    say("== 2 FAR AND FAITHFUL by the picture's region class: FF / pictures (share); and by the round's centre's class")
    for arm, tg, w in ARMS:
        tags = [f'qgcclip_{tg}_s{n}' for n in STARTS]
        pc = np.concatenate([cls[res[t]['reg']] for t in tags]); ccl = np.concatenate([cls[res[t]['creg'][res[t]['it']]] for t in tags])
        ff = np.concatenate([res[t]['FF'] for t in tags])
        say(f'  {w:24s} picture class: ' + ' | '.join(f'{CLS[i]} {int(ff[pc == i].sum())}/{int((pc == i).sum())}' + (f' ({ff[pc == i].mean():.2f})' if (pc == i).any() else '') for i in range(4)))
        say(f'  {"":24s} centre class:  ' + ' | '.join(f'{CLS[i]} {int(ff[ccl == i].sum())}/{int((ccl == i).sum())}' + (f' ({ff[ccl == i].mean():.2f})' if (ccl == i).any() else '') for i in range(4)))
    say()
    say("== 3 SCORE by round, median of the round's FEASIBLE pictures, pooled median over 5 starts, rounds 0 5 10 15 20 30 40 50 59")
    RS = (0, 5, 10, 15, 20, 30, 40, 50, 59)
    for arm, tg, w in ARMS:
        tags = [f'qgcclip_{tg}_s{n}' for n in STARTS]
        so = np.nanmedian([res[t]['so'] for t in tags], axis=0); sc = np.nanmedian([res[t]['sc'] for t in tags], axis=0)
        say(f"  {w:24s} observers' rescaled (smaller of 2): " + ' '.join(f'r{t}:{so[t]:+.2f}' for t in RS))
        say(f"  {w:24s} CLIP's own rescaled z:             " + ' '.join(f'r{t}:{sc[t]:+.2f}' for t in RS))
        L = np.concatenate([res[t]['loc'] for t in tags])
        say(f'  {w:24s} centre moves in the 39 local directions: median {np.median(L):.3f} [p10 {np.percentile(L, 10):.3f}, p90 {np.percentile(L, 90):.3f}]; random 0.051')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
