"""Where the three loops' centres climb, through the 32 CLIP regions of the region map (geom_v1.py): yardstick, centre path, far-and-faithful share by region class, score by round, off-manifold pictures, free search and walk.
Usage (cwd = the search folder): python -B r5_v4.py OUTPREFIX -> OUTPREFIX.txt + OUTPREFIX.npz (refused if either exists)"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
T1 = f'{B}/tests12/t1/run_rob_clip'
ALLFIT = f'{T1}/clip-vit-l-14_H2_allfit_K32.npz'
PERWORK = f'{T1}/clip-vit-l-14_H1_perwork.npz'
VEC = os.environ['DUROOM_EMB'] + '/clip-vit-l-14.npz'
GEOM = f'{B}/tests12/geom/geom_clip-vit-l-14_v1.json'
OO = f'{B}/tests12/onoff/out'
ONC = f'{OO}/onoff_v3_C.npz'
ONF = f'{OO}/onoff_ff_v1.npz'
EV_FILE = f'{C_}/controls_evaluate.py'
R1B = f'{C_}/r1b_v1.py'
STARTS = (6, 7, 8, 9, 10)
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
REACH, KLOC, NDIR = 0.61, 100, 39
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


def unit(X):
    X = np.asarray(X, np.float64)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def deg(a, b):
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


def visits(seq):
    """runs of equal consecutive values: list of (region, length)"""
    out = []
    for r in seq:
        if out and out[-1][0] == r:
            out[-1][1] += 1
        else:
            out.append([r, 1])
    return out


def runs_of(rounds):
    """sorted distinct rounds -> number of runs of consecutive rounds, longest run"""
    r = sorted(set(int(x) for x in rounds))
    if not r:
        return 0, 0
    n, best, cur = 1, 1, 1
    for a, b in zip(r, r[1:]):
        if b == a + 1:
            cur += 1
            best = max(best, cur)
        else:
            n, cur = n + 1, 1
    return n, best


def local_share(c, move, W):
    """share of |move|^2 in the top NDIR directions of the offsets (along the sphere) of the KLOC nearest works W (unit rows) to centre c"""
    s = W @ c
    nn = W[np.argpartition(-s, KLOC)[:KLOC]]
    cs = np.clip(nn @ c, -1, 1)
    tan = nn - cs[:, None] * c[None]
    off = tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12) * np.arccos(cs)[:, None]
    V = np.linalg.svd(off, full_matrices=False)[2][:NDIR]
    mv = move - (move @ c) * c
    return float(np.sum((V @ mv) ** 2) / (mv @ mv))


def main(prefix):
    t0 = time.time()
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists'
    FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    ev = load_mod('controls_evaluate', EV_FILE)
    r1b = load_mod('r1b', R1B)
    say(f'numpy {np.__version__}')
    # ---------------- regions
    af = np.load(ALLFIT, allow_pickle=True)
    pw = np.load(PERWORK, allow_pickle=True)
    vz = np.load(VEC, allow_pickle=True)
    idk = [k for k in ('ids', 'id', 'work_ids') if k in vz.files]
    assert idk and 'embeddings' in vz.files, vz.files
    vid = np.asarray(vz[idk[0]]).astype(np.int64)
    Vall = unit(vz['embeddings'])
    V32 = Vall.astype(np.float32)
    row = {int(i): j for j, i in enumerate(vid)}
    Qids, Qreg = np.asarray(af['Q_ids']).astype(np.int64), np.asarray(af['Q_region'])
    assert np.array_equal(Qids, np.asarray(pw['id']).astype(np.int64)), 'perwork rows are not Q_ids'
    XQ = Vall[[row[int(i)] for i in Qids]]
    mean, cen_n, cen_r = np.asarray(af['mean'], np.float64), unit(af['centres']), unit(af['raw_centres'])
    craw, cc0 = np.asarray(af['raw_centres'], np.float64), np.asarray(af['centres'], np.float64)
    rules = {'cosine, mean-removed centres': lambda X: np.argmax(unit(unit(X) - mean) @ cen_n.T, axis=1),
             'cosine, raw centres': lambda X: np.argmax(unit(X) @ cen_r.T, axis=1),
             'euclidean, raw centres': lambda X: np.argmin((craw ** 2).sum(1)[None] - 2 * unit(X) @ craw.T, axis=1),
             'euclidean, mean-removed centres': lambda X: np.argmin((cc0 ** 2).sum(1)[None] - 2 * (unit(X) - mean) @ cc0.T, axis=1)}
    match = {k: float(np.mean(f(XQ) == Qreg)) for k, f in rules.items()}
    best = max(match, key=match.get)
    exact_rules = [k for k, v in match.items() if v == 1.0]
    want(f'G1 a region rule reproduces Q_region (rules at 1.0: {exact_rules}; all rules {match}; norms of the mean-removed centres {np.linalg.norm(cc0, axis=1).min():.6f}-{np.linalg.norm(cc0, axis=1).max():.6f})',
         len(exact_rules) >= 1, len(exact_rules), '>= 1')
    DIS = [0, 0]

    def region(X):
        out = rules[exact_rules[0]](X) if exact_rules else rules[best](X)
        for k in exact_rules[1:]:
            DIS[0] += int(np.sum(rules[k](X) != out))
        DIS[1] += len(out)
        return out
    g = json.load(open(GEOM))['mapA']['regions']
    assert [r['id'] for r in g] == list(range(32))
    grp, yr = np.asarray(pw['group_0A_1B_2later']), np.asarray(pw['year'], np.float64)
    got = [[int(np.sum((Qreg == k) & (grp == 0))), int(np.sum((Qreg == k) & (grp == 1))), int(np.sum((Qreg == k) & (grp == 2) & (yr < 1960))),
            int(np.sum((Qreg == k) & (grp == 2) & (yr >= 1960)))] for k in range(32)]
    exp = [[r['n_A'], r['n_B'], r['n_1900_1959'], r['n_1960_on']] for r in g]
    want('G2 per-region counts n_A / n_B / n_1900_1959 / n_1960_on = the map json (32 regions)', got == exp, f'{sum(a == b for a, b in zip(got, exp))}/32 equal', '32/32')
    share = np.array([r['pre1900_share_of_A_plus_later'] for r in g])
    cls = np.digitize(share, [0.01, 0.10, 0.50])
    Wold = XQ[yr < 1900]
    say(f'   works before 1900 for the local directions: {len(Wold)}; corpus for reach: {len(Vall)}')
    # ---------------- 0 yardstick
    P, T, Pz = ev.references(O)
    Pread = unit(Pz['emb_clip-vit-l-14'])
    pid = np.asarray(P['id']).astype(np.int64)
    qrow = {int(i): j for j, i in enumerate(Qids)}
    have = np.array([int(i) in qrow for i in pid])
    oreg = np.array([Qreg[qrow[int(i)]] if int(i) in qrow else -1 for i in pid])
    preg = region(Pread)
    yard = (f'== 0 yardstick: P repaints with their original in the fit: {int(have.sum())} of {len(pid)}; reading in the original\'s region '
        f'{np.mean(preg[have] == oreg[have]):.3f}, in the original\'s class {np.mean(cls[preg[have]] == cls[oreg[have]]):.3f}; '
        f'reading classes {[int(np.sum(cls[preg] == i)) for i in range(4)]}')
    say('   (yardstick computed; printed after the guards)')
    # ---------------- runs
    ids, q = ev.startids(O), ev.qfun(O)
    onc, onf = np.load(ONC, allow_pickle=True), np.load(ONF, allow_pickle=True)
    res = {}
    for arm, tg, w in ARMS:
        for n in STARTS:
            tag = f'qgc_{tg}_s{n}'
            d = f'{O}/runs/{tag}_committee_euclidean_min_seedrow'
            r = ev.load(f'{tag}={d},{O}/eval/{tag}_emb.npz,{B}/logs/{tag}.out', P, T, ids, q, conf=True)
            z = r['z']
            rounds, picked, cen, cen_next, gd, Rq, Cr, it = r1b.replay(z, n)
            CHK.append(bool(gd[0] >= (0.99999 if tg == 'c2' else 0.98) and gd[1] >= 0.95))
            m = it >= 0
            ex = np.load(f'{O}/eval/{tag}_emb.npz', allow_pickle=True)
            assert [str(a) for a in ex['paths']] == [str(b) for b in z['paths']]
            FF = np.asarray(r['F'], bool) & np.asarray(r['ok_exact'], bool)
            assert [str(a) for a in onc[f'{tag}__paths']] == [str(b) for b in np.asarray(z['paths'])[m]], f'{tag}: onoff rows'
            OFF = np.asarray(onc[f'{tag}__FAR'], bool) & np.asarray(onf[f'{tag}__FF'], bool)
            assert np.array_equal(np.asarray(onf[f'{tag}__FF'], bool), FF), f'{tag}: onoff FF differs from the evaluator FF'
            reg_pic = region(np.asarray(ex['emb_clip-vit-l-14'], np.float64)[m])
            creg = region(np.array(cen))
            zag = np.asarray(z['z_agg'], np.float64)
            itm = it[m]
            sc_round = np.array([np.nanmedian(zag[rounds[t]]) for t in range(60)])
            sc_pick = np.array([np.nanmedian(zag[picked[t]]) for t in range(60)])
            fz = np.asarray(z['feasible']).astype(bool)
            sc_feas = np.array([np.nanmedian(zag[rounds[t][fz[rounds[t]]]]) if fz[rounds[t]].any() else np.nan for t in range(60)])
            loc = np.array([local_share(cen[t], cen_next[t] - cen[t], Wold) for t in range(59)])
            req = unit(z['style_embs'])
            res[tag] = dict(creg=creg, reg=reg_pic, it=itm, FF=FF, OFF=OFF, scr=sc_round, scp=sc_pick, scf=sc_feas, loc=loc,
                            ang_start=deg(req[m], req[0]), ang_cen=deg(req[m], np.array(cen)[itm]), rang_cen=deg(np.asarray(ex['emb_clip-vit-l-14'])[m], np.array(cen)[itm]),
                            read=unit(np.asarray(ex['emb_clip-vit-l-14'], np.float64)[m]))
            say(f'      {tag}: read; replay guard {"same" if CHK[-1] else "DIFFERENT"}; {time.time() - t0:.0f} s')
    want(f'G1b the exact rules agree on every vector assigned so far ({DIS[1]})', DIS[0] == 0, DIS[0], 0)
    if not all(CHK):
        say('STOP: a guard failed; no R5 number.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x', encoding='utf-8').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say(f'   GUARD PASSED ({len(CHK)} checks)')
    say(yard)
    for arm, tg, w in ARMS:
        say(f'   {arm}: FF / OFF per start: ' + ' | '.join(f"s{n} {int(res[f'qgc_{tg}_s{n}']['FF'].sum())}/{int(res[f'qgc_{tg}_s{n}']['OFF'].sum())}" for n in STARTS))
    sv = {}
    # ---------------- 1 path
    say()
    say("== 1 THE CENTRE'S PATH (region of each round's centre; class 0 under 1% ... 3 50% and over)")
    for arm, tg, w in ARMS:
        say(f'  {arm} {w}')
        ends = []
        for n in STARTS:
            c = res[f'qgc_{tg}_s{n}']['creg']
            vs = visits(c.tolist())
            seen, ret = set(), 0
            for rg, _ in vs:
                ret += rg in seen
                seen.add(rg)
            first = next((t for t in range(60) if cls[c[t]] <= 1), None)
            ends.append(int(c[59]))
            say(f'     s{n}: regions by round {"-".join(f"{rg}x{ln}" for rg, ln in vs)} | distinct {len(seen)} | visits {len(vs)}, mean length {np.mean([ln for _, ln in vs]):.1f} | '
                f'returns {ret} | first round in a region under 10%: {first} | classes start {cls[c[0]]} end {cls[c[59]]}')
            sv[f'{arm}_s{n}__creg'] = c
        say(f'     end regions s6..s10: {ends} ({len(set(ends))} distinct)')
    say('   same start, 3 loops, end region: ' + ' | '.join(f's{n}: ' + '/'.join(str(int(res[f"qgc_{tg}_s{n}"]["creg"][59])) for _, tg, _ in ARMS) for n in STARTS))
    # ---------------- 2 FF by class
    say()
    say("== 2 FAR AND FAITHFUL SHARE by the picture's region class and by its round's centre's class: FF / pictures (share)")
    for arm, tg, w in ARMS:
        tags = [f'qgc_{tg}_s{n}' for n in STARTS]
        pc = np.concatenate([cls[res[t]['reg']] for t in tags]); ccl = np.concatenate([cls[res[t]['creg'][res[t]['it']]] for t in tags])
        ff = np.concatenate([res[t]['FF'] for t in tags])
        say(f'  {arm} {w:18s} picture class: ' + ' | '.join(f'{CLS[i]} {int(ff[pc == i].sum())}/{int((pc == i).sum())} ({ff[pc == i].mean() if (pc == i).any() else float("nan"):.2f})' for i in range(4)))
        say(f'  {arm} {"":18s} centre class:  ' + ' | '.join(f'{CLS[i]} {int(ff[ccl == i].sum())}/{int((ccl == i).sum())} ({ff[ccl == i].mean() if (ccl == i).any() else float("nan"):.2f})' for i in range(4)))
        for n in STARTS:
            t = f'qgc_{tg}_s{n}'
            pcn = cls[res[t]['reg']]
            say(f'     s{n} picture class FF/pictures: ' + ' | '.join(f'{int(res[t]["FF"][pcn == i].sum())}/{int((pcn == i).sum())}' for i in range(4)))
    # ---------------- 3 score
    say()
    say('== 3 SCORE by round (median z_agg of the 32 | of the picked 4), pooled median over 5 starts, rounds 0 5 10 15 20 30 40 50 59')
    RS = (0, 5, 10, 15, 20, 30, 40, 50, 59)
    for arm, tg, w in ARMS:
        tags = [f'qgc_{tg}_s{n}' for n in STARTS]
        a = np.median([res[t]['scr'] for t in tags], axis=0); b = np.median([res[t]['scp'] for t in tags], axis=0)
        fz_ = np.nanmedian([res[t]['scf'] for t in tags], axis=0)
        say(f'  {arm} {w:18s} FEASIBLE ONLY (the told fitness): ' + ' '.join(f'r{t}:{fz_[t]:+.2f}' for t in RS))
        sv[f'{arm}__score_feasible'] = fz_
        say(f'  {arm} {w:18s} all 32 | picked 4: ' + ' '.join(f'r{t}:{a[t]:+.2f}|{b[t]:+.2f}' for t in RS) + f' ; picked minus round, mean over rounds {np.mean(b - a):+.3f}')
        ch, nc = [], []
        for t_ in tags:
            c, s = res[t_]['creg'], res[t_]['scr']
            for t in range(3, 56):
                (ch if c[t + 1] != c[t] else nc).append(np.mean(s[t + 1:t + 4]) - np.mean(s[t - 2:t + 1]))
        say(f'     around a centre region change (rounds t+1..t+3 minus t-2..t): changes {len(ch)} mean {np.mean(ch):+.3f} | no change {len(nc)} mean {np.mean(nc):+.3f}')
        L = np.concatenate([res[t]['loc'] for t in tags])
        say(f'     centre moves in the {NDIR} local directions of the {KLOC} nearest works before 1900: share of squared length median {np.median(L):.3f} '
            f'[p10 {np.percentile(L, 10):.3f}, p90 {np.percentile(L, 90):.3f}]; random {NDIR / 768:.3f}; rounds 0-9 {np.median(np.concatenate([res[t]["loc"][:10] for t in tags])):.3f}, '
            f'50-58 {np.median(np.concatenate([res[t]["loc"][50:] for t in tags])):.3f}')
        sv[f'{arm}__score_round'], sv[f'{arm}__score_picked'], sv[f'{arm}__local_share'] = a, b, L
    # ---------------- 4 OFF
    say()
    say('== 4 OFF THE MANIFOLD (far and faithful, nearest dated work beyond the line in both observers) against ON (far and faithful, the rest)')
    for arm, tg, w in ARMS:
        say(f'  {arm} {w}')
        for sel_name, pick in (('OFF', lambda R: R['OFF']), ('ON ', lambda R: R['FF'] & ~R['OFF'])):
            rows = []
            for n in STARTS:
                R = res[f'qgc_{tg}_s{n}']
                s = pick(R)
                if s.sum() == 0:
                    rows.append(f's{n}: 0')
                    continue
                rd = R['read'][s]
                reach = np.concatenate([(np.sqrt(np.clip(2 - 2 * (rd[i:i + 256].astype(np.float32) @ V32.T), 0, None)) <= REACH).sum(1) for i in range(0, len(rd), 256)])
                nr, lr = runs_of(R['it'][s])
                cc = cls[R['reg'][s]]
                rows.append(f's{n}: n {int(s.sum())}, rounds median {np.median(R["it"][s]):.0f} [{R["it"][s].min()}-{R["it"][s].max()}], {nr} runs of rounds (longest {lr}), '
                            f'classes {[int(np.sum(cc == i)) for i in range(4)]}, centre classes {[int(np.sum(cls[R["creg"][R["it"][s]]] == i)) for i in range(4)]}, '
                            f'request from start {np.median(R["ang_start"][s]):.1f} deg, request from centre {np.median(R["ang_cen"][s]):.1f}, reading from centre '
                            f'{np.median(R["rang_cen"][s]):.1f}, works within reach median {np.median(reach):.0f} [p10 {np.percentile(reach, 10):.0f}, p90 {np.percentile(reach, 90):.0f}]')
            say(f'   {sel_name} ' + '\n       '.join(rows))
    R7 = res['qgc_ra_s7']
    s = R7['OFF']
    ts = sorted(set(R7['it'][s].tolist()))
    say(f'  search s7 (Fra Angelico start): OFF pictures in {len(ts)} rounds; centre region / class in those rounds: '
        + ', '.join(f'r{t}:{R7["creg"][t]}/{cls[R7["creg"][t]]}' for t in ts[:40]) + (' ...' if len(ts) > 40 else ''))
    regs7 = R7['reg'][s]
    say(f'  search s7 OFF pictures by their own region: ' + ', '.join(f'{k}(class {cls[k]}, {g[k]["n_fit_A_plus_later"]} works, before 1900 {share[k]:.3f}): {int(np.sum(regs7 == k))}'
                                                            for k in sorted(set(regs7.tolist()), key=lambda k: -np.sum(regs7 == k))))
    # ---------------- 5 free search and walk
    say()
    say('== 5 FOR SCALE: free search (centre = unit mean of the round\'s requests; no restart) and the single walk (pictures only)')
    for n in (1, 2, 3):
        tag = f'main_s{n}'
        z = np.load(f'{O}/runs/{tag}_committee_euclidean_min_seedrow/all_candidates.npz', allow_pickle=True)
        ex = np.load(f'{O}/eval/{tag}_emb.npz', allow_pickle=True)
        assert [str(a) for a in ex['paths']] == [str(b) for b in z['paths']]
        it = np.asarray(z['iteration'])
        req = unit(z['style_embs'])
        assert all((it == t).any() for t in range(int(it.max()) + 1)), f'{tag}: an empty round'
        cm = np.array([unit(req[it == t].mean(0)) for t in range(int(it.max()) + 1)])
        cr = region(cm)
        first = next((t for t in range(len(cr)) if cls[cr[t]] <= 1), None)
        pc = cls[region(np.asarray(ex['emb_clip-vit-l-14'], np.float64)[it >= 0])]
        say(f'   {tag}: centre regions {"-".join(f"{rg}x{ln}" for rg, ln in visits(cr.tolist()))}; first round in a region under 10%: {first}; '
            f'pictures by class {[int(np.sum(pc == i)) for i in range(4)]}')
    for n in STARTS:
        tag = f'qgw_walk_s{n}'
        f = f'{O}/eval/{tag}_emb_h200_v1.npz'
        if not os.path.exists(f):
            say(f'   {tag}: no exact file {f}')
            continue
        ex = np.load(f, allow_pickle=True)
        assert len(ex['paths']) == 1921, f'{tag}: {len(ex["paths"])} rows'
        pc = cls[region(np.asarray(ex['emb_clip-vit-l-14'], np.float64)[1:])]
        say(f'   {tag}: pictures by class {[int(np.sum(pc == i)) for i in range(4)]} (rows after the start)')
    say(f'   rules at 1.0 disagree on {DIS[0]} of {DIS[1]} vectors assigned in all (must stay 0 after the free search and walk too)')
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'saved {prefix}.npz; done {time.time() - t0:.0f} s')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
