"""Where the lean grows on the shared part, per round: all 32 pictures against the picked 4, the four-step split, far-and-faithful share by round, and Spearman of lean (exact reading) with the observers' score.
Usage (cwd = the search folder): python -B r1b_v1.py OUTPREFIX -> OUTPREFIX.txt + OUTPREFIX.npz (refused if either exists)"""
import importlib.util
import os
import sys
import time

import numpy as np

B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
R, L, E = f'{O}/runs', f'{B}/logs', f'{O}/eval'
EV_FILE = f'{C_}/controls_evaluate.py'
STARTS = (6, 7, 8, 9, 10)
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
SEED = 20261120
FH, OUT, CHK = [None], [], []


def say(s=''):
    OUT.append(str(s))
    print(s, flush=True)
    FH[0].write(str(s) + '\n')
    FH[0].flush()



def unit(X):
    X = np.asarray(X, np.float64)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def picked4(fit, feas, k=4):
    order = sorted(range(len(fit)), key=lambda i: fit[i])
    return [i for i in order if feas[i]][:k]


def ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind='stable')] = np.arange(len(x))
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(np.asarray(a, np.float64)), ranks(np.asarray(b, np.float64)))[0, 1])


def replay(z, n):
    """picks and centres as r1_gap_v5.py; returns rounds, picked, cen (60), cen_next (60), guard (min cos, win share)"""
    fl = str(z['flags'])
    assert "'reanchor': 4" in fl and "'popsize': 32" in fl and f"'seed': {n}," in fl
    rf = "'random_fitness': True" in fl
    it, Rq, Cr = np.asarray(z['iteration']), unit(z['style_embs']), unit(z['render_clip'])
    feas, nov = np.asarray(z['feasible']).astype(bool), np.asarray(z['novelties'], np.float64)
    rng_fit, rng_alt = np.random.default_rng(n + 7919), np.random.default_rng(SEED + n)
    rounds, picked, cen_next, alts = [], [], [], []
    for t in range(60):
        ix = np.flatnonzero(it == t)
        assert len(ix) == 32 and (np.diff(np.asarray(z['index'])[ix]) > 0).all()
        fit = list(rng_fit.standard_normal(32)) if rf else [(-nov[i] if feas[i] else np.inf) for i in ix]
        p = picked4(fit, feas[ix])
        assert len(p) >= 1
        fe = [i for i in sorted(range(32), key=lambda i: fit[i]) if feas[ix][i]]
        fit_o = [(-nov[i] if feas[i] else np.inf) for i in ix] if rf else list(rng_alt.standard_normal(32))
        cand = ([fe[1:5], fe[:3] + [fe[4]], [fe[4]] + fe[1:4]] if len(fe) >= 5 else []) + [picked4(fit_o, feas[ix])]
        alts.append([unit(Cr[ix[a]].mean(0)) for a in cand if set(a) != set(p)])
        picked.append(ix[p]); cen_next.append(unit(Cr[ix[p]].mean(0))); rounds.append(ix)
    cs, wins = [float(unit(Rq[rounds[0]].mean(0)) @ Rq[0])], []
    for t in range(59):
        rm = unit(Rq[rounds[t + 1]].mean(0))
        cs.append(float(cen_next[t] @ rm))
        if alts[t]:
            wins.append(cs[-1] > max(float(a @ rm) for a in alts[t]))
    return rounds, picked, [Rq[0]] + cen_next[:59], cen_next, (min(cs), float(np.mean(wins))), Rq, Cr, it


def main(prefix):
    t0 = time.time()
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    spec = importlib.util.spec_from_file_location('controls_evaluate', EV_FILE)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    say(f'numpy {np.__version__}')
    P, T, Pz = ev.references(O)
    ids, q = ev.startids(O), ev.qfun(O)
    res = {}
    for arm, tg, w in ARMS:
        for n in STARTS:
            tag = f'qgc_{tg}_s{n}'
            d = f'{R}/{tag}_committee_euclidean_min_seedrow'
            r = ev.load(f'{tag}={d},{E}/{tag}_emb.npz,{L}/{tag}.out', P, T, ids, q, conf=True)
            z = r['z']
            rounds, picked, cen, cen_next, g, Rq, Cr, it = replay(z, n)
            CHK.append(bool(g[0] >= (0.99999 if tg == 'c2' else 0.98) and g[1] >= 0.95))
            say(f'   GUARD {tag} replay: min cos {g[0]:.6f}, nearer than the alternatives in {100 * g[1]:.0f}% of rounds ({"same" if CHK[-1] else "DIFFERENT"})')
            lc = np.array([float(q(c[None])[0]) for c in cen] + [float(q(cen_next[59][None])[0])])
            lr, lk = q(Rq), q(Cr)
            ST = np.array([[np.mean(lr[rounds[t]]) - lc[t], np.mean(lk[rounds[t]]) - np.mean(lr[rounds[t]]),
                            np.mean(lk[picked[t]]) - np.mean(lk[rounds[t]]), lc[t + 1] - np.mean(lk[picked[t]])] for t in range(59)])
            all32 = np.array([float(q(unit(Cr[rounds[t]].mean(0))[None])[0]) - lc[t] for t in range(59)])
            p4 = np.array([float(q(unit(Cr[picked[t]].mean(0))[None])[0]) - lc[t] for t in range(59)])
            assert np.allclose(p4, lc[1:60] - lc[:59], atol=1e-9) and np.allclose(ST.sum(1), p4, atol=1e-9), f'{tag}: identities'
            m = it >= 0
            F, ok = np.asarray(r['F'], bool), np.asarray(r['ok_exact'], bool)
            DH = np.asarray(r['ev'], bool) & ok & np.asarray(r['C'], bool)
            itm = it[m]
            ff_round = np.array([np.mean((F & ok)[itm == t]) for t in range(60)])
            rho = spearman(np.asarray(r['q_exact']), np.asarray(r['m']))
            res[tag] = dict(ST=ST, all32=all32, p4=p4, FF=int((F & ok).sum()), DH=int(DH.sum()), ffr=ff_round, rho=rho, ql=np.asarray(r['q_exact']), sc=np.asarray(r['m']))
            say(f'      {tag} done {time.time() - t0:.0f} s')
    if not all(CHK):
        say('STOP: a guard did not reproduce; no new number.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x', encoding='utf-8').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say(f'   GUARD PASSED ({len(CHK)} checks)')
    sv = {}
    say()
    say("== SPLIT per round (loop's own reading; mean over 5 starts x 59 rounds; 6 decimals)")
    say('   loop          | drawing   | gap       | picking   | averaging | drawing+averaging | all 4 steps, one round')
    for arm, tg, w in ARMS:
        S = np.concatenate([res[f'qgc_{tg}_s{n}']['ST'] for n in STARTS])
        mm = S.mean(0)
        say(f'   {arm:3s} {w:18s} | {mm[0]:+.6f} | {mm[1]:+.6f} | {mm[2]:+.6f} | {mm[3]:+.6f} | {mm[0] + mm[3]:+.6f} | {S.sum(1).mean():+.6f}')
        sv[f'{arm}__split'] = S
    say()
    say('== 1 ALL 32 vs 2 PICKED 4, per round: lean(unit mean) - lean(centre); mean [share of rounds > 0]; by start s6..s10 then pooled')
    for arm, tg, w in ARMS:
        tags = [f'qgc_{tg}_s{n}' for n in STARTS]
        a = np.concatenate([res[t]['all32'] for t in tags]); p = np.concatenate([res[t]['p4'] for t in tags])
        S = np.concatenate([res[t]['ST'] for t in tags])
        say(f'   {arm} {w:18s} ALL 32: ' + ' | '.join(f'{res[t]["all32"].mean():+.4f} [{np.mean(res[t]["all32"] > 0):.2f}]' for t in tags) + f' || pooled {a.mean():+.6f} [{np.mean(a > 0):.2f}]')
        say(f'   {arm} {w:18s} PICKED 4: ' + ' | '.join(f'{res[t]["p4"].mean():+.4f} [{np.mean(res[t]["p4"] > 0):.2f}]' for t in tags) + f' || pooled {p.mean():+.6f} [{np.mean(p > 0):.2f}]')
        say(f'   {arm} {w:18s} drawing+averaging (one term): pooled {(S[:, 0] + S[:, 3]).mean():+.6f} [{np.mean(S[:, 0] + S[:, 3] > 0):.2f}]; '
            f'ALL 32 minus PICKED 4: {(a - p).mean():+.6f}')
        sv[f'{arm}__all32'], sv[f'{arm}__p4'] = a, p
    say()
    say('== 3 far-and-faithful share by round (pooled 5 starts x 32 = 160 a round)')
    for arm, tg, w in ARMS:
        fr = np.mean([res[f'qgc_{tg}_s{n}']['ffr'] for n in STARTS], axis=0)
        half = int(np.argmax(fr >= fr[50:].mean() / 2))
        say(f'   {arm} {w:18s} ' + ' '.join(f'r{t}:{fr[t]:.2f}' for t in (0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 59))
            + f' | rounds 0-9 {fr[:10].mean():.3f}, 50-59 {fr[50:].mean():.3f}; first round at half of the 50-59 level: {half}')
        sv[f'{arm}__ff_by_round'] = fr
    say()
    say("== 4 Spearman, per picture, lean (exact reading) vs the observers' score (smaller of the 2 over its bar)")
    for arm, tg, w in ARMS:
        tags = [f'qgc_{tg}_s{n}' for n in STARTS]
        pooled = spearman(np.concatenate([res[t]['ql'] for t in tags]), np.concatenate([res[t]['sc'] for t in tags]))
        say(f'   {arm} {w:18s} by start ' + ' | '.join(f'{res[t]["rho"]:+.3f}' for t in tags) + f' || pooled 9,600 {pooled:+.3f}')
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'saved {prefix}.npz; done {time.time() - t0:.0f} s')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
