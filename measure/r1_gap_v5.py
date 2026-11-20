"""Gap between request and reading in the 15 main runs: alignment, persistence, direction, end angle, the picked four, the direct change per picture, and where the lean grows each round (A drawing, B gap, C picking, D averaging).
Usage: python -B r1_gap_v5.py OUTPREFIX -> OUTPREFIX.txt + OUTPREFIX.npz (refused if either exists)"""
import json
import os
import re
import sys
import time

import importlib.util

import numpy as np

B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
NEWAXIS = f'{B}/newaxis/k32/newaxis.json'
STARTS = (6, 7, 8, 9, 10)
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
NSIM, NNULL, SEED = 1000, 1000, 20261120
EV_FILE = f'{C_}/controls_evaluate.py'
FH, OUT, CHK = [None], [], []


def say(s=''):
    OUT.append(str(s))
    print(s, flush=True)
    if FH[0] is not None:
        FH[0].write(str(s) + '\n')
        FH[0].flush()


def unit(X):
    X = np.asarray(X, np.float64)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def tangent(c, r):
    """tangent part at r of (c - r), rows; c and r unit"""
    return c - (c * r).sum(-1, keepdims=True) * r


def deg(a, b):
    return float(np.degrees(np.arccos(np.clip(float(unit(a) @ unit(b)), -1, 1))))


def alignment(G):
    """G: (n, d) gaps. mean pairwise cosine of directions, and |mean g| / mean |g|"""
    U = unit(G)
    n = len(U)
    s = U.sum(0)
    pair = (float(s @ s) - n) / (n * (n - 1))
    return pair, float(np.linalg.norm(G.mean(0)) / np.linalg.norm(G, axis=1).mean())


def share(v, d):
    """share of |v|^2 along unit direction d"""
    v = np.asarray(v, np.float64)
    return float((v @ unit(d)) ** 2 / (v @ v))


def picked4(fit, feas, k=4):
    """the first k feasible in ascending told fitness (stable on ties: stored order, as Python's sorted)"""
    order = sorted(range(len(fit)), key=lambda i: fit[i])
    return [i for i in order if feas[i]][:k]


def walk_end(start, moves, rng):
    """a walk from start taking the given step angles (radians) in random tangent directions; returns the end angle (deg) from start"""
    x = unit(start)
    for a in moves:
        t = rng.standard_normal(len(x))
        t = unit(t - (t @ x) * x)
        x = np.cos(a) * x + np.sin(a) * t
    return deg(x, start)


def main(prefix):
    t0 = time.time()
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    say(f'numpy {np.__version__}')
    spec = importlib.util.spec_from_file_location('controls_evaluate', EV_FILE)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    qf = ev.qfun(O)
    DIRECT, STEPS = {}, {}
    dirs = {k: unit(v) for k, v in json.load(open(NEWAXIS))['cutoffs']['1900']['directions'].items()}
    dirs['mean_pre'] = unit(np.load(f'{O}/pools/starts_pre1900.npz', allow_pickle=True)['mean_pre'])
    DN = ('shift', 'axis1', 'axis2', 'mean_pre')
    rng_null = np.random.default_rng(SEED)
    NULL = unit(rng_null.standard_normal((NNULL, 768)))
    Pc = unit(np.load(f'{O}/push_emb.npz', allow_pickle=True)['emb_clip-vit-l-14'])
    assert Pc.shape == (2048, 768), Pc.shape
    ab = rng_null.integers(0, len(Pc), (2 * NNULL, 2))
    ab = ab[ab[:, 0] != ab[:, 1]][:NNULL]
    assert len(ab) == NNULL
    SPAN = unit(Pc[ab[:, 0]] - Pc[ab[:, 1]])
    rng_sim = np.random.default_rng(SEED + 1)
    res, sv, worst = {}, {}, {}
    for arm, tg, w in ARMS:
        for n in STARTS:
            tag = f'qgc_{tg}_s{n}'
            z = np.load(f'{O}/runs/{tag}_committee_euclidean_min_seedrow/all_candidates.npz', allow_pickle=True)
            fl = str(z['flags'])
            assert "'reanchor': 4" in fl and "'popsize': 32" in fl and f"'seed': {n}," in fl, f'{tag}: flags'
            rf = "'random_fitness': True" in fl
            it, R, Cr = np.asarray(z['iteration']), unit(z['style_embs']), unit(z['render_clip'])
            feas, nov = np.asarray(z['feasible']).astype(bool), np.asarray(z['novelties'], np.float64)
            rng_fit = np.random.default_rng(n + 7919)
            rounds, cen_next, picked, alts = [], [], [], []
            rng_alt = np.random.default_rng(SEED + n)
            for t in range(60):
                ix = np.flatnonzero(it == t)
                assert len(ix) == 32 and (np.diff(np.asarray(z['index'])[ix]) > 0).all(), f'{tag} round {t}'
                fit = list(rng_fit.standard_normal(32)) if rf else [(-nov[i] if feas[i] else np.inf) for i in ix]
                p = picked4(fit, feas[ix])
                assert len(p) >= 1, f'{tag} round {t}: no feasible candidate (the search keeps its CMA mean there; not rebuildable)'
                fe = [i for i in sorted(range(32), key=lambda i: fit[i]) if feas[ix][i]]
                fit_o = [(-nov[i] if feas[i] else np.inf) for i in ix] if rf else list(rng_alt.standard_normal(32))
                cand = ([fe[1:5], fe[:3] + [fe[4]], [fe[4]] + fe[1:4]] if len(fe) >= 5 else []) + [picked4(fit_o, feas[ix])]
                alts.append([unit(Cr[ix[a]].mean(0)) for a in cand if set(a) != set(p)])
                picked.append(ix[p])
                cen_next.append(unit(Cr[ix[p]].mean(0)))
                rounds.append(ix)
            # guard: rebuilt centre t+1 vs round t+1's requests
            cs, wins, nalt = [float(unit(R[rounds[0]].mean(0)) @ R[0])], [], 0   # round 0: drawn around the start request
            for t in range(59):
                rm = unit(R[rounds[t + 1]].mean(0))
                cs.append(float(cen_next[t] @ rm))
                if alts[t]:
                    nalt += len(alts[t])
                    wins.append(cs[-1] > max(float(a @ rm) for a in alts[t]))
            cs, wf = np.array(cs), float(np.mean(wins))
            worst[tag] = float(cs.min())
            CHK.append(bool(cs.min() >= (0.99999 if tg == 'c2' else 0.98) and wf >= 0.95))
            say(f'   GUARD {tag}: rebuilt centre vs round t+1 requests, min cos {cs.min():.6f} median {np.median(cs):.6f}; nearer than every alternative pick '
                f'in {100 * wf:.0f}% of {len(wins)} rounds ({nalt} alternatives) ({"same" if CHK[-1] else "DIFFERENT"})')
            # centres: round t's centre = rebuilt from round t-1 (round 0: the start request = row 0's request)
            c0 = R[0]
            cen = [c0] + cen_next[:59]
            G = [tangent(Cr[ix], R[ix]) for ix in rounds]
            al = np.array([alignment(g) for g in G])
            M = np.array([g.mean(0) for g in G])
            pers = np.array([float(unit(M[t]) @ unit(M[t + 1])) for t in range(59)])
            moves = np.radians([deg(cen[t], cen[t + 1]) for t in range(59)])
            mv = [tangent(cen[t + 1][None], cen[t][None])[0] for t in range(59)]
            mpers = np.array([float(unit(mv[t]) @ unit(mv[t + 1])) for t in range(58)])
            sh = {d: np.array([share(m, dirs[d]) for m in M]) for d in DN}
            Mu = unit(M)
            n_iso, n_span = (Mu @ NULL.T) ** 2, (Mu @ SPAN.T) ** 2
            pct = {d: float(np.mean(n_iso < sh[d][:, None])) for d in DN}
            pcs = {d: float(np.mean(n_span < sh[d][:, None])) for d in DN}
            sgn = {d: Mu @ dirs[d] for d in DN}
            spanmed = float(np.median(n_span))
            end_obs = deg(cen[-1], c0)
            sims = np.array([walk_end(c0, moves, rng_sim) for _ in range(NSIM)])
            gang = np.array([np.degrees(np.arccos(np.clip((Cr[p] * R[p]).sum(1), -1, 1))).mean() for p in picked])
            gcen = np.array([np.degrees(np.arccos(np.clip(Cr[picked[t]] @ cen[t], -1, 1))).mean() for t in range(60)])
            theta = np.radians(np.median(gcen))
            Mp = np.array([tangent(Cr[p], R[p]).mean(0) for p in picked])
            pvsall = np.array([float(unit(Mp[t]) @ unit(M[t])) for t in range(60)])
            shp = np.array([share(m, dirs['shift']) for m in Mp])
            res[tag] = dict(arm=arm, pair=al[:, 0], ratio=al[:, 1], pers=pers, moves=np.degrees(moves), mpers=mpers, sh=sh, pct=pct, end_obs=end_obs,
                            sims=sims, gang=gang, gcen=gcen, pcs=pcs, sgn=sgn, spanmed=spanmed, theta=np.degrees(theta), pred=np.degrees(np.arctan(np.tan(theta) / 2)), pvsall=pvsall, shp=shp, M=M)
            for k in ('pair', 'ratio', 'pers', 'moves', 'mpers', 'sims', 'gang', 'pvsall', 'shp'):
                sv[f'{tag}__{k}'] = np.asarray(res[tag][k])
            for d in DN:
                sv[f'{tag}__share_{d}'] = sh[d]
            ex = np.load(f'{O}/eval/{tag}_emb.npz', allow_pickle=True)
            assert [str(a) for a in z['paths']] == [str(b) for b in ex['paths']], f'{tag}: exact file rows'
            mm, late = it >= 0, it >= 50
            dd = {d: Cr @ dirs[d] - R @ dirs[d] for d in ('shift', 'mean_pre')}
            dq = qf(ex['emb_clip-vit-l-14']) - qf(z['style_embs'])
            lc = np.array([float(qf(cen[t][None])[0]) for t in range(60)] + [float(qf(cen_next[59][None])[0])])
            lr, lk = qf(R), qf(Cr)
            ST = np.array([[np.mean(lr[rounds[t]]) - lc[t], np.mean(lk[rounds[t]]) - np.mean(lr[rounds[t]]),
                            np.mean(lk[picked[t]]) - np.mean(lk[rounds[t]]), lc[t + 1] - np.mean(lk[picked[t]])] for t in range(59)])
            assert np.allclose(ST.sum(1), lc[1:60] - lc[:59], atol=1e-9), f'{tag}: telescoping'
            STEPS[tag] = (ST, lc)
            DIRECT[tag] = {k: (v[mm], v[late]) for k, v in (('shift', dd['shift']), ('mean_pre', dd['mean_pre']), ('lean', dq))}
            say(f'      {tag} done, {time.time() - t0:.0f} s')
    if not all(CHK):
        say('STOP: the picked-4 replay does not rebuild the centres; no number is computed.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x', encoding='utf-8').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say(f'   GUARD PASSED ({len(CHK)} runs).')
    med = lambda x: float(np.median(x))  # noqa: E731
    for arm, tg, w in ARMS:
        say()
        say(f'== {arm} {w} (per start 6..10; medians over rounds)')
        tags = [f'qgc_{tg}_s{n}' for n in STARTS]
        row = lambda f: ' | '.join(f(res[t]) for t in tags)  # noqa: E731
        say('  1 alignment: pairwise cos ' + row(lambda r: f'{med(r["pair"]):+.3f}') + ' ; |mean g|/mean|g| ' + row(lambda r: f'{med(r["ratio"]):.3f}') + ' (unrelated 0.177)')
        say('  2 persistence: consecutive mean gaps ' + row(lambda r: f'{med(r["pers"]):+.3f}') + ' ; centre moves consecutive ' + row(lambda r: f'{med(r["mpers"]):+.3f}'))
        for d in ('shift', 'axis1', 'axis2', 'mean_pre'):
            say(f'  3 share of mean gap |.|^2 along {d:8s} ' + row(lambda r, d=d: f'{med(r["sh"][d]):.4f} (pct iso {100 * r["pct"][d]:.0f}, span {100 * r["pcs"][d]:.0f}; signed cos {med(r["sgn"][d]):+.3f})')
                + ' (isotropic 0.0013; span median ' + row(lambda r: f'{r["spanmed"]:.4f}') + ')')
        say('  4 centre move per round (deg) ' + row(lambda r: f'{med(r["moves"]):.1f}') + ' ; end angle observed ' + row(lambda r: f'{r["end_obs"]:.1f}')
            + ' ; simulated median [p2.5, p97.5] ' + row(lambda r: f'{med(r["sims"]):.1f} [{np.percentile(r["sims"], 2.5):.1f}, {np.percentile(r["sims"], 97.5):.1f}]'))
        say('  5 picked 4: gap angle ' + row(lambda r: f'{med(r["gang"]):.1f}') + ' ; angle from centre ' + row(lambda r: f'{r["theta"]:.1f}') + ' ; atan(tan/2) ' + row(lambda r: f'{r["pred"]:.1f}') + ' ; observed move / that '
            + row(lambda r: f'{med(r["moves"]) / r["pred"]:.2f}') + ' ; cos(picked-4 mean gap, all-32) ' + row(lambda r: f'{med(r["pvsall"]):+.3f}')
            + ' ; shift share picked-4 vs all ' + row(lambda r: f'{med(r["shp"]):.4f} vs {med(r["sh"]["shift"]):.4f}'))
    say()
    say('== 6 DIRECT change, reading minus request, per picture (all 1,920 per start; rounds 50-59 after |): median [share > 0]')
    for arm, tg, w in ARMS:
        for k in ('mean_pre', 'shift', 'lean'):
            lab = {'mean_pre': 'c.d - r.d along the pre-1900 mean', 'shift': 'c.d - r.d along the shift', 'lean': 'lean(exact reading) - lean(request)'}[k]
            say(f'   {arm} {w:18s} {lab:36s} ' + ' | '.join(f'{np.median(DIRECT[t][k][0]):+.4f} [{np.mean(DIRECT[t][k][0] > 0):.2f}] / {np.median(DIRECT[t][k][1]):+.4f}'
                                                         for t in [f'qgc_{tg}_s{n}' for n in STARTS]))
            pool = np.concatenate([DIRECT[f'qgc_{tg}_s{n}'][k][0] for n in STARTS])
            say(f'      pooled 9,600: median {np.median(pool):+.4f}, share > 0 {np.mean(pool > 0):.3f}')
            sv[f'{arm}__direct_{k}'] = pool
    say()
    say("== 7 where the lean grows (loop's own reading; per round, mean over rounds 0-58; then the sum over the 59 rounds)")
    for arm, tg, w in ARMS:
        for n in STARTS:
            ST, lc = STEPS[f'qgc_{tg}_s{n}']
            m, sm = ST.mean(0), ST.sum(0)
            say(f'   {arm} s{n:<2d} lean of centre: start {lc[0]:+.3f} -> round 59 {lc[59]:+.3f} | per round: drawing {m[0]:+.4f} gap {m[1]:+.4f} picking {m[2]:+.4f} '
                f'averaging {m[3]:+.4f} | summed: drawing {sm[0]:+.3f} gap {sm[1]:+.3f} picking {sm[2]:+.3f} averaging {sm[3]:+.3f} (total {sm.sum():+.3f})')
        allst = np.concatenate([STEPS[f'qgc_{tg}_s{n}'][0] for n in STARTS])
        say(f'   {arm} pooled per round: drawing {allst[:, 0].mean():+.4f} gap {allst[:, 1].mean():+.4f} picking {allst[:, 2].mean():+.4f} averaging {allst[:, 3].mean():+.4f}'
            f'; rounds where averaging > 0: {np.mean(allst[:, 3] > 0):.2f}, drawing < 0: {np.mean(allst[:, 0] < 0):.2f}')
        sv[f'{arm}__steps'] = allst
    say()
    say('== 2b persistence between starts at the same round (mean over the 10 start pairs, median over rounds)')
    for arm, tg, w in ARMS:
        Ms = [unit(res[f'qgc_{tg}_s{n}']['M']) for n in STARTS]
        c = [np.mean([float(Ms[a][t] @ Ms[b][t]) for a in range(5) for b in range(a + 1, 5)]) for t in range(60)]
        say(f'   {arm} {w}: {np.median(c):+.3f} (rounds 0-9 {np.median(c[:10]):+.3f}, 50-59 {np.median(c[50:]):+.3f})')
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'saved {prefix}.npz; done {time.time() - t0:.0f} s')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
