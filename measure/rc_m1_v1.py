"""Counts for the five restart-from-1 runs (qgm_m1_s6..s10) beside the search runs (qgc_ra_s6..s10): far and faithful, kept, hits, large leftover and dated classes (one reader, unmasked), by the same functions as the on/off chain.
Usage (CPU, cwd = the search folder): python -B rc_m1_v1.py OUTPREFIX -> OUTPREFIX.txt and OUTPREFIX.npz, each refused if it exists."""
import importlib.util
import os
import sys
import types

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
R, L, E = f'{O}/runs', f'{B}/logs', f'{O}/eval'
LO, T3D, OO = f'{B}/tests12/leftover', f'{B}/tests12/t3', f'{B}/tests12/onoff/out'
R2 = f'{B}/tests12/t2/renders_v1'
EV_FILE = f'{C_}/controls_evaluate.py'
LF_FILE = f'{C_}/leftover_fn.py'
T3_FILE = f'{C_}/t3_measure_v5.py'
TR_FILE = f'{C_}/tr_readout_v2.py'
BASIS = f'{LO}/out/pre_v1_basis.npz'
SCALES = f'{T3D}/out/scales_v5.npz'
CC_NPZ = f'{OO}/onoff_v3_C.npz'
FF_NPZ = f'{OO}/onoff_ff_v1.npz'
DT_NPZ = f'{OO}/onoff_dated_v2.npz'
STARTS = (6, 7, 8, 9, 10)
POOLS = ('PRE', 'POST', 'UNDATED')
c_ = lambda x: int(np.sum(x))  # noqa: E731
FH, OUT, CHK = [None], [], []


def say(s=''):
    OUT.append(s)
    print(s, flush=True)
    FH[0].write(s + '\n')
    FH[0].flush()


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def want(label, got, exp):
    CHK.append(got == exp)
    say(f'   GUARD {label}: got {got} | expected {exp} | {"same" if got == exp else "DIFFERENT"}')


def main(prefix):
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x')
    ev = load_mod('controls_evaluate', EV_FILE)
    lf = load_mod('leftover_fn', LF_FILE)
    t3 = load_mod('t3_measure_v5', T3_FILE)
    tr = load_mod('tr_readout_v2', TR_FILE)
    say('rc_m1_v1.py. Counts only.')
    BZ = np.load(BASIS, allow_pickle=True)
    assert int(BZ['K']) == 32
    RD = ('dinov3-l', 'siglip-so400m')
    line = {o: float(BZ[f'null_q99_{o}']) for o in RD}
    fit = {o: (np.asarray(BZ[f'mean_{o}'], np.float64), np.asarray(BZ[f'V_{o}'], np.float64)) for o in RD}
    say('leftover line (K 32, fold seed 0, 99th): ' + ', '.join(f'{o} {line[o]:.6f}' for o in RD))
    Cz, Fz, Dz = (np.load(f, allow_pickle=True) for f in (CC_NPZ, FF_NPZ, DT_NPZ))
    P, T, Pz = ev.references(O)
    ids, q = ev.startids(O), ev.qfun(O)
    ns = types.SimpleNamespace(corpus='real', root=O, rows=1921, rounds=60, per_round=32, scales=SCALES,
                               run_tmpl=f'{O}/runs/{{tag}}_committee_euclidean_min_seedrow/all_candidates.npz', exact_tmpl=f'{O}/eval/{{tag}}_emb.npz',
                               acc_prior=f'{R2}/prior256_v1', acc_noise=f'{R2}/noise160_v1', acc_aim_prior=None, acc_aim_noise=None)
    C, RP, idx = t3.load_common(ns)
    cal = t3.load_scales(ns)['cal'][99]
    BAR = np.asarray(cal['theta'] * cal['med'], np.float64)
    pools = {'PRE': C['dated'] & (C['year'] < 1900), 'POST': C['dated'] & (C['year'] >= 1900), 'UNDATED': ~C['dated']}
    say(f'manifold line (bars) D {BAR[0]:.6f} / S {BAR[1]:.6f}; pools PRE {c_(pools["PRE"])}, POST {c_(pools["POST"])}, UNDATED {c_(pools["UNDATED"])}')

    def cell(tag, guard):
        d = f'{R}/{tag}_committee_euclidean_min_seedrow'
        if guard:
            exact, log, conf, etmpl, arm = f'{E}/{tag}_emb.npz', f'{L}/{tag}.out', True, ns.exact_tmpl, 'V'
        else:
            exact, log, conf, etmpl, arm = f'{E}/{tag}_emb_h200_v1.npz', '-', False, f'{O}/eval/{{tag}}_emb_h200_v1.npz', 'V'
        r = ev.load(f'{tag}={d},{exact},{log}', P, T, ids, q, conf=conf)
        st = ev.stats(r, 'exact')
        z = r['z']
        m = z['iteration'] >= 0
        FF = np.asarray(r['F'], bool) & np.asarray(r['ok_exact'], bool)
        sh = {o: lf.share(np.asarray(z[f'emb_{o}'])[m], *fit[o]) for o in RD}
        LL = (sh[RD[0]] > line[RD[0]]) & (sh[RD[1]] > line[RD[1]])
        lc = t3.load_cell(tag, ns.run_tmpl, etmpl, 1921, C, layout=(60, 32), arm=arm, vtag=tag)   # arm 'V': the run's flags must show the search (no random score, no step path)
        assert lc['valid'], lc['why']
        paths = np.array([str(p) for p in np.asarray(z['paths'])[m]])
        assert np.array_equal(np.array([str(p) for p in lc['paths'][1:]]), paths), f'{tag}: run and exact rows differ'
        n = len(lc['src'])
        pc = {}
        for k in POOLS:
            dd = t3.nearest(lc['E'], C['E'], [np.zeros(0, int)] * n, pools[k])
            pc[k] = np.stack([dd[o][:, 0] for o in t3.OBS], 1)
        anyb = lambda x: np.any(x <= BAR, axis=1)  # noqa: E731
        pre = anyb(pc['PRE'])
        since = ~pre & anyb(pc['POST'])
        und = ~pre & ~since & anyb(pc['UNDATED'])
        cls = np.select([pre, since, und, ~pre & ~since & ~und], [0, 1, 2, 3], -1)
        fl = str(z['flags'])
        rf = [s for s in fl.split(', ') if any(k in s for k in ("'tag'", "'seed'", "'reanchor'", "'fidelity_max'", "'random_fitness'"))]
        say(f'   {tag}: flags {"; ".join(rf)} | start work {r["wid"]} | card {r["card"][0]} | loader {r["loader"]} | exact {os.path.basename(exact)} '
            f'| cos(loop CLIP, exact CLIP) median {r["rc"][0]:.4f} min {r["rc"][1]:.4f} | evaluator-admitted {int(r["ev"].sum())}, run own {int(z["admitted"][m].sum())}')
        return dict(FF=FF, LL=LL, cls=cls, kept=np.asarray(r['ev'], bool), hits=np.asarray(st['DH'], bool), paths=paths, sh=sh, nF=c_(r['F']), nfaith=c_(r['ok_exact']))

    say()
    say('GUARD: the search runs qgc_ra_s6..s10 through the same functions')
    G = {f'qgc_ra_s{n}': cell(f'qgc_ra_s{n}', True) for n in STARTS}
    for t, g in G.items():
        same = (np.array_equal(g['paths'], Cz[f'{t}__paths']) and np.array_equal(g['FF'], Fz[f'{t}__FF']) and np.array_equal(g['LL'], Cz[f'{t}__LL'])
                and np.array_equal(g['hits'], Cz[f'{t}__hits']) and np.array_equal(g['kept'], Cz[f'{t}__kept']) and np.array_equal(g['cls'], Dz[f'{t}__class_one']))
        want(f'{t} picture by picture: paths, FF, LL, hits, kept, class_one = saved arrays', same, True)
    cat = lambda gs, k: np.concatenate([g[k] for g in gs])  # noqa: E731
    four = lambda cl, s: tuple(c_(s & (cl == i)) for i in range(4))  # noqa: E731
    gl = list(G.values())
    if not all(CHK):
        say('STOP: the guard did not reproduce; no qgm number is computed.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say('   GUARD PASSED.')
    say()
    say('RESTART FROM 1 PICTURE: qgm_m1_s6..s10 (exact files <tag>_emb_h200_v1.npz, one run per start)')
    M = {f'qgm_m1_s{n}': cell(f'qgm_m1_s{n}', False) for n in STARTS}
    ml = list(M.values())

    def table(name, gs):
        say(f' {name}:')
        say('   per start (6, 7, 8, 9, 10) and sum')
        for lab, fn in (('far (both observers, P 99th)', lambda g: g['nF']), ('faithful (exact CLIP <= FID)', lambda g: g['nfaith']),
                        ('far and faithful FF', lambda g: c_(g['FF'])), ('kept (evaluator replay)', lambda g: c_(g['kept'])),
                        ('hits (distinct: admitted, far, faithful, covered)', lambda g: c_(g['hits'])), ('FF and LL', lambda g: c_(g['FF'] & g['LL'])),
                        ('lead: FF, SINCE (dated 1900 on near, none before), LL', lambda g: c_(g['FF'] & g['LL'] & (g['cls'] == 1))),
                        ('FF and SINCE (no leftover condition)', lambda g: c_(g['FF'] & (g['cls'] == 1)))):
            v = [fn(g) for g in gs]
            say(f'   {lab:58s} {v} = {sum(v)}')
        cl, ff, ll = cat(gs, 'cls'), cat(gs, 'FF'), cat(gs, 'LL')
        say('   dated split, one reader, unmasked, PRE-NEAR / SINCE / UNDATED-ONLY / NONE:')
        say(f'     among FF          {four(cl, ff)}  (n {c_(ff)})')
        say(f'     among FF and LL   {four(cl, ff & ll)}  (n {c_(ff & ll)})')
        say(f'     among FF, not LL  {four(cl, ff & ~ll)}  (n {c_(ff & ~ll)})')
        say(f'     among hits        {four(cl, cat(gs, "hits"))}  (n {c_(cat(gs, "hits"))})')
    table('the search, restart on 4 (qgc_ra)', gl)
    table('restart on 1 (qgm_m1)', ml)
    say()
    say('RULE FORM (tr_readout_v2.rule: more at all five starts and at least a quarter more in sum):')
    for lab, fn in (('FF', lambda g: c_(g['FF'])), ('hits', lambda g: c_(g['hits'])), ('lead', lambda g: c_(g['FF'] & g['LL'] & (g['cls'] == 1)))):
        a, b = [fn(g) for g in ml], [fn(g) for g in gl]
        say(f'   {lab}: restart on 1 {a} vs restart on 4 {b} -> restart on 1 is {tr.rule(a, b).upper()}')
    sv = {}
    for t, g in list(G.items()) + list(M.items()):
        for k in ('FF', 'LL', 'cls', 'kept', 'hits', 'paths'):
            sv[f'{t}__{k}'] = g[k]
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'wrote {prefix}.txt and {prefix}.npz')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
