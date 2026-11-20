"""Nearest-work chord with and without masking (A) and the dated split into PRE <1900 / POST / UNDATED pools of originals (B).
Usage (cwd = the search folder): python -B onoff_dated_v2.py OUTPREFIX -> OUTPREFIX.txt + OUTPREFIX.npz (refused if either exists)."""
import importlib.util
import os
import sys
import types

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']; CD = os.path.dirname(os.path.abspath(__file__))
O = f'{B}/sphere'
OO = f'{B}/tests12/onoff/out'
T3 = f'{B}/tests12/t3'
T3_FILE = f'{CD}/t3_measure_v5.py'
SCALES = f'{T3}/out/scales_v5.npz'
TR_FILE = f'{CD}/tr_readout_v2.py'
AB_NPZ = f'{OO}/onoff_v3_AB.npz'
CC_NPZ = f'{OO}/onoff_v3_C.npz'
FF_NPZ = f'{OO}/onoff_ff_v1.npz'
R2 = f'{B}/tests12/t2/renders_v1'
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
STARTS = (6, 7, 8, 9, 10)
POOLS = ('PRE', 'POST', 'UNDATED')
c_ = lambda x: int(np.sum(x))
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


def check(label, ok):
    CHK.append(bool(ok))
    say(f'   ASSERT {label}: {"same" if ok else "DIFFERS"}')


def main(prefix):
    assert 'qgt' not in prefix
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x')
    t3 = load_mod('t3_measure_v5', T3_FILE)
    tr = load_mod('tr_readout_v2', TR_FILE)
    A, Cz, Fz = (np.load(f, allow_pickle=True) for f in (AB_NPZ, CC_NPZ, FF_NPZ))
    say('onoff_dated_v2.py')
    ns = types.SimpleNamespace(corpus='real', root=O, rows=1921, rounds=60, per_round=32, scales=SCALES,
                               run_tmpl=f'{O}/runs/{{tag}}_committee_euclidean_min_seedrow/all_candidates.npz', exact_tmpl=f'{O}/eval/{{tag}}_emb.npz',
                               acc_prior=f'{R2}/prior256_v1', acc_noise=f'{R2}/noise160_v1', acc_aim_prior=None, acc_aim_noise=None)
    C, RP, idx = t3.load_common(ns)
    sc = t3.load_scales(ns)
    cal = sc['cal'][99]
    BAR = np.asarray(cal['theta'] * cal['med'], np.float64)
    OBS = t3.OBS
    pools = {'PRE': C['dated'] & (C['year'] < 1900), 'POST': C['dated'] & (C['year'] >= 1900), 'UNDATED': ~C['dated']}
    sz = {k: c_(v) for k, v in pools.items()}
    assert sum(sz.values()) == len(C['ids']) and not (pools['PRE'] & pools['POST']).any()
    say(f'bars D {BAR[0]:.6f} / S {BAR[1]:.6f}. Pools of originals (corpus year, dated = finite and > 0): PRE {sz["PRE"]}, POST {sz["POST"]}, UNDATED {sz["UNDATED"]}; total {len(C["ids"])}')

    cells = {}
    for arm, tg, _ in ARMS:
        for n in STARTS:
            tag = f'qgc_{tg}_s{n}'
            lc = t3.load_cell(tag, ns.run_tmpl, ns.exact_tmpl, 1921, C, layout=(60, 32), arm=arm, vtag=f'qgc_ra_s{n}')
            assert lc['valid'], lc['why']
            assert np.array_equal(np.array([str(p) for p in lc['paths'][1:]]), Cz[f'{tag}__paths']), tag
            cells[tag] = dict(E=lc['E'], src=lc['src'], e=np.asarray(Cz[f'{tag}__e'], np.float64), LL=Cz[f'{tag}__LL'], FF=Fz[f'{tag}__FF'],
                              kept=Cz[f'{tag}__kept'], hits=Cz[f'{tag}__hits'])
    known = {}
    for key, rk in (('P2', 'P2'), ('pos', 'pos'), ('today70', 'T')):
        s = RP[rk]
        pos_of = {p: i for i, p in enumerate(s['paths'])}
        ix = np.array([pos_of[str(p)] for p in A[f'{key}__paths']])
        known[key] = dict(E={o: s['E'][o][ix] for o in OBS}, src=s['src'][ix], e=np.asarray(A[f'{key}__e'], np.float64), LL=A[f'{key}__LL'], FF=Fz[f'{key}__FF'])
    acc = t3.load_acc(ns, need_aim=False)
    for key in ('noise', 'prior'):
        pos_of = {p: i for i, p in enumerate(acc['paths'])}
        ix = np.array([pos_of[str(p)] for p in A[f'{key}__paths']])
        known[key] = dict(E={o: acc['E'][o][ix] for o in OBS}, src=acc['src'][ix], e=np.asarray(A[f'{key}__e'], np.float64), LL=A[f'{key}__LL'], FF=Fz[f'{key}__FF'],
                          kind=A[f'{key}__kind'])
    allsrc = np.concatenate([g['src'] for g in list(cells.values()) + list(known.values())])
    dups = t3.dup_rows(C, allsrc[allsrc >= 0])
    full = t3.Ctx(C, idx, dups)

    def unmasked_min(E, src):
        u = np.full((len(src), 2), np.inf)
        for s in np.unique(src):
            if s < 0:
                continue
            rows, k = full.rows(int(s)), np.flatnonzero(src == s)
            for j, o in enumerate(OBS):
                u[k, j] = t3.chord((t3.unit(E[o][k]) @ C['E'][o][rows].T).max(1))
        return u

    def pool_chords(E, src, masked):
        masks = [full.rows(int(s)) for s in src] if masked else [np.zeros(0, int)] * len(src)
        out = {}
        for k in POOLS:
            d = t3.nearest(E, C['E'], masks, pools[k])
            out[k] = np.stack([d[o][:, 0] for o in OBS], 1)
        return out

    farb = lambda x: np.all(x > BAR, axis=1)
    nearb = lambda x: np.all(x <= BAR, axis=1)
    say()
    say('ASSERTS')
    for g in list(cells.values()) + list(known.values()):
        g['u'] = unmasked_min(g['E'], g['src'])
        g['dall'] = np.minimum(g['e'], g['u'])
        g['pool'] = pool_chords(g['E'], g['src'], False)
        g['pmin'] = np.minimum.reduce([g['pool'][k] for k in POOLS])
    for key in ('P2', 'pos', 'today70'):
        known[key]['pool_out'] = pool_chords(known[key]['E'], known[key]['src'], True)
    G = list(cells.values()) + list(known.values())
    # 1e-5: float32 dot products summed over different column subsets of the corpus agree only to about 1e-6
    check('d_all <= e everywhere', all((g['dall'] <= g['e']).all() for g in G))
    check('far in both unmasked implies far in both masked', all((~farb(g['dall']) | farb(g['e'])).all() for g in G))
    check('min over the three pools (unmasked) = d_all, to 1e-5', all(np.allclose(g['pmin'], g['dall'], rtol=0, atol=1e-5) for g in G))
    check('NONE (no pool within the bar in either reader) = unmasked far in both, picture by picture', all(np.array_equal(farb(g['pmin']), farb(g['dall'])) for g in G))
    check('repaints, source left out: min over the pools = the masked e (1e-5) and its NONE = masked far in both',
          all(np.allclose(np.minimum.reduce([known[k]['pool_out'][p] for p in POOLS]), known[k]['e'], rtol=0, atol=1e-5)
              and np.array_equal(farb(np.minimum.reduce([known[k]['pool_out'][p] for p in POOLS])), farb(known[k]['e'])) for k in ('P2', 'pos', 'today70')))
    say('   largest |min over pools - d_all|: ' + f'{max(float(np.abs(g["pmin"] - g["dall"]).max()) for g in G):.2e}')
    if not all(CHK):
        say('STOP: an assert failed; no table follows.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say('   ASSERTS PASSED.')

    def classes(P, strict):
        f = (lambda x: np.all(x <= BAR, axis=1)) if strict else (lambda x: np.any(x <= BAR, axis=1))
        pre = f(P['PRE'])
        since = ~pre & f(P['POST'])
        und = ~pre & ~since & f(P['UNDATED'])
        return pre, since, und, ~pre & ~since & ~und

    for g in G:
        g['cls'] = classes(g['pool'], False)
        g['cls2'] = classes(g['pool'], True)
    for k in ('P2', 'pos', 'today70'):
        known[k]['cls_out'] = classes(known[k]['pool_out'], False)
        known[k]['cls2_out'] = classes(known[k]['pool_out'], True)

    three = lambda x, s: f'{c_(s & nearb(x))}/{c_(s & ~nearb(x) & ~farb(x))}/{c_(s & farb(x))}'
    four = lambda cl, s: '/'.join(str(c_(s & c)) for c in cl)

    def row(label, g, sel, out=False):
        cl, cl2 = (g['cls_out'], g['cls2_out']) if out else (g['cls'], g['cls2'])
        say(f'   {label:46s} n {c_(sel):5d} | A masked {three(g["e"], sel)} | unmasked {three(g["dall"], sel)} || B one reader PRE/SINCE/UND/NONE {four(cl, sel)} '
            f'| B both readers PRE/SINCE/UND/rest {four(cl2, sel)}')

    def cat(tags):
        cs = [cells[t] for t in tags]
        g = {k: np.concatenate([c[k] for c in cs]) for k in ('e', 'dall', 'LL', 'FF', 'kept', 'hits')}
        g['cls'] = tuple(np.concatenate([c['cls'][i] for c in cs]) for i in range(4))
        g['cls2'] = tuple(np.concatenate([c['cls2'][i] for c in cs]) for i in range(4))
        return g

    say()
    say('A = near in both / near in one / far in both (masked = the E chords; unmasked = d_all). B = classes of part B (counts in the order named).')
    say('TABLE 1. Loops: FF candidates per start and pooled, FF x LL, FF x not LL, kept, hits (all kept and all hits are FF)')
    for arm, tg, w in ARMS:
        for n in STARTS + ('pooled',):
            g = cat([f'qgc_{tg}_s{k}' for k in (STARTS if n == 'pooled' else (n,))])
            lab = f'{arm} {"s" + str(n) if n != "pooled" else "pooled"}'
            row(f'{lab} FF', g, g['FF'])
            if n == 'pooled':
                row(f'{lab} FF and LL', g, g['FF'] & g['LL'])
                row(f'{lab} FF and not LL', g, g['FF'] & ~g['LL'])
                row(f'{lab} kept', g, g['kept'])
                row(f'{lab} hits', g, g['hits'])
    say()
    say('TABLE 2. Known answers (B unmasked: the source work counts in its own pool; repaints also with the source work and its near-duplicates left out)')
    for key, lab in (('P2', 'repaints before 1900, held out (P2)'), ('pos', 'repaints 1900 to 1969 (pos)'), ('today70', 'repaints 1970 on (T rows)')):
        g = known[key]
        al = np.ones(len(g['e']), bool)
        row(f'{lab}, all', g, al)
        row(f'{lab}, FF', g, g['FF'])
        row(f'{lab}, all, source left out', g, al, out=True)
        row(f'{lab}, FF, source left out', g, g['FF'], out=True)
    for key, lab in (('noise', 'noise requests'), ('prior', 'random requests')):
        g = known[key]
        row(f'{lab}, all', g, np.ones(len(g['e']), bool))
        row(f'{lab}, FF', g, g['FF'])
    for x in ('cone', 'sphere'):
        row(f'random requests, {x}', known['prior'], known['prior']['kind'] == x)
    say()
    say('RULE (tr_readout_v2.rule): "ahead" = more at all five starts and 4 x sum_search >= 5 x sum_control and, when the control\'s sum is 0, sum_search >= 10; '
        '"behind" the same way round; otherwise "cannot tell". Classes: part B, one-reader version, unmasked.')
    rules = {'FF and SINCE': lambda c: c['FF'] & c['cls'][1], 'FF, SINCE and LL': lambda c: c['FF'] & c['cls'][1] & c['LL'],
             'FF and NONE': lambda c: c['FF'] & c['cls'][3], 'FF, NONE and LL': lambda c: c['FF'] & c['cls'][3] & c['LL'],
             'FF and PRE-NEAR': lambda c: c['FF'] & c['cls'][0]}
    for lab, fn in rules.items():
        v = [c_(fn(cells[f'qgc_ra_s{n}'])) for n in STARTS]
        for x, tg in (('C1', 'pin'), ('C2', 'c2')):
            xx = [c_(fn(cells[f'qgc_{tg}_s{n}'])) for n in STARTS]
            say(f'   {lab}: search vs {x}: search {v} = {sum(v)}, {x} {xx} = {sum(xx)} -> {tr.rule(v, xx).upper()}')
    sv = {'bars': BAR, 'pool_sizes': np.array([sz[k] for k in POOLS])}
    for name, g in list(cells.items()) + list(known.items()):
        sv[f'{name}__u'], sv[f'{name}__dall'] = g['u'], g['dall']
        for k in POOLS:
            sv[f'{name}__pool_{k}'] = g['pool'][k]
        sv[f'{name}__class_one'] = np.select(list(g['cls']), [0, 1, 2, 3], -1)
        sv[f'{name}__class_both'] = np.select(list(g['cls2']), [0, 1, 2, 3], -1)
        if 'pool_out' in g:
            for k in POOLS:
                sv[f'{name}__poolout_{k}'] = g['pool_out'][k]
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'wrote {prefix}.txt and {prefix}.npz (per picture u, d_all, pool chords, classes 0 PRE 1 SINCE 2 UND 3 NONE/rest)')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
