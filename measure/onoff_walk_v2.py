"""Per-picture on/off chain (far, faithful, nearest-work classes, large leftover) on the 15 loop cells and the 5 walk cells, with the search-against-walk rule words.
Usage (CPU, cwd = the search folder): python -B onoff_walk_v2.py OUTPREFIX [--walk-exact rtx|h200] -> OUTPREFIX.txt and OUTPREFIX.npz, each refused if it exists."""
import argparse
import importlib.util
import os
import sys
import types

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
R, L, E = f'{O}/runs', f'{B}/logs', f'{O}/eval'
OO = f'{B}/tests12/onoff/out'
T3 = f'{B}/tests12/t3'
EV_FILE = f'{C_}/controls_evaluate.py'
T3_FILE = f'{C_}/t3_measure_v5.py'
TR_FILE = f'{C_}/tr_readout_v2.py'
LF_FILE = f'{C_}/leftover_fn.py'
WR_FILE = f'{C_}/walk_readout_v4.py'
SCALES = f'{T3}/out/scales_v5.npz'
BASIS = f'{B}/tests12/leftover/out/pre_v1_basis.npz'
CC_NPZ = f'{OO}/onoff_v3_C.npz'
FF_NPZ = f'{OO}/onoff_ff_v1.npz'
DT_NPZ = f'{OO}/onoff_dated_v2.npz'
RUN_TMPL = f'{R}/{{tag}}_committee_euclidean_min_seedrow/all_candidates.npz'
EX_W = {'rtx': f'{E}/{{tag}}_emb.npz', 'h200': f'{E}/{{tag}}_emb_h200_v1.npz'}
EX_TMPL = {'loop': f'{E}/{{tag}}_emb.npz', 'W': None}
LOOPS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
COLS = ('W', 'V', 'C1', 'C2')
STARTS = (6, 7, 8, 9, 10)
POOLS = ('PRE', 'POST', 'UNDATED')
DINO, SIG = 'dinov3-l', 'siglip-so400m'
RD = (DINO, SIG)
FN = {'F': lambda c: c['F'], 'okx': lambda c: c['okx'], 'okl': lambda c: c['okl'], 'FF': lambda c: c['FF'], 'FFl': lambda c: c['F'] & c['okl'],
      'NB': lambda c: c['FF'] & c['NEAR'], 'N1': lambda c: c['FF'] & ~c['FAR'] & ~c['NEAR'], 'N0': lambda c: c['FF'] & c['FAR'],
      'ND': lambda c: c['FF'] & c['nearD'], 'NS': lambda c: c['FF'] & c['nearS'],
      'PRE': lambda c: c['FF'] & c['cls1'][0], 'SINCE': lambda c: c['FF'] & c['cls1'][1], 'UND': lambda c: c['FF'] & c['cls1'][2], 'NONE': lambda c: c['FF'] & c['cls1'][3],
      'PRE2': lambda c: c['FF'] & c['cls2'][0], 'SINCE2': lambda c: c['FF'] & c['cls2'][1], 'UND2': lambda c: c['FF'] & c['cls2'][2], 'REST2': lambda c: c['FF'] & c['cls2'][3],
      'FFLL': lambda c: c['FF'] & c['LL'], 'LEAD': lambda c: c['FF'] & c['cls1'][1] & c['LL'], 'LEAD2': lambda c: c['FF'] & c['cls2'][1] & c['LL'],
      'OFFall': lambda c: c['FAR'], 'NONEall': lambda c: c['cls1'][3]}
TABLES = (('F', 'far from P alone (run-file readings), among all 1,920'), ('okx', 'exact-faithful (exact gap <= FID), among all 1,920'),
          ('FF', 'FF = far and exact-faithful (exact gap <= FID)'), ('FFl', 'FF by the loop gap (far and fidelity_clip <= 0.7788), beside'),
          ('NB', 'among FF: near in both readers (nearest original, start work and near-duplicates masked)'), ('N1', 'among FF: near in one reader only'),
          ('N0', 'among FF: near in neither (far in both, masked)'), ('ND', 'among FF: near in DINOv3'), ('NS', 'among FF: near in SigLIP'),
          ('PRE', 'among FF, one reader, nothing masked: PRE-NEAR (a work dated before 1900 near)'), ('SINCE', 'among FF, one reader: SINCE'),
          ('UND', 'among FF, one reader: UNDATED-ONLY'), ('NONE', 'among FF, one reader: NONE (off)'),
          ('PRE2', 'among FF, both readers: PRE'), ('SINCE2', 'among FF, both readers: SINCE'), ('UND2', 'among FF, both readers: UNDATED-ONLY'), ('REST2', 'among FF, both readers: rest'),
          ('FFLL', 'FF and LL (large leftover)'), ('LEAD', 'LEAD ROW: FF and SINCE (one reader) and LL'), ('LEAD2', 'both-readers lead row: FF and SINCE (both readers) and LL'),
          ('OFFall', 'OFF among all 1,920: far in both from every original, masked'),
          ('NONEall', 'NONE among all 1,920: one reader, nothing masked'))
RULES = (('a', 'FF (exact)', 'FF'), ('b', 'FF and SINCE', 'SINCE'), ('c', 'the lead row: FF and SINCE and LL', 'LEAD'), ('d', 'NONE (off) among FF', 'NONE'))
c_ = lambda x: int(np.sum(x))  # noqa: E731
FH, OUT, CHK = [None], [], []


def say(s=''):
    OUT.append(s)
    print(s, flush=True)
    FH[0].write(s + '\n')
    FH[0].flush()


def nq(*ss):
    for s in ss:
        assert 'qgt_' not in str(s), f'refused: a path or tag containing qgt_ ({s})'


def load_mod(name, path):
    nq(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stop_if_bad(prefix, msg='a saved flag did not reproduce; no walk file was opened'):
    if not all(CHK):
        say(f'STOP: {msg}.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x').write('\n'.join(OUT) + '\n')
        sys.exit(3)


def tag_of(arm, n):
    t = f'qgw_walk_s{n}' if arm == 'W' else f'qgc_{dict((a, g) for a, g, _ in LOOPS)[arm]}_s{n}'
    nq(t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--walk-exact', choices=('rtx', 'h200'), default='rtx')
    a = ap.parse_args()
    prefix = a.out
    EX_TMPL['W'] = EX_W[a.walk_exact]
    nq(prefix)
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x')
    ev = load_mod('controls_evaluate', EV_FILE)
    t3 = load_mod('t3_measure_v5', T3_FILE)
    tr = load_mod('tr_readout_v2', TR_FILE)
    lf = load_mod('leftover_fn', LF_FILE)
    assert (tr.rule([6] * 5, [5] * 5) == 'cannot tell' and tr.rule([20] * 5, [1] * 5) == 'ahead' and tr.rule([1] * 5, [20] * 5) == 'behind'
            and tr.rule([2] * 5, [0] * 5) == 'ahead' and tr.rule([2, 2, 2, 2, 1], [0] * 5) == 'cannot tell'), 'rule self-test'
    Cz, Fz, Dz = (np.load(f, allow_pickle=True) for f in (CC_NPZ, FF_NPZ, DT_NPZ))
    say(f'onoff_walk_v2.py; walk exact files: {a.walk_exact}, {EX_TMPL["W"]}. Counts only.')
    BZ = np.load(BASIS, allow_pickle=True)
    assert int(BZ['K']) == 32
    line = {o: float(BZ[f'null_q99_{o}']) for o in RD}
    fit = {o: (np.asarray(BZ[f'mean_{o}'], np.float64), np.asarray(BZ[f'V_{o}'], np.float64)) for o in RD}
    ns = types.SimpleNamespace(corpus='real', root=O, scales=SCALES)
    C, RP, idx = t3.load_common(ns)
    sc = t3.load_scales(ns)
    cal = sc['cal'][99]
    BAR = np.asarray(cal['theta'] * cal['med'], np.float64)
    assert np.array_equal(BAR, np.asarray(Dz['bars'], np.float64)), 'bars differ from onoff_dated_v2'
    OBS = t3.OBS
    assert tuple(OBS) == RD == tuple(ev.OBS)
    pools = {'PRE': C['dated'] & (C['year'] < 1900), 'POST': C['dated'] & (C['year'] >= 1900), 'UNDATED': ~C['dated']}
    assert [c_(pools[k]) for k in POOLS] == Dz['pool_sizes'].tolist()
    P, T, Pz = ev.references(O)
    ids, q = ev.startids(O), ev.qfun(O)
    dups = {}
    full = t3.Ctx(C, idx, dups)
    say(f'bars (nearest work) D {BAR[0]:.6f} / S {BAR[1]:.6f}; leftover line D {line[DINO]:.6f} / S {line[SIG]:.6f}; FID {float(ev.FID):.6f}; loop gap {ev.GATE_QG}; '
        f'pools PRE / POST / UNDATED {[c_(pools[k]) for k in POOLS]}')

    def classes(P, strict):
        f = (lambda x: np.all(x <= BAR, axis=1)) if strict else (lambda x: np.any(x <= BAR, axis=1))
        pre = f(P['PRE'])
        since = ~pre & f(P['POST'])
        und = ~pre & ~since & f(P['UNDATED'])
        return pre, since, und, ~pre & ~since & ~und

    def cell(arm, n, log, conf):
        """One code path for every cell: loop and walk cells differ in the tag, the exact file, the job log and conf."""
        tag = tag_of(arm, n)
        ex_tmpl = EX_TMPL['W' if arm == 'W' else 'loop']
        d, ex = os.path.dirname(RUN_TMPL.format(tag=tag)), ex_tmpl.format(tag=tag)
        nq(d, ex, log)
        r = ev.load(f'{tag}={d},{ex},{log}', P, T, ids, q, conf=conf)
        z = r['z']
        m = z['iteration'] >= 0
        lc = t3.load_cell(tag, RUN_TMPL, ex_tmpl, 1921, C, layout=(60, 32), arm=arm, vtag=f'qgc_ra_s{n}')
        assert lc['valid'], lc['why']
        paths = np.array([str(p) for p in lc['paths'][1:]])
        assert np.array_equal(paths, np.array([str(p) for p in np.asarray(z['paths'])[m]])), f'{tag}: run and exact rows differ'
        if lc['start'] not in dups:
            dups.update(t3.dup_rows(C, [lc['start']]))
        e = np.asarray(t3.readings(dict(E=lc['E'], src=np.asarray(lc['src'])), full)['e'], np.float64)
        emb = {o: np.asarray(z[f'emb_{o}'])[m] for o in RD}
        sh = {o: lf.share(np.asarray(emb[o]), *fit[o]) for o in RD}
        LL = (sh[DINO] > line[DINO]) & (sh[SIG] > line[SIG])
        masks = [np.zeros(0, int)] * len(lc['src'])
        pool = {}
        for k in POOLS:
            dd = t3.nearest(lc['E'], C['E'], masks, pools[k])
            pool[k] = np.stack([dd[o][:, 0] for o in OBS], 1)
        F, okx, okl = (np.asarray(r[k], bool) for k in ('F', 'ok_exact', 'ok_loop'))
        cls1, cls2 = classes(pool, False), classes(pool, True)
        return dict(tag=tag, F=F, okx=okx, okl=okl, FF=F & okx, e=e, FAR=np.all(e > BAR, axis=1), NEAR=np.all(e <= BAR, axis=1),
                    nearD=e[:, 0] <= BAR[0], nearS=e[:, 1] <= BAR[1], LL=LL, pool=pool, cls1=cls1, cls2=cls2,
                    class_one=np.select(list(cls1), [0, 1, 2, 3], -1), class_both=np.select(list(cls2), [0, 1, 2, 3], -1),
                    paths=paths, rc=r['rc'], start=int(lc['start']), wid=int(r['wid']), in_ref=r['in_ref'], gate=r['gate'])

    CL = {}
    have = lambda arm, n: tag_of(arm, n) in CL  # noqa: E731

    def col(arm, key):
        return [c_(FN[key](CL[tag_of(arm, n)])) if have(arm, n) else None for n in STARTS]

    say()
    say('KNOWN ANSWER: the 15 loop cells through cell(), picture by picture against the saved arrays (rows aligned by path)')
    for arm, tg, _ in LOOPS:
        for n in STARTS:
            t = tag_of(arm, n)
            c = cell(arm, n, f'{L}/{t}.out', True)
            CL[t] = c
            same = {'paths': np.array_equal(c['paths'], np.array([str(p) for p in Cz[f'{t}__paths']])),
                    'FF': np.array_equal(c['FF'], np.asarray(Fz[f'{t}__FF'], bool)),
                    'FAR': np.array_equal(c['FAR'], np.asarray(Cz[f'{t}__FAR'], bool)), 'NEAR': np.array_equal(c['NEAR'], np.asarray(Cz[f'{t}__NEAR'], bool)),
                    'LL': np.array_equal(c['LL'], np.asarray(Cz[f'{t}__LL'], bool)),
                    'class_one': np.array_equal(c['class_one'], np.asarray(Dz[f'{t}__class_one'])), 'class_both': np.array_equal(c['class_both'], np.asarray(Dz[f'{t}__class_both']))}
            de = float(np.abs(c['e'] - np.asarray(Cz[f'{t}__e'], np.float64)).max())
            dp = max(float(np.abs(c['pool'][k] - np.asarray(Dz[f'{t}__pool_{k}'], np.float64)).max()) for k in POOLS)
            CHK.append(all(same.values()) and de <= 1e-5 and dp <= 1e-5)
            say(f'   {t}: ' + ', '.join(f'{k} {"same" if v else "DIFFERS"}' for k, v in same.items()) + f' | max |e - saved| {de:.2e}, max |pool chord - saved| {dp:.2e} '
                f'(tolerance 1e-5) | smallest |e - bar| {float(np.abs(c["e"] - BAR).min()):.2e} | start row {c["start"]} work {c["wid"]}')
    stop_if_bad(prefix)
    say('   known answer passed.')
    say()

    wr = load_mod('walk_readout_v4', WR_FILE)
    sys.path.insert(0, C_)
    import local_rescore as lr
    obs_ids = np.load(wr.IDS, allow_pickle=True)['ids'].astype(np.int64)
    say('WALK CELLS (status read once, before the run file; read only if complete and the exact file exists)')
    for n in STARTS:
        t = tag_of('W', n)
        nq(wr.rundir(t), wr.logs_of(t))
        st = wr.status(t)
        ex = EX_TMPL['W'].format(tag=t)
        nq(ex)
        if st != 'complete' or not os.path.exists(ex):
            say(f'   {t}: skipped ({st}; exact file {"present" if os.path.exists(ex) else "missing"})')
            continue
        z = np.load(RUN_TMPL.format(tag=t), allow_pickle=True)
        bad, f = wr.validate('W', n, t, z, obs_ids, lr)
        if bad:
            CHK.append(False)
            say(f'   INVALID {t}: ' + '; '.join(bad))
            continue
        c = cell('W', n, '-', False)
        v = CL[tag_of('V', n)]
        extra = []
        if c['in_ref'] != (0, 0):
            extra.append(f'start work is a row of P or T {c["in_ref"]}')
        if c['gate'] != ev.GATE_QG:
            extra.append(f'fidelity gate {c["gate"]} is not {ev.GATE_QG}')
        if c['rc'] is None or c['rc'][0] < 0.999:
            extra.append(f'own CLIP reading vs exact: median cosine {None if c["rc"] is None else round(c["rc"][0], 4)} < 0.999')
        if c['start'] != v['start'] or c['wid'] != v['wid']:
            extra.append(f'start row / work {c["start"]} / {c["wid"]} differ from V s{n} {v["start"]} / {v["wid"]}')
        if extra:
            CHK.append(False)
            say(f'   INVALID {t}: ' + '; '.join(extra))
            continue
        CL[t] = c
        say(f'   {t}: card {f["card"]} | start idx {f["start_idx"]} work {f["wid"]} (= V s{n}) | start DINOv3 hash {f["hash"][:12]} | flags as the arm | stored bars = P\'s | '
            f'gate {c["gate"]} | start work in P, T: {c["in_ref"]}')
    stop_if_bad(prefix, 'a complete walk cell failed a validity check')
    got = [n for n in STARTS if have('W', n)]
    walk_note = f'read at starts {got}' if got else 'no walk cell complete'
    say(f'   walk cells read: {len(got)} of 5 ({walk_note})')
    say()

    say('READINGS per cell: own CLIP reading (run file render_clip) vs the exact one, cosine (controls_evaluate.load rc); faithful by the loop gap '
        '(fidelity_clip <= 0.7788) vs the exact gap (<= FID)')
    for t, c in CL.items():
        rc, okl, okx = c['rc'], c['okl'], c['okx']
        say(f'   {t:14s} cosine median {rc[0]:.4f} min {rc[1]:.4f} (under 0.999: {rc[2]}, under 0.9995: {rc[3]}) | faithful loop gap {c_(okl)} exact {c_(okx)} | '
            f'flag changes {c_(okl != okx)} (loop only {c_(okl & ~okx)}, exact only {c_(~okl & okx)}); among far {c_(c["F"] & (okl != okx))}')
    say()

    say(f'TABLES. Columns W = single walk ({walk_note}), V = search, C1 = random-score loop, C2 = no-step loop; 1,920 pictures per cell. '
        'A W pooled value marked * is over fewer than five starts.')
    for key, lab in TABLES:
        v = {x: col(x, key) for x in COLS}
        say(f' {lab}')
        say('   ' + f'{"":8s}' + ''.join(f'{x:>9s}' for x in COLS))
        for i, n in enumerate(STARTS):
            say('   ' + f'{"s" + str(n):8s}' + ''.join(f'{("-" if v[x][i] is None else v[x][i]):>9}' for x in COLS))
        pooled = {x: ('-' if all(u is None for u in v[x]) else f'{sum(u for u in v[x] if u is not None)}' + ('*' if any(u is None for u in v[x]) else '')) for x in COLS}
        say('   ' + f'{"pooled":8s}' + ''.join(f'{pooled[x]:>9}' for x in COLS))
    say()

    say('RULE (tr_readout_v2.rule): "ahead" = more at all five starts and 4 x sum_search >= 5 x sum_other, and sum_search >= 10 when the other\'s sum is 0; '
        '"behind" the same the other way round; otherwise "cannot tell" (not "equal"). Each word separately.')
    full5 = all(have('W', n) for n in STARTS)
    for k, lab, key in RULES:
        vv = col('V', key)
        if full5:
            ww = col('W', key)
            say(f'   ({k}) {lab}: search vs walk: search {vv} = {sum(vv)} | walk {ww} = {sum(ww)} -> search is "{tr.rule(vv, ww).upper()}"')
        else:
            say(f'   ({k}) {lab}: search vs walk: not computed (the walk is not read at all five starts: {walk_note})')
    say()

    sv = {'bars': BAR}
    for t, c in CL.items():
        for k in ('F', 'okx', 'okl', 'FF', 'e', 'FAR', 'NEAR', 'LL', 'class_one', 'class_both', 'paths'):
            sv[f'{t}__{k}'] = c[k]
        for k in POOLS:
            sv[f'{t}__pool_{k}'] = c['pool'][k]
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'wrote {prefix}.txt and {prefix}.npz (per cell: F, okx, okl, FF, e, FAR, NEAR, LL, class_one, class_both (0 PRE 1 SINCE 2 UND 3 NONE/rest), pool chords, paths)')
    FH[0].close()


if __name__ == '__main__':
    main()
