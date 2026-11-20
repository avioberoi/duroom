"""How many leftover directions (K = 16 to 64) the rows FF and SINCE and LL need on the three loops: LL(K) per picture, counts per start with rule words, Spearman of the share against the far score and the logged raw distance, and the known-answer groups.
Usage (CPU, cwd = the search folder): python -B onoff_dirs_v1.py OUTPREFIX -> OUTPREFIX.txt and OUTPREFIX.npz, each refused if it exists."""
import importlib.util
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
OO = f'{B}/tests12/onoff/out'
LO = f'{B}/tests12/leftover'
R2 = f'{B}/tests12/t2/renders_v1'
EV_FILE = f'{C}/controls_evaluate.py'
LF_FILE = f'{C}/leftover_fn.py'
TR_FILE = f'{C}/tr_readout_v2.py'
BASIS = f'{LO}/out/pre_v1_basis.npz'
GRID = f'{B}/tests12/grid/grid_v1.npz'
AB_NPZ = f'{OO}/onoff_v3_AB.npz'
CC_NPZ = f'{OO}/onoff_v3_C.npz'
FF_NPZ = f'{OO}/onoff_ff_v1.npz'
DT_NPZ = f'{OO}/onoff_dated_v2.npz'
AR_NPZ = f'{OO}/onoff_artist_v1.npz'
DINO, SIG = 'dinov3-l', 'siglip-so400m'
RD = (DINO, SIG)
RN = {DINO: 'DINOv3', SIG: 'SigLIP'}
ARMS = (('V', 'ra'), ('C1', 'pin'), ('C2', 'c2'))
STARTS = (6, 7, 8, 9, 10)
KS = (16, 24, 32, 48, 64)
FOLDS, SEED = 5, 0
c_ = lambda x: int(np.sum(x))  # noqa: E731
FH, OUT, CHK = [None], [], []


def say(s=''):
    OUT.append(s)
    print(s, flush=True)
    FH[0].write(s + '\n')
    FH[0].flush()


def guard(p):
    assert 'qgt' not in str(p), f'refused path {p}'
    return p


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, guard(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(label, ok):
    CHK.append(bool(ok))
    say(f'   ASSERT {label}: {"same" if ok else "DIFFERS"}')


def avrank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    u, inv = np.unique(x, return_inverse=True)
    s = np.bincount(inv, weights=r)
    c = np.bincount(inv)
    return (s / c)[inv]


def spearman(a, b):
    ra, rb = avrank(a), avrank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main(prefix):
    guard(prefix)
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x')
    ev = load_mod('controls_evaluate', EV_FILE)
    lf = load_mod('leftover_fn', LF_FILE)
    tr = load_mod('tr_readout_v2', TR_FILE)
    for f in (BASIS, GRID, AB_NPZ, CC_NPZ, FF_NPZ, DT_NPZ, AR_NPZ):
        guard(f)
    BZ, G, A, Cz, Fz, Dz, Az = (np.load(f, allow_pickle=True) for f in (BASIS, GRID, AB_NPZ, CC_NPZ, FF_NPZ, DT_NPZ, AR_NPZ))
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1) < 1e-12 and abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    assert abs(spearman([1, 2, 2, 3], [1, 2, 3, 4]) - np.corrcoef([1, 2.5, 2.5, 4], [1, 2, 3, 4])[0, 1]) < 1e-12
    assert tr.rule([5] * 5, [4] * 5) == 'ahead' and tr.rule([2, 2, 2, 2, 1], [0] * 5) == 'cannot tell' and tr.rule([0] * 5, [2] * 5) == 'behind'
    say('onoff_dirs_v1.py. Counts only.')
    say(f'basis per K: leftover_fn.fit(all P rows, K); line: 99th percentile of leftover_fn.crossfit(P, artist, K, {FOLDS} folds, seed {SEED}); one fold draw. K = {list(KS)}.')

    P, T, Pz = ev.references(O)
    Part = np.array([str(a) for a in Pz['artist']])
    EP = {o: np.asarray(Pz[f'emb_{o}'], np.float64) for o in RD}
    fits, line = {}, {}
    for K in KS:
        for o in RD:
            fits[K, o] = lf.fit(EP[o], K)
            Lcf, _ = lf.crossfit(EP[o], Part, K, FOLDS, SEED)
            line[K, o] = lf.null(Lcf)[2]
        say(f'   K {K:2d}: line DINOv3 {line[K, DINO]:.6f}, SigLIP {line[K, SIG]:.6f}')
    saved = {o: (np.asarray(BZ[f'mean_{o}'], np.float64), np.asarray(BZ[f'V_{o}'], np.float64)) for o in RD}

    cells = {}
    for arm, tg in ARMS:
        for n in STARTS:
            tag = f'qgc_{tg}_s{n}'
            z = np.load(guard(f'{O}/runs/{tag}_committee_euclidean_min_seedrow/all_candidates.npz'), allow_pickle=True)
            m = np.asarray(z['iteration']) >= 0
            assert c_(m) == 1920, tag
            paths = np.array([str(p) for p in np.asarray(z['paths'])[m]])
            assert np.array_equal(paths, np.asarray(Cz[f'{tag}__paths']).astype(str)), tag
            c = dict(arm=arm, n=n, emb={o: np.asarray(z[f'emb_{o}'])[m] for o in RD},
                     raw={o: (np.asarray(z[f'raw_{o}'], np.float64)[m] if f'raw_{o}' in z.files else None) for o in RD},
                     FF=np.asarray(Fz[f'{tag}__FF'], bool), SINCE=np.asarray(Dz[f'{tag}__class_one']) == 1,
                     SINCEa=np.asarray(Az[f'{tag}__class_one_on']) == 1, CLEANa=np.asarray(Az[f'{tag}__clean_on'], bool), LL32=np.asarray(Cz[f'{tag}__LL'], bool))
            assert all(len(c[k]) == 1920 for k in ('FF', 'SINCE', 'SINCEa', 'CLEANa', 'LL32'))
            assert not (c['CLEANa'] & ~c['SINCEa']).any()
            cells[tag] = c
    rawok = all(c['raw'][o] is not None for c in cells.values() for o in RD)
    say(f'run files: 15 cells, 1,920 rows each (iteration >= 0), paths = onoff_v3 npz paths; raw_<reader> present in every run file: {rawok}')

    Tz = np.load(guard(f'{O}/push_today_emb.npz'), allow_pickle=True)
    i70 = np.flatnonzero(np.asarray(T['year']) >= 1970)
    assert len(i70) == 663
    assert np.array_equal(np.array([str(p) for p in Tz['paths']])[i70], np.asarray(A['today70__paths']).astype(str))
    em = np.load(guard(f'{R2}/noise160_v1/emb.npz'), allow_pickle=True)
    kat = [ln.strip() for ln in open(guard(f'{R2}/noise160_v1/inputs/expected_kat.txt')) if ln.strip()]
    assert [str(p) for p in em['paths'][:len(kat)]] == kat
    k0 = len(kat)
    assert np.array_equal(np.array([str(p) for p in em['paths'][k0:]]), np.asarray(A['noise__paths']).astype(str))
    BARD = np.asarray(Dz['bars'], np.float64)
    anyn = lambda x: np.any(np.asarray(x) <= BARD, axis=1)  # noqa: E731
    known = {'today70': dict(emb={o: np.asarray(Tz[f'emb_{o}'])[i70] for o in RD}), 'noise': dict(emb={o: np.asarray(em[f'emb_{o}'])[k0:] for o in RD})}
    for key, g in known.items():
        g['FF'] = np.asarray(Fz[f'{key}__FF'], bool)
        g['SINCE'] = np.asarray(Dz[f'{key}__class_one']) == 1
        g['LL32'] = np.asarray(A[f'{key}__LL'], bool)
        pre = anyn(Dz[f'{key}__pool_PRE'])
        g['since_re'] = ~pre & anyn(Dz[f'{key}__pool_POST'])
    pre_o = anyn(Dz['today70__poolout_PRE'])
    known['today70']['SINCE_out'] = ~pre_o & anyn(Dz['today70__poolout_POST'])

    sh, LL = {}, {}
    for K in KS:
        for tag, c in list(cells.items()) + list(known.items()):
            sh[K, tag] = {o: lf.share(c['emb'][o], *fits[K, o]) for o in RD}
            LL[K, tag] = (sh[K, tag][DINO] > line[K, DINO]) & (sh[K, tag][SIG] > line[K, SIG])

    COUNTS = (('FF, SINCE, LL', lambda c, L: c['FF'] & c['SINCE'] & L), ('FF, SINCE*, LL', lambda c, L: c['FF'] & c['SINCEa'] & L),
              ('FF, SINCE*-CLEAN, LL', lambda c, L: c['FF'] & c['CLEANa'] & L), ('FF, LL', lambda c, L: c['FF'] & L))
    per = {(K, lab, arm): [c_(fn(cells[f'qgc_{tg}_s{n}'], LL[K, f'qgc_{tg}_s{n}'])) for n in STARTS] for K in KS for lab, fn in COUNTS for arm, tg in ARMS}
    allc = {(K, arm): [c_(LL[K, f'qgc_{tg}_s{n}']) for n in STARTS] for K in KS for arm, tg in ARMS}

    say()
    say('ASSERTS (any failure stops before the tables)')
    dmax = max(float(np.abs(sh[32, t][o] - lf.share(cells[t]['emb'][o], *saved[o])).max()) for t in cells for o in RD)
    say(f'   K 32 refit vs saved basis: largest |share difference| over the 15 cells {dmax:.2e}; line refit - saved: '
        + ', '.join(f'{RN[o]} {line[32, o] - float(BZ[f"null_q99_{o}"]):+.2e}' for o in RD))
    check('K 32 line = saved basis null_q99 (1e-12), both readers', all(abs(line[32, o] - float(BZ[f'null_q99_{o}'])) < 1e-12 for o in RD))
    gap = min(float(np.abs(sh[32, t][o] - line[32, o]).min()) for t in cells for o in RD)
    say(f'   smallest |share - line| at K 32 over the 15 cells and both readers: {gap:.2e}')
    bad = {t: c_(LL[32, t] != cells[t]['LL32']) for t in cells}
    check(f'K 32 LL = onoff_v3 saved LL picture by picture in all 15 cells (pictures that differ per cell {list(bad.values())})', all(v == 0 for v in bad.values()))
    check('K 32 LL = onoff_v3 saved LL picture by picture, 1970-on repaints (663) and noise requests (160)',
          all(np.array_equal(LL[32, k], known[k]['LL32']) for k in known))
    check('SINCE recomputed from onoff_dated_v2 pool chords and bars = its class_one, both known groups', all(np.array_equal(g['since_re'], g['SINCE']) for g in known.values()))
    gname = [str(x) for x in G['group_names']]

    def grp(name):
        j = gname.index(name)
        pre = f'g{j:02d}__'
        return {k[len(pre):]: G[k] for k in G.files if k.startswith(pre)}
    for arm, tg in ARMS:
        g = grp(f'4 {arm}, all candidates (5 starts)')
        assert len(g['d_dino']) == 9600 and len(g['d_siglip']) == 9600
        for j, n in enumerate(STARTS):
            sl = slice(1920 * j, 1920 * (j + 1))
            cells[f'qgc_{tg}_s{n}']['far_d'] = {DINO: np.asarray(g['d_dino'][sl], float), SIG: np.asarray(g['d_siglip'][sl], float)}
    cat = lambda arm, f: np.concatenate([f(cells[f'qgc_{tg}_s{n}']) for a_, tg in ARMS if a_ == arm for n in STARTS])  # noqa: E731
    if not all(CHK):
        say('STOP: an assert failed; no table follows.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say('   ASSERTS PASSED.')

    say()
    say('DENOMINATORS (no K): per start s6..s10 = pooled')
    for lab, key in (('FF', None), ('FF, SINCE', 'SINCE'), ('FF, SINCE*', 'SINCEa'), ('FF, SINCE*-CLEAN', 'CLEANa')):
        say(f'   {lab:18s} ' + ' | '.join(f'{arm} ' + str(v) + f' = {sum(v)}' for arm, tg in ARMS
                                         for v in [[c_(cells[f"qgc_{tg}_s{n}"]["FF"] & (cells[f"qgc_{tg}_s{n}"][key] if key else True)) for n in STARTS]]))
    say()
    say('Per count and K: per start s6..s10 = pooled, per loop; rule (tr_readout_v2.rule) search vs each loop. Also LL among all candidates.')
    for lab, _ in COUNTS:
        say(f' {lab}:')
        for K in KS:
            v = per[K, lab, 'V']
            say(f'   K {K:2d} | ' + ' | '.join(f'{arm} {per[K, lab, arm]} = {sum(per[K, lab, arm])}' for arm, _ in ARMS)
                + ' | vs C1 ' + tr.rule(v, per[K, lab, 'C1']).upper() + ' | vs C2 ' + tr.rule(v, per[K, lab, 'C2']).upper())
    say(' LL among all candidates (no FF, no dating):')
    for K in KS:
        say(f'   K {K:2d} | ' + ' | '.join(f'{arm} {allc[K, arm]} = {sum(allc[K, arm])}' for arm, _ in ARMS)
            + ' | vs C1 ' + tr.rule(allc[K, 'V'], allc[K, 'C1']).upper() + ' | vs C2 ' + tr.rule(allc[K, 'V'], allc[K, 'C2']).upper())

    say()
    say('Spearman (average ranks) between the leftover share at K and a distance, SEARCH FF pictures pooled over s6..s10 (n '
        f'{sum(c_(cells[f"qgc_ra_s{n}"]["FF"]) for n in STARTS)}), per reader')
    say('   (a) grid_v1.npz group "4 V, all candidates (5 starts)" d_dino / d_siglip: mean chord to the 15 nearest P rows')
    say('   (b) run file all_candidates.npz raw_dinov3-l / raw_siglip-so400m (iteration >= 0): the per-reader k-NN mean distance the loop logged and turned into its told value')
    FFv = cat('V', lambda c: c['FF'])
    rho = {}
    for K in KS:
        parts = []
        for o in RD:
            s = np.concatenate([sh[K, f'qgc_ra_s{n}'][o] for n in STARTS])[FFv]
            ra = spearman(s, cat('V', lambda c: c['far_d'][o])[FFv])
            rb = spearman(s, cat('V', lambda c: c['raw'][o])[FFv]) if rawok else float('nan')
            rho[K, o] = (ra, rb)
            parts.append(f'{RN[o]} (a) {ra:+.4f} (b) {rb:+.4f}')
        say(f'   K {K:2d} | ' + ' | '.join(parts))

    say()
    say('Known-answer groups: FF and SINCE and LL(K) [and FF and LL(K)]. SINCE unmasked (the source work counts in its own pool, as onoff_dated_v2); '
        'for the 1970-on repaints also with the source and its near-duplicates left out (onoff_dated_v2 poolout chords, its bars).')
    t7, nz = known['today70'], known['noise']
    say(f'   n FF: 1970-on repaints {c_(t7["FF"])} (FF and SINCE {c_(t7["FF"] & t7["SINCE"])}, source out {c_(t7["FF"] & t7["SINCE_out"])}); '
        f'noise requests {c_(nz["FF"])} (FF and SINCE {c_(nz["FF"] & nz["SINCE"])})')
    for K in KS:
        say(f'   K {K:2d} | 1970-on FF,SINCE,LL {c_(t7["FF"] & t7["SINCE"] & LL[K, "today70"])}; source out {c_(t7["FF"] & t7["SINCE_out"] & LL[K, "today70"])}; '
            f'FF,LL {c_(t7["FF"] & LL[K, "today70"])} | noise FF,SINCE,LL {c_(nz["FF"] & nz["SINCE"] & LL[K, "noise"])}; FF,LL {c_(nz["FF"] & LL[K, "noise"])}')

    say()
    say('SUMMARY (pooled over s6..s10; rule words search vs C1 / vs C2)')
    for lab, _ in COUNTS:
        say(f' {lab}: ' + '; '.join(f'K {K} {sum(per[K, lab, "V"])} / {sum(per[K, lab, "C1"])} / {sum(per[K, lab, "C2"])} '
                                    f'{tr.rule(per[K, lab, "V"], per[K, lab, "C1"]).upper()} / {tr.rule(per[K, lab, "V"], per[K, lab, "C2"]).upper()}' for K in KS))
    for K in (16, 64):
        for lab, _ in COUNTS:
            say(f' per start K {K} {lab}: ' + ' | '.join(f'{arm} {per[K, lab, arm]}' for arm, _ in ARMS))
    say(' Spearman search FF (a) grid far score / (b) run-file raw: ' + '; '.join(
        f'K {K} D {rho[K, DINO][0]:+.3f}/{rho[K, DINO][1]:+.3f} S {rho[K, SIG][0]:+.3f}/{rho[K, SIG][1]:+.3f}' for K in KS))
    say('No loop was told the leftover at any K; these counts describe where the pictures fell.')

    sv = {'Ks': np.array(KS), 'lines': np.array([[line[K, o] for o in RD] for K in KS])}
    for K in KS:
        for t in list(cells) + list(known):
            sv[f'K{K}__{t}__LL'] = LL[K, t]
            for o in RD:
                sv[f'K{K}__{t}__share_{o}'] = sh[K, t][o]
    with open(prefix + '.npz', 'xb') as fh:
        np.savez_compressed(fh, **sv)
    say(f'wrote {prefix}.txt and {prefix}.npz (per K, per cell / known group: LL flags and shares, run-file row order = onoff_v3 npz order)')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
