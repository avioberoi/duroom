"""On/off classes with an artist era: an undated original takes the side of 1900 that all of its artist's dated originals share (PRE*, POST*, UNDATED* pools), else as onoff_dated_v2; counts on the main runs and the known-answer sets.
Usage (CPU, cwd = the search folder): python -B onoff_artist_v1.py OUTPREFIX -> OUTPREFIX.txt and OUTPREFIX.npz, each refused if it exists."""
import importlib.util
import os
import sys
import types

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
O = f'{B}/sphere'
OO = f'{B}/tests12/onoff/out'
T3 = f'{B}/tests12/t3'
T3_FILE = os.path.dirname(os.path.abspath(__file__)) + '/t3_measure_v5.py'
SCALES = f'{T3}/out/scales_v5.npz'
TR_FILE = os.path.dirname(os.path.abspath(__file__)) + '/tr_readout_v2.py'
AB_NPZ = f'{OO}/onoff_v3_AB.npz'
CC_NPZ = f'{OO}/onoff_v3_C.npz'
FF_NPZ = f'{OO}/onoff_ff_v1.npz'
DT_NPZ = f'{OO}/onoff_dated_v2.npz'
R2 = f'{B}/tests12/t2/renders_v1'
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
STARTS = (6, 7, 8, 9, 10)
POOLS = ('PRE', 'POST', 'UNDATED')
CASES = ('PRE-BY-ARTIST', 'POST-BY-ARTIST', 'straddles 1900', 'artist has no dated work', 'label empty')
c_ = lambda x: int(np.sum(x))  # noqa: E731
FH, OUT, CHK = [None], [], []


def say(s=''):
    OUT.append(s)
    print(s, flush=True)
    if FH[0] is not None:
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


def artist_counts(artist, year, dated):
    """per row: non-empty label?, number of dated originals of the row's artist before 1900 / from 1900 on (the row itself included if dated)"""
    lab = np.asarray(artist).astype(str)
    pre_r = dated & (np.where(dated, year, np.inf) < 1900)
    post_r = dated & (np.where(dated, year, -np.inf) >= 1900)
    _, inv = np.unique(lab, return_inverse=True)
    inv = inv.ravel()
    npre = np.bincount(inv, weights=pre_r.astype(float)).astype(int)[inv]
    npost = np.bincount(inv, weights=post_r.astype(float)).astype(int)[inv]
    return lab != '', npre, npost, pre_r, post_r


def decide(nonempty, npre, npost):
    """0 PRE-BY-ARTIST, 1 POST-BY-ARTIST, 2 straddles 1900, 3 artist has no dated work, 4 label empty"""
    out = np.full(len(npre), -1, np.int8)
    out[~nonempty] = 4
    out[nonempty & (npre > 0) & (npost == 0)] = 0
    out[nonempty & (npost > 0) & (npre == 0)] = 1
    out[nonempty & (npre > 0) & (npost > 0)] = 2
    out[nonempty & (npre == 0) & (npost == 0)] = 3
    assert (out >= 0).all()
    return out


def era(artist, year, dated):
    """undated rows: their case (decide on the artist's dated works); dated rows: the same with the row's own year hidden (leave one out). Also the counts."""
    ne, npre, npost, pre_r, post_r = artist_counts(artist, year, dated)
    und = decide(ne, npre, npost)
    loo = decide(ne, npre - pre_r.astype(int), npost - post_r.astype(int))
    return np.where(dated, -1, und).astype(np.int8), np.where(dated, loo, -1).astype(np.int8), npre, npost, pre_r, post_r


def fixture():
    a = np.array(['a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', '', '', 'f', 'f'])
    y = np.array([1850, 1870, np.nan, 1950, np.nan, 1850, 1950, 0, np.nan, np.nan, 1850, np.nan, 1900, np.nan], float)
    d = np.isfinite(y) & (y > 0)
    u, lo, *_ = era(a, y, d)
    want_u = [-1, -1, 0, -1, 1, -1, -1, 2, 3, 3, -1, 4, -1, 1]
    want_lo = [0, 0, -1, 3, -1, 1, 0, -1, -1, -1, 4, -1, 3, -1]
    return u.tolist() == want_u and lo.tolist() == want_lo


def main(prefix):
    assert 'qgt' not in prefix
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x')
    t3 = load_mod('t3_measure_v5', T3_FILE)
    tr = load_mod('tr_readout_v2', TR_FILE)
    A, Cz, Fz, Dz = (np.load(f, allow_pickle=True) for f in (AB_NPZ, CC_NPZ, FF_NPZ, DT_NPZ))
    say('onoff_artist_v1.py. Counts only.')
    ns = types.SimpleNamespace(corpus='real', root=O, rows=1921, rounds=60, per_round=32, scales=SCALES,
                               run_tmpl=f'{O}/runs/{{tag}}_committee_euclidean_min_seedrow/all_candidates.npz', exact_tmpl=f'{O}/eval/{{tag}}_emb.npz',
                               acc_prior=f'{R2}/prior256_v1', acc_noise=f'{R2}/noise160_v1', acc_aim_prior=None, acc_aim_noise=None)
    C, RP, idx = t3.load_common(ns)
    sc = t3.load_scales(ns)
    cal = sc['cal'][99]
    BAR = np.asarray(cal['theta'] * cal['med'], np.float64)
    OBS = t3.OBS
    N = len(C['ids'])
    dated, year, artist = C['dated'], C['year'], np.asarray(C['artist']).astype(str)

    say()
    say('ASSERTS')
    check('artist-era rule on the hand-made fixture (all-pre, all-post incl. 1900, straddle, no dated work, empty label, single dated work, leave-one-out)', fixture())
    check('non-empty labels = the keys of t3_measure_v5 by_artist', set(np.unique(artist[artist != ''])) == set(C['by_artist'].keys()))
    ucode, loo, npre, npost, pre_r, post_r = era(artist, year, dated)
    off = {'PRE': dated & (year < 1900), 'POST': dated & (year >= 1900), 'UNDATED': ~dated}
    check('rule-off pools = the sizes of the saved onoff_dated_v2 arrays and = era() sides', tuple(c_(off[k]) for k in POOLS) == tuple(Dz['pool_sizes'])
          and np.array_equal(off['PRE'], pre_r) and np.array_equal(off['POST'], post_r))
    pba, poba = ~dated & (ucode == 0), ~dated & (ucode == 1)
    on = {'PRE': off['PRE'] | pba, 'POST': off['POST'] | poba, 'UNDATED': ~dated & ~pba & ~poba}
    szon = tuple(c_(on[k]) for k in POOLS)
    check('new pools disjoint and cover the corpus', sum(szon) == N and not (on['PRE'] & on['POST']).any() and not (on['PRE'] & on['UNDATED']).any()
          and not (on['POST'] & on['UNDATED']).any())
    check('cases cover the undated originals exactly once', all(ucode[~dated] >= 0) and (ucode[dated] == -1).all()
          and sum(c_(~dated & (ucode == i)) for i in range(5)) == c_(~dated))

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
    OUTK = ('P2', 'pos', 'today70')

    def pool_chords(E, src, masked, pools):
        masks = [full.rows(int(s)) for s in src] if masked else [np.zeros(0, int)] * len(src)
        out = {}
        for k in POOLS:
            d = t3.nearest(E, C['E'], masks, pools[k])
            out[k] = np.stack([d[o][:, 0] for o in OBS], 1)
        return out

    anyn = lambda x: np.any(x <= BAR, axis=1)  # noqa: E731
    farb = lambda x: np.all(x > BAR, axis=1)  # noqa: E731

    def classes(P, strict):
        f = (lambda x: np.all(x <= BAR, axis=1)) if strict else (lambda x: np.any(x <= BAR, axis=1))
        pre = f(P['PRE'])
        since = ~pre & f(P['POST'])
        und = ~pre & ~since & f(P['UNDATED'])
        return pre, since, und, ~pre & ~since & ~und

    def derive(P):
        one = classes(P, False)
        return dict(one=one, both=classes(P, True), clean=one[1] & ~anyn(P['UNDATED']), undnear=one[1] & anyn(P['UNDATED']))

    G = dict(list(cells.items()) + list(known.items()))
    for name, g in G.items():
        for r, pools in (('off', off), ('on', on)):
            g[f'pool_{r}'] = pool_chords(g['E'], g['src'], False, pools)
            g[r] = derive(g[f'pool_{r}'])
            if name in OUTK:
                g[f'poolout_{r}'] = pool_chords(g['E'], g['src'], True, pools)
                g[f'{r}_out'] = derive(g[f'poolout_{r}'])

    gap = max(float(np.abs(g['pool_off'][k] - np.asarray(Dz[f'{nm}__pool_{k}'])).max()) for nm, g in G.items() for k in POOLS)
    check('rule off: pool chords = onoff_dated_v2 npz, every group and pool, to 1e-5', gap <= 1e-5)
    gapo = max(float(np.abs(G[nm]['poolout_off'][k] - np.asarray(Dz[f'{nm}__poolout_{k}'])).max()) for nm in OUTK for k in POOLS)
    check('rule off: source-left-out pool chords = onoff_dated_v2 npz, to 1e-5', gapo <= 1e-5)
    say(f'   largest gap to the v2 npz chords: {max(gap, gapo):.2e}')
    sel1 = lambda d: np.select(list(d['one']), [0, 1, 2, 3], -1)  # noqa: E731
    sel2 = lambda d: np.select(list(d['both']), [0, 1, 2, 3], -1)  # noqa: E731
    bad = {nm: (c_(sel1(g['off']) != Dz[f'{nm}__class_one']), c_(sel2(g['off']) != Dz[f'{nm}__class_both'])) for nm, g in G.items()}
    check('rule off: one-reader and both-reader classes = onoff_dated_v2 npz, picture by picture', all(v == (0, 0) for v in bad.values()))
    if not all(v == (0, 0) for v in bad.values()):
        for nm, v in bad.items():
            if v != (0, 0):
                g = G[nm]
                dif = (sel1(g['off']) != Dz[f'{nm}__class_one']) | (sel2(g['off']) != Dz[f'{nm}__class_both'])
                gb = min(float(np.abs(g['pool_off'][k][dif] - BAR).min()) for k in POOLS)
                say(f'      {nm}: differ one {v[0]} both {v[1]}; smallest |chord - bar| among them {gb:.2e}')

    if not all(CHK):
        say('STOP: an assert failed; no table follows.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say('   ASSERTS PASSED.')
    say('   Printed checks, rule on (set logic says 0 each; a picture within ~1e-6 of a bar could flip), violations over all groups, unmasked and source-out:')
    pairs = [(G[nm]['on'], G[nm]['off'], G[nm]['pool_on'], G[nm]['pool_off']) for nm in G] + \
            [(G[nm]['on_out'], G[nm]['off_out'], G[nm]['poolout_on'], G[nm]['poolout_off']) for nm in OUTK]
    v_none = sum(c_(a['one'][3] != b['one'][3]) for a, b, _, _ in pairs)
    v_pre = sum(c_(b['one'][0] & ~a['one'][0]) for a, b, _, _ in pairs)
    v_und = sum(c_(a['one'][2] & ~b['one'][2]) for a, b, _, _ in pairs)
    v_cln = sum(c_(b['clean'] & ~a['clean']) for a, b, _, _ in pairs)
    pmg = max(float(np.abs(np.minimum.reduce([pa[k] for k in POOLS]) - np.minimum.reduce([pb[k] for k in POOLS])).max()) for _, _, pa, pb in pairs)
    say(f'   NONE* != NONE {v_none}; PRE-NEAR not PRE*-NEAR {v_pre}; UNDATED*-ONLY not UNDATED-ONLY {v_und}; SINCE-CLEAN not SINCE*-CLEAN {v_cln}; '
        f'largest |min over new pools - min over v2 pools| {pmg:.2e}')

    say()
    say(f'bars D {BAR[0]:.6f} / S {BAR[1]:.6f}. Pools PRE {c_(off["PRE"])}, POST {c_(off["POST"])}, UNDATED {c_(off["UNDATED"])}. '
        f'New pools PRE* {szon[0]}, POST* {szon[1]}, UNDATED* {szon[2]}; total {sum(szon)}')
    say(f'ARTIST ERA of the {c_(~dated):,} undated originals (artists = distinct labels among the undated works in that case):')
    for i, lab in enumerate(CASES):
        m = ~dated & (ucode == i)
        na = len(np.unique(artist[m])) if i < 4 else 0
        say(f'   {lab:26s} works {c_(m):6d}' + (f' | artists {na}' if i < 4 else ' | (no label)'))
    for i, lab in ((0, 'PRE-BY-ARTIST'), (1, 'POST-BY-ARTIST')):
        m = ~dated & (ucode == i)
        say(f'   {lab}: works whose artist has exactly 1 dated work {c_(m & (npre + npost == 1))}; 2-4 {c_(m & (npre + npost >= 2) & (npre + npost <= 4))}; '
            f'5 or more {c_(m & (npre + npost >= 5))}')
        lb, cn = np.unique(artist[m], return_counts=True)
        o = np.argsort(-cn, kind='stable')[:10]
        say(f'   {lab}: top 10 labels by undated works (label: undated in case / dated before 1900 / dated 1900 on): ' + '; '.join(
            f'{lb[j]!r}: {cn[j]} / {int(npre[artist == lb[j]][0])} / {int(npost[artist == lb[j]][0])}' for j in o))
    say('   Labels that look like placeholders (exact strings; all rows / undated / its case for the undated): ' + '; '.join(
        f'{s!r} {c_(artist == s)} / {c_((artist == s) & ~dated)} / ' + (CASES[int(ucode[(artist == s) & ~dated][0])] if c_((artist == s) & ~dated) else '-')
        for s in ('nan', 'None', 'none', 'NaN', 'Unknown', 'unknown', 'Anonymous', 'anonymous', 'Unknown Artist', 'unknown artist')))

    def four(t, s):
        return '/'.join(str(c_(s & c)) for c in t)

    def row(label, d, sel):
        say(f'   {label:44s} n {c_(sel):5d} | one reader PRE*/SINCE*/UND*/NONE {four(d["one"], sel)} | SINCE*-CLEAN {c_(sel & d["clean"])}, '
            f'SINCE* with UND* near {c_(sel & d["undnear"])} | both readers PRE*/SINCE*/UND*/rest {four(d["both"], sel)}')

    def cat(tags):
        cs = [cells[t] for t in tags]
        g = {k: np.concatenate([c[k] for c in cs]) for k in ('LL', 'FF', 'kept', 'hits')}
        g['on'] = dict(one=tuple(np.concatenate([c['on']['one'][i] for c in cs]) for i in range(4)),
                       both=tuple(np.concatenate([c['on']['both'][i] for c in cs]) for i in range(4)),
                       clean=np.concatenate([c['on']['clean'] for c in cs]), undnear=np.concatenate([c['on']['undnear'] for c in cs]))
        return g

    say()
    say('TABLE 1. The main runs, artist rule on, unmasked. Per start: FF, FF and LL, FF and not LL; pooled also kept, hits.')
    for arm, tg, w in ARMS:
        for n in STARTS + ('pooled',):
            g = cat([f'qgc_{tg}_s{k}' for k in (STARTS if n == 'pooled' else (n,))])
            lab = f'{arm} {"s" + str(n) if n != "pooled" else "pooled"}'
            row(f'{lab} FF', g['on'], g['FF'])
            row(f'{lab} FF and LL', g['on'], g['FF'] & g['LL'])
            row(f'{lab} FF and not LL', g['on'], g['FF'] & ~g['LL'])
            if n == 'pooled':
                row(f'{lab} kept', g['on'], g['kept'])
                row(f'{lab} hits', g['on'], g['hits'])
    say()
    say('TABLE 2. Known answers, artist rule on (source work in its own pool; repaints also with the source and its near-duplicates left out)')
    for key, lab in (('P2', 'repaints before 1900, held out (P2)'), ('pos', 'repaints 1900 to 1969 (pos)'), ('today70', 'repaints 1970 on (T rows)')):
        g = known[key]
        al = np.ones(len(g['e']), bool)
        row(f'{lab}, all', g['on'], al)
        row(f'{lab}, FF', g['on'], g['FF'])
        row(f'{lab}, all, source left out', g['on_out'], al)
        row(f'{lab}, FF, source left out', g['on_out'], g['FF'])
    for key, lab in (('noise', 'noise requests'), ('prior', 'random requests')):
        g = known[key]
        row(f'{lab}, all', g['on'], np.ones(len(g['e']), bool))
        row(f'{lab}, FF', g['on'], g['FF'])
    for x in ('cone', 'sphere'):
        row(f'random requests, {x}', known['prior']['on'], known['prior']['kind'] == x)

    say()
    say('RULE (tr_readout_v2.rule): "ahead" = more at all five starts and 4 x sum_search >= 5 x sum_control and, when the control\'s sum is 0, sum_search >= 10; '
        '"behind" the same way round; otherwise "cannot tell". Artist rule on, unmasked.')
    rules = {'FF and SINCE*': lambda c: c['FF'] & c['on']['one'][1], 'FF and SINCE*-CLEAN': lambda c: c['FF'] & c['on']['clean'],
             'FF, SINCE* and LL': lambda c: c['FF'] & c['on']['one'][1] & c['LL'], 'FF, SINCE*-CLEAN and LL': lambda c: c['FF'] & c['on']['clean'] & c['LL'],
             'FF and PRE*-NEAR': lambda c: c['FF'] & c['on']['one'][0], 'FF and SINCE* (both readers)': lambda c: c['FF'] & c['on']['both'][1],
             'FF, SINCE* (both readers) and LL': lambda c: c['FF'] & c['on']['both'][1] & c['LL']}
    for lab, fn in rules.items():
        v = [c_(fn(cells[f'qgc_ra_s{n}'])) for n in STARTS]
        for x, tg in (('C1', 'pin'), ('C2', 'c2')):
            xx = [c_(fn(cells[f'qgc_{tg}_s{n}'])) for n in STARTS]
            say(f'   {lab}: search vs {x}: search {v} = {sum(v)}, {x} {xx} = {sum(xx)} -> {tr.rule(v, xx).upper()}')

    say()
    say('DATED ORIGINALS, own year hidden: the rule on the artist\'s OTHER dated originals; assigned side vs the true side of 1900.')
    title = np.asarray(C['title']).astype(str)
    key2 = np.char.add(np.char.add(artist, '\x01'), title)
    _, inv2 = np.unique(key2, return_inverse=True)
    inv2 = inv2.ravel()
    spre = np.bincount(inv2, weights=pre_r.astype(float)).astype(int)[inv2]
    spost = np.bincount(inv2, weights=post_r.astype(float)).astype(int)[inv2]
    loo_t = decide(artist != '', np.where(title != '', npre - spre, npre - pre_r.astype(int)), np.where(title != '', npost - spost, npost - post_r.astype(int)))
    for lab_v, code in (('leave-one-out', loo), ('variant: also leaving out the artist\'s other works with the same non-empty title', loo_t)):
        say(f'   {lab_v}:')
        for side, m, agree_c in (('truly before 1900', off['PRE'], 0), ('truly 1900 on', off['POST'], 1)):
            k = [c_(m & (code == i)) for i in range(5)]
            asg = k[0] + k[1]
            assert sum(k) == c_(m)
            say(f'      {side:18s} n {c_(m):6d} | assigned {asg} (agree {k[agree_c]}, disagree {k[1 - agree_c]}) | not assigned: straddle {k[2]}, '
                f'no other dated work {k[3]}, label empty {k[4]}')

    sv = {'bars': BAR, 'pool_sizes_on': np.array(szon), 'undated_case': ucode, 'loo_case': loo}
    for name, g in G.items():
        for k in POOLS:
            sv[f'{name}__poolon_{k}'] = g['pool_on'][k].astype(np.float32)
        sv[f'{name}__class_one_on'] = sel1(g['on']).astype(np.int8)
        sv[f'{name}__class_both_on'] = sel2(g['on']).astype(np.int8)
        sv[f'{name}__clean_on'] = g['on']['clean']
        if name in OUTK:
            sv[f'{name}__class_one_on_out'] = sel1(g['on_out']).astype(np.int8)
            sv[f'{name}__clean_on_out'] = g['on_out']['clean']
    with open(prefix + '.npz', 'xb') as fh:
        np.savez_compressed(fh, **sv)
    say(f'wrote {prefix}.txt and {prefix}.npz (per picture new-pool chords float32, classes 0 PRE* 1 SINCE* 2 UND* 3 NONE/rest, SINCE*-CLEAN; per corpus row '
        f'undated_case / loo_case 0..4 = {CASES}, -1 n/a)')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
