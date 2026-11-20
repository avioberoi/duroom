"""On/off classes split by undated originals: SINCE-CLEAN (no undated original within the bar in either reader) against SINCE with an undated original near, the both-reader classes per start, and the same split for the pictures painted with no search and the oracle lines.
Usage (CPU, cwd = the search folder): python -B onoff_since_v1.py OUTPREFIX -> OUTPREFIX.txt and OUTPREFIX.npz, each refused if it exists."""
import importlib.util
import os
import sys
import types

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
OO = f'{B}/tests12/onoff/out'
T3 = f'{B}/tests12/t3'
T3_FILE = f'{C_}/t3_measure_v5.py'
SCALES = f'{T3}/out/scales_v5.npz'
TR_FILE = f'{C_}/tr_readout_v2.py'
EV_FILE = f'{C_}/controls_evaluate.py'
LF_FILE = f'{C_}/leftover_fn.py'
BASIS = f'{B}/tests12/leftover/out/pre_v1_basis.npz'
AB_NPZ = f'{OO}/onoff_v3_AB.npz'
CC_NPZ = f'{OO}/onoff_v3_C.npz'
FF_NPZ = f'{OO}/onoff_ff_v1.npz'
DT_NPZ = f'{OO}/onoff_dated_v2.npz'
R2 = f'{B}/tests12/t2/renders_v1'
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
STARTS = (6, 7, 8, 9, 10)
POOLS = ('PRE', 'POST', 'UNDATED')
DINO, SIG = 'dinov3-l', 'siglip-so400m'
base = lambda p: os.path.basename(str(p))  # noqa: E731
c_ = lambda x: int(np.sum(x))  # noqa: E731
FH, OUT = [None], []


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


def load_shards(folder, n_expected):
    man, emb = [], []
    for s in range(64):
        d = f'{folder}/shard{s}'
        if not os.path.isdir(d):
            break
        m, e = np.load(f'{d}/manifest.npz', allow_pickle=True), np.load(f'{d}/emb.npz', allow_pickle=True)
        assert len(m['paths']) == len(e['paths']) and [base(p) for p in m['paths']] == [base(p) for p in e['paths']], d
        man.append(m)
        emb.append(e)
    row = np.concatenate([m['row'] for m in man]).astype(int)
    order = np.argsort(row)
    assert (row[order] == np.arange(n_expected)).all()
    cat = lambda parts, k: np.concatenate([p[k] for p in parts])[order]  # noqa: E731
    return {k: cat(man, k) for k in ('tag', 'seed', 'v', 'paths')}, {k: cat(emb, k) for k in ('emb_clip-vit-l-14', 'emb_dinov3-l', 'emb_siglip-so400m')}


def main(prefix):
    assert 'qgt' not in prefix
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x')
    tr = load_mod('tr_readout_v2', TR_FILE)
    A, Cz, Fz, Dz = (np.load(f, allow_pickle=True) for f in (AB_NPZ, CC_NPZ, FF_NPZ, DT_NPZ))
    BAR = np.asarray(Dz['bars'], np.float64)
    say('onoff_since_v1.py. Counts only.')
    say(f'bars D {BAR[0]:.6f} / S {BAR[1]:.6f}; pools PRE / POST / UNDATED {Dz["pool_sizes"].tolist()}')
    anyn = lambda x: np.any(x <= BAR, axis=1)  # noqa: E731
    alln = lambda x: np.all(x <= BAR, axis=1)  # noqa: E731

    def derive(name):
        P = {k: np.asarray(Dz[f'{name}__pool_{k}'], np.float64) for k in POOLS}
        one, both = np.asarray(Dz[f'{name}__class_one']), np.asarray(Dz[f'{name}__class_both'])
        since = one == 1
        return dict(since=since, clean=since & ~anyn(P['UNDATED']), und=since & anyn(P['UNDATED']), post_both=since & alln(P['POST']),
                    pre2=both == 0, since2=both == 1, und2=both == 2, rest2=both == 3, one=one)

    cells = {}
    for arm, tg, _ in ARMS:
        for n in STARTS:
            t = f'qgc_{tg}_s{n}'
            d = derive(t)
            d.update(FF=Fz[f'{t}__FF'], LL=Cz[f'{t}__LL'], kept=Cz[f'{t}__kept'], hits=Cz[f'{t}__hits'])
            cells[t] = d
    known = {}
    for k in ('P2', 'pos', 'today70', 'noise', 'prior'):
        d = derive(k)
        d.update(FF=Fz[f'{k}__FF'], LL=A[f'{k}__LL'])
        if k == 'prior':
            d['kind'] = A['prior__kind']
        known[k] = d
    say('CHECK: SINCE-CLEAN + SINCE with an undated original near = SINCE, every group: ' + str(all(np.array_equal(d['clean'] | d['und'], d['since']) and not (d['clean'] & d['und']).any()
                                                                                                  for d in list(cells.values()) + list(known.values()))))
    cat = lambda tags, k: np.concatenate([cells[t][k] for t in tags])  # noqa: E731
    keys = ('FF', 'LL', 'kept', 'hits', 'since', 'clean', 'und', 'post_both', 'pre2', 'since2', 'und2', 'rest2')

    def rowA(label, g, sel):
        say(f'   {label:40s} n {c_(sel):5d} | SINCE {c_(sel & g["since"]):5d} | SINCE-CLEAN {c_(sel & g["clean"]):5d} | SINCE, undated near {c_(sel & g["und"]):5d}')

    say()
    say('A. SINCE (one reader) / SINCE-CLEAN (no undated original within the bar in either reader) / SINCE with an undated original near')
    for arm, tg, w in ARMS:
        for n in STARTS + ('pooled',):
            g = {k: cat([f'qgc_{tg}_s{x}' for x in (STARTS if n == 'pooled' else (n,))], k) for k in keys}
            lab = f'{arm} {"s" + str(n) if n != "pooled" else "pooled"}'
            rowA(f'{lab} FF', g, g['FF'])
            rowA(f'{lab} FF and LL', g, g['FF'] & g['LL'])
            rowA(f'{lab} FF and not LL', g, g['FF'] & ~g['LL'])
            if n == 'pooled':
                rowA(f'{lab} kept', g, g['kept'])
                rowA(f'{lab} hits', g, g['hits'])
    for k, lab in (('P2', 'repaints before 1900 (P2)'), ('pos', 'repaints 1900 to 1969'), ('today70', 'repaints 1970 on'), ('noise', 'noise requests'), ('prior', 'random requests')):
        g = known[k]
        rowA(f'{lab}, all', g, np.ones(len(g['FF']), bool))
        rowA(f'{lab}, FF', g, g['FF'])
    say('   (known answers: the source work counts in its own pool, as onoff_dated_v2 unmasked)')
    rule_hdr = ('RULE (tr_readout_v2.rule): "ahead" = more at all five starts and 4 x sum_search >= 5 x sum_control and, when the control\'s sum is 0, '
                'sum_search >= 10; "behind" the same way round; otherwise "cannot tell"')

    def rules(lines):
        say(rule_hdr)
        for lab, fn in lines.items():
            v = [c_(fn(cells[f'qgc_ra_s{n}'])) for n in STARTS]
            for x, tg in (('C1', 'pin'), ('C2', 'c2')):
                xx = [c_(fn(cells[f'qgc_{tg}_s{n}'])) for n in STARTS]
                say(f'   {lab}: search vs {x}: search {v} = {sum(v)}, {x} {xx} = {sum(xx)} -> {tr.rule(v, xx).upper()}')
    rules({'FF and SINCE-CLEAN': lambda c: c['FF'] & c['clean'], 'FF, SINCE-CLEAN and LL': lambda c: c['FF'] & c['clean'] & c['LL']})
    say()
    say('B. Both-reader classes (onoff_dated_v2, strict version) per start: PRE / SINCE / UND / rest; and among the one-reader SINCE pictures, POST near in both readers')
    for arm, tg, w in ARMS:
        for n in STARTS + ('pooled',):
            g = {k: cat([f'qgc_{tg}_s{x}' for x in (STARTS if n == 'pooled' else (n,))], k) for k in keys}
            lab = f'{arm} {"s" + str(n) if n != "pooled" else "pooled"}'
            for sl, sel in (('FF', g['FF']), ('FF and LL', g['FF'] & g['LL'])):
                say(f'   {lab + " " + sl:40s} n {c_(sel):5d} | both readers PRE/SINCE/UND/rest {c_(sel & g["pre2"])}/{c_(sel & g["since2"])}/{c_(sel & g["und2"])}/{c_(sel & g["rest2"])} | '
                    f'one-reader SINCE {c_(sel & g["since"])}, of them POST near in both readers {c_(sel & g["post_both"])}')
    rules({'FF and SINCE (both readers)': lambda c: c['FF'] & c['since2'], 'FF, SINCE (both readers) and LL': lambda c: c['FF'] & c['since2'] & c['LL']})
    say()

    t3 = load_mod('t3_measure_v5', T3_FILE)
    ev = load_mod('controls_evaluate', EV_FILE)
    lf = load_mod('leftover_fn', LF_FILE)
    BZ = np.load(BASIS, allow_pickle=True)
    line = {o: float(BZ[f'null_q99_{o}']) for o in (DINO, SIG)}
    fit = {o: (np.asarray(BZ[f'mean_{o}'], np.float64), np.asarray(BZ[f'V_{o}'], np.float64)) for o in (DINO, SIG)}
    ns = types.SimpleNamespace(corpus='real', root=O, scales=SCALES)
    C, RP, idx = t3.load_common(ns)
    sc = t3.load_scales(ns)
    assert np.allclose(np.asarray(sc['cal'][99]['theta'] * sc['cal'][99]['med'], np.float64), BAR, rtol=0, atol=0)
    pools = {'PRE': C['dated'] & (C['year'] < 1900), 'POST': C['dated'] & (C['year'] >= 1900), 'UNDATED': ~C['dated']}
    assert [c_(pools[k]) for k in POOLS] == Dz['pool_sizes'].tolist()
    P, T, Pz = ev.references(O)
    OBS, K, unit, chord, FID = ev.OBS, ev.K, ev.unit, ev.chord, float(ev.FID)
    assert tuple(OBS) == tuple(t3.OBS)
    assert np.array_equal(P['id'], ev.pid(RP['P']['paths']))
    lev = {o: float(np.percentile(P[o]['own'], 99)) for o in OBS}
    Psrc = RP['P']['src']

    def far_alone(emb):
        out = []
        for o in OBS:
            D = chord(unit(emb[o]) @ P[o]['E'].T)
            out.append(np.sort(D, axis=1)[:, :K].mean(1) > lev[o])
        return out[0] & out[1]

    groups = {}   # label -> dict(E, v, clip, srcs list of corpus-row lists, srcdesc)
    R = np.load(f'{O}/eval/regions/regions_requests.npz', allow_pickle=True)
    M, E = load_shards(f'{O}/eval/regions/stage1', len(R['v']))
    fam = np.array([str(x) for x in R['family']])
    assert (np.array([str(x) for x in M['tag']]) == fam).all() and len(fam) == 868
    a_row, b_row = R['a_row'].astype(int), R['b_row'].astype(int)

    def add(label, s, Em, v, srcs, desc):
        g = groups.setdefault(label, dict(E={o: [] for o in OBS}, clip=[], v=[], srcs=[], desc=desc))
        for o in OBS:
            g['E'][o].append(Em[f'emb_{o}'][s])
        g['clip'].append(Em['emb_clip-vit-l-14'][s])
        g['v'].append(v[s])
        g['srcs'] += srcs

    s = np.flatnonzero(fam == 'IW')
    add('midpoints (IW)', s, E, R['v'], [[int(Psrc[a_row[i]]), int(Psrc[b_row[i]])] for i in s], 'the two works (P rows a, b)')
    s = np.flatnonzero(fam == 'CL')
    add('between centres (CL)', s, E, R['v'], [[] for _ in s], 'none')
    for f_ in ('PP35', 'RD35', 'SH35', 'GD35'):
        s = np.flatnonzero(fam == f_)
        add('35 deg steps (PP35, RD35, SH35, GD35)', s, E, R['v'], [[int(Psrc[b_row[i]])] for i in s], 'the base work (P row b)')
    Q = np.load(f'{O}/eval/probe/probe_requests.npz', allow_pickle=True)
    M, E = load_shards(f'{O}/eval/probe/stage1', len(Q['v']))
    fam = np.array([str(x) for x in Q['family']])
    assert (np.array([str(x) for x in M['tag']]) == fam).all() and len(fam) == 1056
    srow, partners = Q['start_row'].astype(int), Q['partner_rows'].astype(int)
    start_rows_missing = []

    def obs_to_corpus(r_):
        i = C['raw_ids'][r_]
        if i in C['row_of']:
            return [int(C['row_of'][i])]
        start_rows_missing.append(int(r_))
        return []
    for f_ in ('LR', 'L2', 'LT'):
        s = np.flatnonzero((fam == f_) & (Q['filter_fail'].astype(int) == 0))
        add('lines past a work, in window (LR, L2, LT)', s, E, Q['v'],
            [[int(Psrc[j]) for j in partners[i] if j >= 0] + (obs_to_corpus(srow[i]) if srow[i] >= 0 else []) for i in s], 'the partner works (P rows) and the start work')
    man =np.load(f'{R2}/prior256_v1/prior/manifest.npz', allow_pickle=True)
    em = np.load(f'{R2}/prior256_v1/emb.npz', allow_pickle=True)
    k0 = len([ln for ln in open(f'{R2}/prior256_v1/inputs/expected_kat.txt') if ln.strip()])
    assert [base(p) for p in em['paths'][k0:]] == [base(p) for p in man['paths']]
    kind = np.array([str(x) for x in man['kind']])
    emk = {k: em[k][k0:] for k in ('emb_clip-vit-l-14', 'emb_dinov3-l', 'emb_siglip-so400m')}
    for kd in ('cone', 'sphere'):
        s = np.flatnonzero(kind == kd)
        add(f'random requests, {kd} (H200)', s, emk, man['v'], [[] for _ in s], 'none')
    q = np.load(f'{B}/tests12/nosearch/nosearch_v5_oracle_requests.npz', allow_pickle=True)
    oem = np.load(f'{B}/tests12/afterrun/oracle/oracle_emb_h200_v1.npz', allow_pickle=True)
    assert [base(p) for p in oem['paths']] == [str(x) for x in q['frame_file']] and len(oem['paths']) == 74
    inp = np.load(f'{B}/inputs/oracle_inputs.npz', allow_pickle=True)
    starts, stags, targets = unit(inp['starts']), [str(t) for t in inp['start_tags']], unit(inp['targets'])
    Ec = unit(np.load(ev.IDS, allow_pickle=True)['embeddings'])
    srow_of = {}
    for k, tg_ in enumerate(stags):
        c = Ec @ starts[k]
        srow_of[tg_] = int(np.argmax(c))
        assert c[srow_of[tg_]] > 0.99999
    pz = np.load(f'{O}/pos_emb.npz', allow_pickle=True)
    pos_v = unit(pz['vec'])
    assert np.array_equal(RP['pos']['paths'], np.array([str(p) for p in pz['paths']]))
    tgt = q['target'].astype(int)
    trow = {j: int(np.argmax(pos_v @ targets[j])) for j in sorted(set(tgt.tolist()) - {-1})}
    s = np.flatnonzero(tgt >= 0)
    qstart = [str(x) for x in q['start']]
    add('lines aimed at real later works (71)', s, oem, q['v'], [obs_to_corpus(srow_of[qstart[i]]) + [int(RP['pos']['src'][trow[tgt[i]]])] for i in s],
        'the start work and the target work')

    allsrc = sorted({x for g in groups.values() for ss in g['srcs'] for x in ss})
    dups = t3.dup_rows(C, allsrc)
    full = t3.Ctx(C, idx, dups)
    say('C. PICTURES PAINTED WITH NO SEARCH (nosearch_v5 families; oracle lines of nosearch_oracle_v1). Exact readings as those scripts read them.')
    say(f'   start works not in the corpus rows (cannot be masked): {sorted(set(start_rows_missing))}')
    sv = {}
    res = {}
    for lab, g in groups.items():
        Em = {o: np.concatenate(g['E'][o]) for o in OBS}
        clip, v = np.concatenate(g['clip']), np.concatenate(g['v'])
        n = len(clip)
        ff = far_alone(Em) & (chord((unit(v) * unit(clip)).sum(1)) <= FID)
        far = far_alone(Em)
        ll = (lf.share(Em[DINO], *fit[DINO]) > line[DINO]) & (lf.share(Em[SIG], *fit[SIG]) > line[SIG])
        masks = [np.unique(np.concatenate([full.rows(x) for x in ss])) if ss else np.zeros(0, int) for ss in g['srcs']]
        d_un = t3.nearest(Em, C['E'], [np.zeros(0, int)] * n)
        d_m = t3.nearest(Em, C['E'], masks)
        e_un = np.stack([d_un[o][:, 0] for o in OBS], 1)
        e_m = np.stack([d_m[o][:, 0] for o in OBS], 1)
        pc = {k: np.stack([x[:, 0] for x in t3.nearest(Em, C['E'], [np.zeros(0, int)] * n, pools[k]).values()], 1) for k in POOLS}
        assert np.allclose(np.minimum.reduce([pc[k] for k in POOLS]), e_un, rtol=0, atol=1e-5)
        pre = anyn(pc['PRE'])
        since = ~pre & anyn(pc['POST'])
        und = ~pre & ~since & anyn(pc['UNDATED'])
        none = ~pre & ~since & ~und
        assert np.array_equal(none, np.all(e_un > BAR, axis=1))
        clean = since & ~anyn(pc['UNDATED'])
        res[lab] = dict(n=n, ff=ff, far=far, ll=ll, e_un=e_un, e_m=e_m, cls=(pre, since, und, none), clean=clean, desc=g['desc'], nmask=sum(1 for m in masks if len(m)))
        key = lab.split(' (')[0].replace(' ', '_').replace(',', '')
        sv.update({f'{key}__ff': ff, f'{key}__far': far, f'{key}__ll': ll, f'{key}__e_unmasked': e_un, f'{key}__e_masked': e_m,
                   f'{key}__class_one': np.select([pre, since, und, none], [0, 1, 2, 3], -1), f'{key}__since_clean': clean})
        for k in POOLS:
            sv[f'{key}__pool_{k}'] = pc[k]
    tot = [l_ for l_ in res if l_ != 'lines aimed at real later works (71)']
    three = lambda x, s: f'{c_(s & np.all(x <= BAR, axis=1))}/{c_(s & ~np.all(x <= BAR, axis=1) & ~np.all(x > BAR, axis=1))}/{c_(s & np.all(x > BAR, axis=1))}'  # noqa: E731
    say('   columns: n | near both / near one / far both, nothing masked | the same, source works masked | PRE / SINCE / UNDATED-ONLY / NONE (unmasked) | SINCE-CLEAN | LL')

    def rowC(lab, r, sel):
        say(f'   {lab:52s} n {c_(sel):5d} | unmasked {three(r["e_un"], sel)} | masked {three(r["e_m"], sel)} | {"/".join(str(c_(sel & x)) for x in r["cls"])} | '
            f'SINCE-CLEAN {c_(sel & r["clean"])} | LL {c_(sel & r["ll"])}')
    tot_r = {k: np.concatenate([res[l_][k] for l_ in tot]) for k in ('ff', 'll', 'e_un', 'e_m', 'clean')}
    tot_r['cls'] = tuple(np.concatenate([res[l_]['cls'][i] for l_ in tot]) for i in range(4))
    for lab, r in list(res.items()) + [('total of the six groups', tot_r)]:
        al = np.ones(len(r['ff']), bool)
        if 'desc' in r:
            say(f'  {lab}: source works masked = {r["desc"]} ({r["nmask"]} of {r["n"]} pictures have some)')
        rowC(f'{lab}, all', r, al)
        rowC(f'{lab}, FF', r, r['ff'])
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'wrote {prefix}.txt and {prefix}.npz (part C per picture)')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
