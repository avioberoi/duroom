"""onoff_v3.py: counts of large leftover, reading 1, reading 2 and the nearest-work rule (FAR / NEAR / MIXED) per picture group.
Usage (cwd = the search folder): python -B onoff_v3.py OUTPREFIX AB|C -> OUTPREFIX.txt + OUTPREFIX.npz; refuses to overwrite."""
import importlib.util
import os
import re
import sys
import types

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
R, L, E = f'{O}/runs', f'{B}/logs', f'{O}/eval'
LO = f'{B}/tests12/leftover'
R2 = f'{B}/tests12/t2/renders_v1'
T3 = f'{B}/tests12/t3'
EV_FILE = f'{C_}/controls_evaluate.py'
LF_FILE = f'{C_}/leftover_fn.py'
CLO_FILE = f'{C_}/compare_leftover_v1.py'
T3_FILE = f'{C_}/t3_measure_v5.py'
SCORES = f'{LO}/out/pre_v1_scores.npz'
BASIS = f'{LO}/out/pre_v1_basis.npz'
SCALES = f'{T3}/out/scales_v5.npz'
GUARDS = f'{T3}/out/guards_v3.npz'
DINO, SIG = 'dinov3-l', 'siglip-so400m'
RD = (DINO, SIG)
KFAR, KNB = 15, 10
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
STARTS = (6, 7, 8, 9, 10)


HEADER = ("Nearest-work bars: theta_99 x median per reader from the scales file; FAR = both readers above, NEAR = both at or below, MIXED = the rest. "
          "Index: all corpus rows; the source work and its near-duplicates are masked.")
pid = lambda paths: np.array([int(re.search(r'push_(\d+)\.png$', str(p)).group(1)) for p in paths])
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
    say(f'   {label}: got {got} | expected {exp} | {"same" if got == exp else "DIFFERENT"}')


def stop_if_bad(prefix):
    if not all(CHK):
        say('STOP: a required number did not reproduce; no table follows.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x').write('\n'.join(OUT) + '\n')
        sys.exit(3)


c_ = lambda x: int(np.sum(x))


def main(prefix, part):
    assert part in ('AB', 'C') and 'qgt' not in prefix
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x')
    ev = load_mod('controls_evaluate', EV_FILE)
    lf = load_mod('leftover_fn', LF_FILE)
    clo = load_mod('compare_leftover_v1', CLO_FILE)
    t3 = load_mod('t3_measure_v5', T3_FILE)
    SV, BZ = np.load(SCORES, allow_pickle=True), np.load(BASIS, allow_pickle=True)
    assert int(BZ['K']) == 32
    line = {o: float(BZ[f'null_q99_{o}']) for o in RD}
    for o in RD:
        assert float(np.percentile(np.asarray(BZ[f'crossfit_share_{o}'], np.float64), 99)) == line[o]
    fit = {o: (np.asarray(BZ[f'mean_{o}'], np.float64), np.asarray(BZ[f'V_{o}'], np.float64)) for o in RD}
    say(f'onoff_v3.py, part {part}')
    say('leftover line (null 99th): ' + ', '.join(f'{o} {line[o]:.6f}' for o in RD))
    say(HEADER)
    say()

    P, T, Pz = ev.references(O)
    Tz = np.load(f'{O}/push_today_emb.npz', allow_pickle=True)
    Tid, Tart, Tyear = T['id'], T['artist'], T['year']
    assert np.array_equal(Tid, pid(Tz['paths'])) and len(Tid) == 2048
    TE = {o: T[o]['E'] for o in RD}
    Tbar = {o: float(T[o]['bar']) for o in RD}
    own, bar = {}, {}
    for o in RD:
        D = ev.chord(TE[o] @ TE[o].T)
        D[Tart[:, None] == Tart[None, :]] = np.inf
        own[o] = np.sort(D, axis=1)[:, :KFAR].mean(1)
        bar[o] = float(np.percentile(own[o], 99))
        assert np.allclose(own[o], T[o]['own'], rtol=0, atol=1e-12) and bar[o] == Tbar[o]
    TL = {o: lf.share(Tz[f'emb_{o}'], *fit[o]) for o in RD}
    t70 = Tyear >= 1970
    dsh = max(float(np.abs(TL[o][t70] - np.asarray(SV[f'repaint__today,__{o}'])).max()) for o in RD)
    assert dsh < 1e-12, dsh
    Tov = (TL[DINO] > line[DINO]) & (TL[SIG] > line[SIG])
    TR = {o: clo.resid(Tz[f'emb_{o}'], *fit[o]) for o in RD}
    row_of = {int(i): k for k, i in enumerate(Tid.tolist())}

    def measure(emb, masks):
        M = dict(sh={}, d15={}, nb={}, nbc={})
        for o in RD:
            X = np.asarray(emb[o])
            D = ev.chord(ev.unit(X) @ TE[o].T)
            Sc = clo.sims(clo.resid(X, *fit[o]), TR[o])
            for i, ks in enumerate(masks):
                if len(ks):
                    D[i, ks] = np.inf
                    Sc[i, ks] = -np.inf
            M['d15'][o] = np.sort(D, axis=1)[:, :KFAR].mean(1)
            nn = np.argsort(D, axis=1, kind='stable')[:, :KNB]
            _, nc = clo.topk(Sc, KNB)
            assert np.isfinite(np.take_along_axis(D, nn, 1)).all() and np.isfinite(np.take_along_axis(Sc, nc, 1)).all()
            M['nb'][o], M['nbc'][o] = Tov[nn].any(1), Tov[nc].any(1)
            M['sh'][o] = lf.share(X, *fit[o])
        M['ov'] = (M['sh'][DINO] > line[DINO]) & (M['sh'][SIG] > line[SIG])
        M['far'] = (M['d15'][DINO] > Tbar[DINO]) & (M['d15'][SIG] > Tbar[SIG])
        return M

    a = types.SimpleNamespace(corpus='real', root=O, rows=1921, rounds=60, per_round=32, scales=SCALES,
                              run_tmpl=f'{O}/runs/{{tag}}_committee_euclidean_min_seedrow/all_candidates.npz', exact_tmpl=f'{O}/eval/{{tag}}_emb.npz',
                              acc_prior=f'{R2}/prior256_v1', acc_noise=f'{R2}/noise160_v1', acc_aim_prior=None, acc_aim_noise=None)
    assert t3.CELL['L'] == 99 and t3.DUP == 0.99
    C, RP, idx = t3.load_common(a)
    dmod = sys.modules.get('data')
    say(f'corpus: {len(C["ids"])} rows; data module {getattr(dmod, "__file__", None)}; repaints ' + ', '.join(f'{n} {len(RP[n]["src"])}' for n in RP))
    assert dmod is not None and os.path.dirname(os.path.abspath(dmod.__file__)) == os.path.abspath(t3.RETRO), 'data module is not the one in t3.RETRO'
    sc = t3.load_scales(a)
    cal = sc['cal'][99]
    BAR = np.asarray(cal['theta'] * cal['med'], np.float64)
    t3.t_readings(C, RP, idx, sc)   # asserts T's chords equal the scales file's T_e (1e-6)
    say(f'bars: theta_99 {cal["theta"]:.6f} x med DINOv3 {cal["med"][0]:.6f} / SigLIP {cal["med"][1]:.6f} = {BAR[0]:.6f} / {BAR[1]:.6f}; '
        f'T chords recomputed = scales_v5 T_e (1e-6, t_readings)')
    want_tags = [f'qgc_c2_s{n}' for n in STARTS] if part == 'AB' else [f'qgc_{tg}_s{n}' for _, tg, _ in ARMS for n in STARTS]
    arm_of = {f'qgc_{tg}_s{n}': arm for arm, tg, _ in ARMS for n in STARTS}
    cells = {}
    for t in want_tags:
        assert 'qgt' not in t and 'qgt' not in a.run_tmpl.format(tag=t) and 'qgt' not in a.exact_tmpl.format(tag=t)
        vt = 'qgc_ra_s' + t.rsplit('_s', 1)[1]
        cells[t] = t3.load_cell(t, a.run_tmpl, a.exact_tmpl, a.rows, C, layout=(a.rounds, a.per_round), arm=arm_of[t], vtag=vt)
        assert cells[t]['valid'], cells[t]['why']
    srcs = np.concatenate([RP[n]['src'] for n in ('T', 'P', 'P2', 'pos')] + [np.array([c['start'] for c in cells.values()], int)])
    dups = t3.dup_rows(C, srcs)
    full, artc = t3.Ctx(C, idx, dups), t3.Ctx(C, idx, dups, artist_mask=True)
    ne = lambda Em, src, ctx: np.asarray(t3.readings(dict(E=Em, src=np.asarray(src)), ctx)['e'], np.float64)
    farE = lambda e: np.all(e > BAR, axis=1)
    nearE = lambda e: np.all(e <= BAR, axis=1)

    g = np.load(GUARDS, allow_pickle=True)
    eP2 = ne(RP['P2']['E'], RP['P2']['src'], full)
    acc = t3.load_acc(a, need_aim=False)
    eA = ne(acc['E'], acc['src'], full)
    c2 = [cells[f'qgc_c2_s{n}'] for n in STARTS]
    eC2 = ne({n: np.concatenate([c['E'][n] for c in c2]) for n in t3.OBS}, np.concatenate([c['src'] for c in c2]), full)
    say('CHECK against the guard file guards_v3.npz, through the measure\'s functions against the full corpus')
    for key, lab, e in (('gno1', 'guard P2', eP2), ('gno3', 'guard ACC', eA), ('gno4c2', 'guard qgc C2 cells', eC2)):
        d = float(np.abs(e - np.asarray(g[f'{key}__e'], np.float64)).max())
        same = bool(np.array_equal(farE(e), np.asarray(g[f'{key}__E'], bool)))
        CHK.append(d < 1e-6 and same)
        say(f'   {lab}: n {len(e)}; max |chord here - guard file| {d:.2e}; FAR flags = guard E flags picture by picture: {same}; '
            f'smallest |chord - bar| {float(np.abs(e - BAR).min()):.2e}')
    want('guard qgc C2: FAR (= E), n, against the guard file', (c_(farE(eC2)), len(eC2)), (c_(g['gno4c2__E']), len(g['gno4c2__E'])))
    stop_if_bad(prefix)
    say('   check against the guard file passed.')
    say()

    sv = {'bars': BAR, 'theta_99': np.float64(cal['theta']), 'med': np.asarray(cal['med'], np.float64)}
    fnm = lambda e, s: f'{c_(s & farE(e))}/{c_(s & nearE(e))}/{c_(s & ~farE(e) & ~nearE(e))}'

    def rowN(label, M, e, sel=None):
        sel = np.ones(len(M['ov']), bool) if sel is None else sel
        ov = M['ov'] & sel
        nbp, nbc = M['nb'][DINO] & M['nb'][SIG], M['nbc'][DINO] & M['nbc'][SIG]
        dist = ' | '.join(f'{n_} med {np.median(e[sel, j]):.4f} p5 {np.percentile(e[sel, j], 5):.4f} p95 {np.percentile(e[sel, j], 95):.4f}'
                          for j, n_ in enumerate(('D', 'S'))) if c_(sel) else 'empty'
        say(f'   {label:44s} n {c_(sel):5d} LL {c_(ov):4d} | LL F/N/M {fnm(e, ov)} | all {fnm(e, sel)} || R2plain ON {fnm(e, ov & nbp)} OFF {fnm(e, ov & ~nbp)} '
            f'| R2cos ON {fnm(e, ov & nbc)} OFF {fnm(e, ov & ~nbc)} || R1 ON {fnm(e, ov & ~M["far"])} OFF {fnm(e, ov & M["far"])} || {dist}')

    def keep(key, M, e, extra):
        sv[f'{key}__e'] = e
        sv[f'{key}__FAR'], sv[f'{key}__NEAR'] = farE(e), nearE(e)
        sv[f'{key}__LL'], sv[f'{key}__R1far'] = M['ov'], M['far']
        sv[f'{key}__R2plain_both'], sv[f'{key}__R2cos_both'] = M['nb'][DINO] & M['nb'][SIG], M['nbc'][DINO] & M['nbc'][SIG]
        for k_, v in extra.items():
            sv[f'{key}__{k_}'] = v

    legend = ('Columns: n; LL = large leftover (over the past\'s leftover line in both readers); F/N/M = FAR / NEAR / MIXED by the nearest-work rule; '
              '"LL F/N/M" among LL, "all" among all n; R2plain / R2cos = reading 2, "in both lists" (an over-line T row among the 10 nearest in '
              'the DINOv3 list AND in the SigLIP list), ON / OFF among LL; R1 = reading 1 (OFF = far against T by the mean of 15), ON / OFF among LL; '
              f'then the nearest-original chord per reader over all n: median, 5th, 95th percentile. Bars D {BAR[0]:.4f} / S {BAR[1]:.4f}.')

    if part == 'AB':
        groups = {}
        i70 = np.flatnonzero(t70)
        own_m = [np.array([i]) for i in i70]
        art_m = [np.flatnonzero(Tart == Tart[i]) for i in i70]
        e70 = {o: np.asarray(Tz[f'emb_{o}'])[i70] for o in RD}
        groups['today70'] = (measure(e70, own_m), measure(e70, art_m), len(i70), 0, max(len(m) for m in art_m))
        pths = {'today70': np.array([str(p) for p in Tz['paths']])[i70]}
        for key, f in (('pos', 'pos_emb.npz'), ('P2', 'push2_emb.npz')):
            z = np.load(f'{O}/{f}', allow_pickle=True)
            ids, art = pid(z['paths']), np.array([str(x) for x in z['artist']])
            inT = np.array([int(i) in row_of for i in ids])
            for i in np.flatnonzero(inT):
                assert Tart[row_of[int(ids[i])]] == art[i], (key, ids[i])
            om = [np.array([row_of[int(i)]]) if int(i) in row_of else np.zeros(0, int) for i in ids]
            am = [np.union1d(np.flatnonzero(Tart == x), m) for x, m in zip(art, om)]
            emb = {o: z[f'emb_{o}'] for o in RD}
            M1, M2 = measure(emb, om), measure(emb, am)
            for o in RD:
                assert np.abs(M1['sh'][o] - np.asarray(SV[f'repaint__{key}__{o}'])).max() < 1e-12, (key, o)
            groups[key] = (M1, M2, len(ids), c_(inT), c_([len(m) > 0 for m in am]))
            pths[key] = np.array([str(p) for p in z['paths']])
        for o in RD:
            assert np.abs(groups['today70'][0]['sh'][o] - np.asarray(SV[f'repaint__today,__{o}'])).max() < 1e-12
        rq = {}
        for key, sub, mf, kk in (('noise', 'noise160_v1', 'noise/manifest.npz', 'tag'), ('prior', 'prior256_v1', 'prior/manifest.npz', 'kind')):
            em = np.load(f'{R2}/{sub}/emb.npz', allow_pickle=True)
            kat = [ln.strip() for ln in open(f'{R2}/{sub}/inputs/expected_kat.txt') if ln.strip()]
            assert [str(p) for p in em['paths'][:len(kat)]] == kat
            man = np.load(f'{R2}/{sub}/{mf}', allow_pickle=True)
            k = len(kat)
            assert [str(p) for p in em['paths'][k:]] == [str(p) for p in man['paths']]
            kind = np.array([str(x) for x in man[kk]])
            assert np.array_equal(kind, np.array([str(x) for x in SV[f'{key}__kind']])), key
            M = measure({o: em[f'emb_{o}'][k:] for o in RD}, [np.zeros(0, int)] * len(kind))
            for o in RD:
                assert np.abs(M['sh'][o] - np.asarray(SV[f'{key}__{o}'])).max() < 1e-12, (key, o)
            rq[key] = (M, kind, np.array([str(p) for p in em['paths'][k:]]))
        # nearest-work chords, the measure's readings of the same pictures (rows aligned by path)
        TS = RP['T']
        assert np.array_equal(TS['paths'], np.array([str(p) for p in Tz['paths']]))
        say(f'T rows from 1970 on: evaluator years {c_(t70)}, corpus years {c_(C["year"][TS["src"]] >= 1970)}, rows that differ {c_(t70 != (C["year"][TS["src"]] >= 1970))} '
            f'(the group is chosen by the evaluator years)')
        NE = {'today70': (ne({n: TS['E'][n][i70] for n in t3.OBS}, TS['src'][i70], full), ne({n: TS['E'][n][i70] for n in t3.OBS}, TS['src'][i70], artc))}
        assert np.allclose(NE['today70'][0], np.asarray(sc['T_e'], np.float64)[i70], rtol=0, atol=1e-6)
        for key in ('pos', 'P2'):
            s = RP[key]
            assert np.array_equal(s['paths'], pths[key]), key
            NE[key] = (eP2 if key == 'P2' else ne(s['E'], s['src'], full), ne(s['E'], s['src'], artc))
        pri = np.char.startswith(acc['kind'].astype(str), 'prior')
        assert np.array_equal(acc['paths'][pri], rq['prior'][2]) and np.array_equal(acc['paths'][~pri], rq['noise'][2])
        assert np.array_equal(np.char.replace(acc['kind'][~pri].astype(str), 'noise_', ''), rq['noise'][1])
        NE['noise'], NE['prior'] = eA[~pri], eA[pri]
        nM, nk = rq['noise'][0], rq['noise'][1]
        pM, pk = rq['prior'][0], rq['prior'][1]
        say('Known-answer groups. ' + legend)
        say(' own (source) work out:')
        rowN('repaints of works from 1970 on (T rows)', groups['today70'][0], NE['today70'][0])
        rowN('repaints of works 1900 to 1969 (pos)', groups['pos'][0], NE['pos'][0])
        rowN('repaints of earlier works, held out (P2)', groups['P2'][0], NE['P2'][0])
        for x in 'abcde':
            rowN(f'noise pictures, kind {x}', nM, NE['noise'], nk == x)
        rowN('noise pictures, all kinds', nM, NE['noise'])
        rowN('random requests, in the region (cone)', pM, NE['prior'], pk == 'cone')
        rowN('random requests, on the sphere', pM, NE['prior'], pk == 'sphere')
        rowN('random requests, all', pM, NE['prior'])
        say(' variant: every work of the same artist out (readings 1 and 2: every T row of the artist; nearest work: every corpus row of the artist):')
        rowN('repaints of works from 1970 on, artist out', groups['today70'][1], NE['today70'][1])
        rowN('repaints of works 1900 to 1969, artist out', groups['pos'][1], NE['pos'][1])
        rowN('repaints of earlier works (P2), artist out', groups['P2'][1], NE['P2'][1])
        say()
        for key in ('today70', 'pos', 'P2'):
            keep(key, groups[key][0], NE[key][0], {'paths': pths[key], 'src_row': (RP['T']['src'][i70] if key == 'today70' else RP[key]['src'])})
            keep(key + '_artistout', groups[key][1], NE[key][1], {})
        keep('noise', nM, NE['noise'], {'kind': nk, 'paths': rq['noise'][2]})
        keep('prior', pM, NE['prior'], {'kind': pk, 'paths': rq['prior'][2]})
    else:
        ids, q = ev.startids(O), ev.qfun(O)
        cl = {}
        swid = []
        for arm, tg, _ in ARMS:
            for n in STARTS:
                tag = f'qgc_{tg}_s{n}'
                d = f'{R}/{tag}_committee_euclidean_min_seedrow'
                assert 'qgt' not in d
                r = ev.load(f'{tag}={d},{E}/{tag}_emb.npz,{L}/{tag}.out', P, T, ids, q, conf=True)
                st = ev.stats(r, 'exact')
                z = r['z']
                m = z['iteration'] >= 0
                wm = np.flatnonzero(Tid == r['wid'])
                emb = {o: np.asarray(z[f'emb_{o}'])[m] for o in RD}
                M = measure(emb, [wm] * int(m.sum()))
                for o in RD:
                    assert np.abs(M['sh'][o] - np.asarray(SV[f'cell__{tag}__loop__{o}'])).max() < 1e-12, (tag, o)
                c = cells[tag]
                assert [str(p) for p in c['paths'][1:]] == [str(p) for p in np.asarray(z['paths'])[m]], f'{tag}: run and exact rows differ'
                swid.append((C['ids'][c['start']], str(r['wid'])))
                e = ne(c['E'], c['src'], full)
                if arm == 'C2':
                    k0 = STARTS.index(n) * 1920
                    assert np.allclose(e, eC2[k0:k0 + 1920], rtol=0, atol=1e-6) and np.array_equal(farE(e), farE(eC2[k0:k0 + 1920]))
                cl[tag] = dict(M=M, e=e, kept=np.asarray(r['ev'], bool), DH=np.asarray(st['DH'], bool), paths=np.array([str(p) for p in c['paths'][1:]]))
        say('CHECK of the start works (rows C)')
        want('start work of the measure\'s load_cell (corpus id) = the evaluator\'s start work id, all 15 cells', all(x == y for x, y in swid), True)
        def pooled(tg, ns, key):
            cs = [cl[f'qgc_{tg}_s{k}'] for k in ns]
            M = {kk: {o: np.concatenate([c['M'][kk][o] for c in cs]) for o in RD} for kk in ('nb', 'nbc')}
            M['ov'] = np.concatenate([c['M']['ov'] for c in cs])
            M['far'] = np.concatenate([c['M']['far'] for c in cs])
            sel = np.ones(len(M['ov']), bool) if key is None else np.concatenate([c[key] for c in cs])
            return M, np.concatenate([c['e'] for c in cs]), sel
        stop_if_bad(prefix)
        say('   check of the start works passed.')
        say()
        say('Search runs; the loops are not scored by the leftover or the nearest-work rule. Start work (and its near-duplicates) out. ' + legend)
        for subset, key in (('candidates', None), ('kept by the loop', 'kept'), ('hits', 'DH')):
            say(f' {subset}:')
            for arm, tg, w in ARMS:
                for n in STARTS + ('pooled',):
                    M, e, sel = pooled(tg, STARTS if n == 'pooled' else (n,), key)
                    rowN(f'{arm} {w}, {"s" + str(n) if n != "pooled" else "pooled"}, {subset}', M, e, sel)
        say()
        for tag, c in cl.items():
            keep(tag, c['M'], c['e'], {'kept': c['kept'], 'hits': c['DH'], 'paths': c['paths']})
    say('Neighbour pool of readings 1 and 2: T only (2,048 repaints of all dates). Index of the nearest-work rule: every corpus original (172,171 rows).')
    assert not os.path.exists(prefix + '.npz')
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'saved per-picture arrays: {prefix}.npz ({len(sv)} keys: <group>__e (n, 2) chords DINOv3 / SigLIP, __FAR, __NEAR, __LL, __R1far, __R2plain_both, '
        f'__R2cos_both, paths, and kept / hits for the loops)')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
