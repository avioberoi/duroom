"""Replay of pycma on the 15 cells (replay) and the bar-free readout of the three arms against a given reference and exact files (readout).
Usage: python -B arms_barfree_qgt_v3.py replay OUT.npz --prefix P [--starts ..] | readout OUTPREFIX --prefix P --ref REF.npz --exact PATTERN --replay OUT.npz [--starts ..] [--label TEXT]"""
import argparse
import ast
import csv
import hashlib
import io
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
IDS = os.environ['DUROOM_EMB'] + '/clip-vit-l-14.npz'   # row of the unfiltered observer file -> work id (start_idx indexes it)
STARTS_OF = {'qgc_': (6, 7, 8, 9, 10), 'qgt_': (11, 12, 13, 14, 15)}
ARMS = (('V', 'ra'), ('C1', 'pin'), ('C2', 'c2'))
QS = (10, 25, 50, 75, 90, 99)
LEVELS = (50, 75, 90, 95, 99, 99.9)
GAP_LITERAL = 0.7785
AIM = 'aimv2-huge'
OUT = []
FH = [None]   # OUTPREFIX.txt, written line by line
REPLAY_ASK_TOL = 1e-6   # tolerance for |replayed ask - saved request|


def say(s=''):
    OUT.append(s)
    print(s, flush=True)
    if FH[0] is not None:
        FH[0].write(s + '\n')
        FH[0].flush()


def stop(msg):
    say(f'STOP: {msg}. Nothing after this line was computed.')
    sys.exit(3)


def rundir(tag):
    return f'{O}/runs/{tag}_committee_euclidean_min_seedrow'


def cell_list(prefix, starts):
    return [(arm, n, f'{prefix}{tg}_s{n}') for arm, tg in ARMS for n in starts]


def unit64(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------- replay
def replay_cell(cell, cma, yaml):
    z = np.load(f'{rundir(cell)}/all_candidates.npz', allow_pickle=True)
    fl = ast.literal_eval(str(z['flags']))
    assert fl['tag'] == cell, (fl['tag'], cell)
    assert fl.get('band') is None, 'a --band run: the band term is not replayed'
    cfg = yaml.safe_load(open(fl['config']))['cma']
    sigma0, it_all = float(cfg['sigma0']), z['iteration'].astype(int)
    sig_path = None
    if fl.get('sigma_path'):
        sig_path = np.asarray(np.load(fl['sigma_path'], allow_pickle=True)['log_sigma'], np.float64)
    gthr = float(fl['gate_threshold'])
    x0 = np.asarray(z['style_embs'][0], np.float32)
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {'popsize': fl['popsize'], 'CMA_diagonal': cfg['diagonal'], 'maxiter': fl['maxiter'], 'seed': fl['seed'],
                                                      'tolfun': cfg['tolfun'], 'tolx': cfg['tolx'], 'verbose': -9})
    rng_fit = np.random.default_rng(fl['seed'] + 7919)
    n, pop = fl['maxiter'], fl['popsize']
    prev, cmac, rea = np.zeros((n, 768)), np.zeros((n, 768)), np.zeros((n, 768))
    s_asked, s_after, askdiff, fits_all, used_replay = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, pop)), np.zeros(n, bool)
    stops = []
    for it in range(n):
        st = es.stop()
        if st:
            stops.append(f'{it}:{st}')
        if sig_path is not None and it > 0:
            es.sigma = float(sig_path[it - 1])
        if fl['sigma_cap'] is not None and es.sigma > fl['sigma_cap'] * sigma0:
            es.sigma = fl['sigma_cap'] * sigma0
        s_asked[it] = float(es.sigma)
        prev[it] = np.asarray(es.mean, np.float64)
        X = es.ask()
        rows = np.flatnonzero(it_all == it)
        assert len(rows) == pop and (z['index'][rows] == np.arange(pop)).all()
        Xu = unit64(np.array(X))
        saved = np.asarray(z['style_embs'][rows], np.float64)
        askdiff[it] = float(np.abs(Xu - saved).max())
        used_replay[it] = askdiff[it] < 1e-6
        told = [Xu[i] for i in range(pop)] if used_replay[it] else [saved[i] for i in range(pop)]
        feas = z['feasible'][rows].astype(bool)
        if fl['random_fitness']:
            fits = [float(v) for v in rng_fit.standard_normal(pop)]
        else:
            nov = np.asarray(z['novelties'][rows], np.float64)
            fid = np.asarray(z['fidelity_clip'][rows], np.float64)
            coh = np.asarray(z['coherence'][rows], np.float64)
            ff = [-nov[i] for i in range(pop) if feas[i]]
            off = np.median(ff) if ff else 0.0
            # search_v2.py: f_offset + max(coh_threshold - coh, 0)^2 + band (None here) + max(fidelity - fidelity_max, 0)^2
            fits = [(-nov[i]) if feas[i] else off + max(gthr - coh[i], 0.0) ** 2 + max(fid[i] - fl['fidelity_max'], 0.0) ** 2 for i in range(pop)]
        fits_all[it] = fits
        es.tell(told, fits)
        s_after[it] = float(es.sigma)
        es.mean = es.mean / np.linalg.norm(es.mean)
        cmac[it] = np.asarray(es.mean, np.float64)
        order = sorted(range(pop), key=lambda i: fits[i])
        best = [np.asarray(z['render_clip'][rows[i]], np.float64) for i in order if feas[i]][:fl['reanchor']]
        if best:
            m = np.mean(np.asarray(best, np.float64), axis=0)
            es.mean = m / np.linalg.norm(m)
        rea[it] = np.asarray(es.mean, np.float64)
    la, ls = np.asarray(z['log_sigma_asked'], np.float64), np.asarray(z['log_sigma'], np.float64)
    print(f'{cell}: sigma0 {sigma0}, sigma path {os.path.basename(str(fl.get("sigma_path")))}; replayed asks equal the saved requests (1e-6) in {int(used_replay.sum())} of {n} '
          f'rounds (max abs diff median {np.median(askdiff):.2e}, max {askdiff.max():.2e}); sigma asked vs log_sigma_asked max |diff| {np.abs(s_asked - la).max():.2e}; '
          f'sigma after tell vs log_sigma max |diff| {np.abs(s_after - ls).max():.2e}; es.stop() lines {stops}', flush=True)
    return dict(prev=prev, cma=cmac, rea=rea, s_asked=s_asked, s_after=s_after, askdiff=askdiff, fits=fits_all, used_replay=used_replay)


def replay_main(a, cells):
    out = a.out
    import cma
    import yaml
    assert not os.path.exists(out), out
    res = {'cma_version': np.array(cma.__version__), 'numpy_version': np.array(np.__version__)}
    status = []
    for arm, n, cell in cells:
        try:
            res.update({f'{cell}__{k}': v for k, v in replay_cell(cell, cma, yaml).items()})
            status.append(f'{cell} ok')
        except Exception as e:   # recorded and printed by readout as FAILED for this cell; never silent
            status.append(f'{cell} FAILED {e!r}')
            print(f'{cell}: REPLAY FAILED {e!r}', flush=True)
    res['status'] = np.array(status)
    with open(out, 'xb') as fh:
        np.savez_compressed(fh, **res)
    print('wrote', out)
    if any(not s_.endswith(' ok') for s_ in status):
        print('REPLAY STOP: a cell failed; the readout must not run', flush=True)
        sys.exit(4)


# ---------------------------------------------------------------- readout
def own_scores(E, artist, K, unit, chord):
    """reference's own score for any reader, written as today_rescore.reference does it (asserted equal for DINOv3 and SigLIP)"""
    E = unit(E)
    D = chord(E @ E.T)
    art = np.array([str(x) for x in artist])
    D[art[:, None] == art[None, :]] = np.inf
    return np.sort(D, axis=1)[:, :K].mean(1)


def cand_scores(X, E, K, unit, chord):
    """mean chord to the K nearest reference rows (today_rescore.score's formula, for a reader outside OBS)"""
    return np.sort(chord(unit(X) @ unit(E).T), axis=1)[:, :K].mean(1)


def rank(v, own):
    return 100.0 * np.searchsorted(np.sort(own), v, 'left') / len(own)


def pcts(v, fmt='.4f'):
    return ' | '.join(f'p{q:g} {x:{fmt}}' for q, x in zip(QS, np.percentile(v, QS)))


def beyond(vals, levs):
    return {q: int(np.all([v > lv[q] for v, lv in zip(vals, levs)], axis=0).sum()) for q in LEVELS}


def bline(cnt, n):
    return ' | '.join(f'>p{q:g} {cnt[q]:5d} ({cnt[q] / n:.4f})' for q in LEVELS)


def ang(a, b):
    """angle in degrees between rows (2 arctan2(|a - b|, |a + b|))"""
    a, b = unit64(a), unit64(b)
    return np.degrees(2 * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1)))


def without_rows(ref, keep, OBS):
    if keep.all():
        return ref
    out = dict(ref, id=ref['id'][keep], artist=ref['artist'][keep], year=None if ref['year'] is None else ref['year'][keep])
    for o in OBS:
        out[o] = dict(ref[o], E=ref[o]['E'][keep])
    return out


def validate(cell, z, x, obs_ids, R, a, OBS, unit):
    """validity lines; returns (reasons list (empty = valid), facts dict)"""
    bad, f = [], {}
    it = np.asarray(z['iteration']).astype(np.int64)
    f['rows'] = len(it)
    if len(it) != 1921 or it[0] >= 0 or not (it[1:] >= 0).all() or it.max() != 59 or not (np.bincount(it[1:], minlength=60) == 32).all():
        bad.append(f'rows {len(it)}, start row first {bool(it[0] < 0)}, not 60 x 32 after the start')
        return bad, f
    if not np.all(np.diff(it * 1000 + np.asarray(z['index']).astype(np.int64)) > 0):
        bad.append('rows not in loop order')
    need_z = (['flags', 'start_idx', 'style_embs', 'render_clip', 'feasible', 'admitted', 'passes_q99', 'log_sigma_asked', 'log_sigma', 'paths', 'index', 'coherence',
               'fidelity_clip'] + [f'emb_{o}' for o in OBS] + [f'norm_q99_{o}' for o in OBS])
    need_x = ['paths', 'emb_clip-vit-l-14', f'emb_{AIM}'] + [f'emb_{o}' for o in OBS]
    miss = [k for k in need_z if k not in z.files] + [f'exact:{k}' for k in need_x if k not in x.files]
    if miss:
        bad.append(f'missing keys {miss}')
        return bad, f
    if len(z['paths']) != len(x['paths']) or not all(str(p) == str(q) for p, q in zip(z['paths'], x['paths'])):
        bad.append('run file and exact file paths differ')
        return bad, f
    fl = ast.literal_eval(str(z['flags']))
    f['fl'] = fl
    if fl.get('tag') != cell:
        bad.append(f'flags tag {fl.get("tag")!r} is not {cell}')
    wid = int(obs_ids[int(z['start_idx'])])
    pool = np.load(fl['start_pool'], allow_pickle=True)
    j = np.flatnonzero(pool['rows_full'] == int(z['start_idx']))
    if len(j) != 1 or int(pool['ids'][j[0]]) != wid:
        bad.append(f'start_idx {int(z["start_idx"])} is not one row of {os.path.basename(fl["start_pool"])} with id {wid}')
    f['wid'] = wid
    m = it >= 0
    cs = {k: (unit(z[ka][m]) * unit(x[kb][m])).sum(1) for k, ka, kb in (('DINOv3', 'emb_dinov3-l', 'emb_dinov3-l'), ('SigLIP', 'emb_siglip-so400m', 'emb_siglip-so400m'),
                                                                     ('CLIP render_clip vs exact', 'render_clip', 'emb_clip-vit-l-14'))}
    f['cos'] = cs
    inref = R['id'] == wid
    f['inref'] = int(inref.sum())
    f['hash'] = hashlib.md5(z['emb_dinov3-l'][0].astype(np.float32).tobytes()).hexdigest()
    return bad, f


def readout_main(a, cells, starts):
    prefix = a.outprefix
    for ext in ('.txt', '.npz', '_rounds.csv'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x')
    sys.path.insert(0, C)
    from local_rescore import FID, K, OBS, auc, chord, unit
    from today_rescore import far, pid, reference, score
    assert auc(np.array([2.0]), np.array([1.0])) == 1.0 and auc(np.array([1.0]), np.array([2.0])) == 0.0 and auc(np.array([1.0]), np.array([1.0])) == 0.5
    refpath = a.ref if os.path.isabs(a.ref) else f'{O}/{a.ref}'
    refbase = os.path.basename(refpath)
    Rz = np.load(refpath, allow_pickle=True)
    R = reference(Rz)
    R['id'] = pid(Rz['paths'])
    assert len(set(R['id'].tolist())) == len(R['id'])
    own = {o: R[o]['own'] for o in OBS}
    for o in OBS:
        assert np.allclose(own_scores(Rz[f'emb_{o}'], Rz['artist'], K, unit, chord), own[o], rtol=0, atol=1e-12), o
    own_aim = own_scores(Rz[f'emb_{AIM}'], Rz['artist'], K, unit, chord)
    lev = {o: {qq: float(np.percentile(own[o], qq)) for qq in LEVELS} for o in OBS}
    lev_aim = {qq: float(np.percentile(own_aim, qq)) for qq in LEVELS}
    for o in OBS:
        assert abs(lev[o][99] - R[o]['bar']) < 1e-12
    gref = chord((unit(Rz['vec']) * unit(Rz['emb_clip-vit-l-14'])).sum(1))
    g99 = float(np.percentile(gref, 99))
    Pz0 = np.load(f'{O}/push_emb.npz', allow_pickle=True)   # FID's own file: the 1900 reference's p99 gap
    fid_chk = float(np.percentile(chord((unit(Pz0['vec']) * unit(Pz0['emb_clip-vit-l-14'])).sum(1)), 99))
    obs_ids = np.load(IDS, allow_pickle=True)['ids'].astype(np.int64)
    S = np.load(a.replay, allow_pickle=True)
    rstat = {s.split(' ')[0]: s for s in S['status'].tolist()}

    say(f'arms_barfree_qgt_v3.py; replay {a.replay} (pycma {S["cma_version"]})')
    if a.label:
        say(f'LABEL: {a.label}')
    say(f'Cells: prefix {a.prefix}, starts {list(starts)}; V = {a.prefix}ra_s*, C1 = {a.prefix}pin_s*, C2 = {a.prefix}c2_s*; exact files O/eval/{a.exact}; 1,920 candidates per cell (start row dropped).')
    say(f'Reference {refbase}, {len(R["id"])} rows; own score = today_rescore.reference (leave-one-out, artist rows out); K = {K}. '
        f'A start work that is a row of it is left out (that row only, as controls_evaluate.without()); levels stay the full reference\'s.')
    say(f'READINGS: items 1, 2, 3 and TABLE 1e score the RUN FILE\'s own DINOv3 and SigLIP readings; the validity lines show their row cosine '
        f'with the exact file\'s; TABLE 1x gives the p99-in-both count on the EXACT file\'s readings. Item 4 (gap) and item 7 (AIMv2) use the exact file.')
    say(f'SOURCES: DINOv3 / SigLIP levels, bars and ranks from {refbase} (emb_dinov3-l, emb_siglip-so400m, artist); AIMv2 levels from {refbase} emb_{AIM}; '
        f'FID from local_rescore.FID, p99 of push_emb.npz\'s own gap ({fid_chk:.6f}); this reference\'s own gap from {refbase} vec and emb_clip-vit-l-14.')
    say('Levels of the reference (np.percentile of its own score): ' + '; '.join(f'{o}: ' + ', '.join(f'p{qq:g} {lev[o][qq]:.4f}' for qq in LEVELS) for o in OBS)
        + f'; bars at full precision ' + ', '.join(f'{o} {R[o]["bar"]:.6f}' for o in OBS))
    say(f'Gap lines: FID {FID:.6f} (push_emb.npz\'s own p99 gap, the faithful line); literal {GAP_LITERAL}; this reference\'s own p99 gap {g99:.6f}')
    say()

    say('VALIDITY per cell (any failure below stops the readout before any table)')
    cells_ok, Z, X, FACT = [], {}, {}, {}
    for arm, n, cell in cells:
        z = np.load(f'{rundir(cell)}/all_candidates.npz', allow_pickle=True)
        x = np.load(f'{O}/eval/{a.exact.replace("{tag}", cell)}', allow_pickle=True)
        bad, f = validate(cell, z, x, obs_ids, R, a, OBS, unit)
        if bad:
            say(f'   INVALID {cell}: ' + '; '.join(bad))
            continue
        fl, cs = f['fl'], f['cos']
        lb = {o: float(z[f'norm_q99_{o}']) for o in OBS}
        same_ref = os.path.basename(str(fl.get('pushforward'))) == refbase
        cmp_ = (('stored bars vs this reference\'s: ' + ', '.join(f'{o} |diff| {abs(lb[o] - R[o]["bar"]):.2e}' for o in lb)) if same_ref else
                'stored bars not compared (the loop\'s reference is another file)')
        say(f'   {cell}: rows {f["rows"]} (start row first, 60 x 32, loop order) | paths equal run/exact | start idx {int(z["start_idx"])} work {f["wid"]} '
            f'(start work a row of the reference: {"yes" if f["inref"] else "no"}; rows of the reference left out for this cell: {f["inref"]}) | start DINOv3 md5 {f["hash"]} | '
            + ' | '.join(f'{k} row cosine min {v.min():.6f} median {np.median(v):.6f}' for k, v in cs.items())
            + f' | CLIP under 0.999: {int((cs["CLIP render_clip vs exact"] < 0.999).sum())}')
        say(f'      flags: seed {fl.get("seed")}, fidelity_max {fl.get("fidelity_max")}, gate_threshold {fl.get("gate_threshold")}, random_fitness {fl.get("random_fitness")}, '
            f'sigma_path {os.path.basename(str(fl.get("sigma_path")))}, pushforward {os.path.basename(str(fl.get("pushforward")))}, start_pool {os.path.basename(str(fl.get("start_pool")))}; '
            f'loop\'s stored bars ' + ', '.join(f'{o} {v:.6f}' for o, v in lb.items()) + f' ({cmp_}) | replay: {rstat.get(cell, "absent")}')
        cells_ok.append((arm, n, cell))
        Z[cell], X[cell], FACT[cell] = z, x, f
    say(f'valid cells: {len(cells_ok)} of {len(cells)}')
    if len(cells_ok) != len(cells):
        stop('a cell is INVALID (line above)')
    hard = []
    tg_ = dict(ARMS)
    for n in starts:
        tri = {arm: f'{a.prefix}{tg_[arm]}_s{n}' for arm, _ in ARMS}
        fls = {arm: FACT[c]['fl'] for arm, c in tri.items()}
        for arm, c in tri.items():
            fl = fls[arm]
            if int(fl['seed']) != n:
                hard.append(f'{c}: stored seed {fl["seed"]} is not the tag number {n}')
            if (int(fl['maxiter']), int(fl['popsize']), int(fl['reanchor'])) != (60, 32, 4):
                hard.append(f'{c}: maxiter / popsize / reanchor {fl["maxiter"]} / {fl["popsize"]} / {fl["reanchor"]}, not 60 / 32 / 4')
            lref = os.path.basename(str(fl.get('pushforward')))
            if lref != refbase:
                hard.append(f'{c}: the loop reference {lref} is not the given {refbase}')
            else:
                for o in OBS:
                    if abs(float(Z[c][f'norm_q99_{o}']) - R[o]['bar']) >= 1e-6:
                        hard.append(f'{c}: stored null q99 {o} {float(Z[c][f"norm_q99_{o}"]):.6f} differs from the reference bar {R[o]["bar"]:.6f}')
            st = rstat.get(c, 'absent')
            if not st.endswith(' ok'):
                hard.append(f'{c}: replay {st}')
            else:
                ad = float(S[f'{c}__askdiff'].max())
                d1 = float(np.abs(S[f'{c}__s_asked'] - np.asarray(Z[c]['log_sigma_asked'], float)).max())
                d2 = float(np.abs(S[f'{c}__s_after'] - np.asarray(Z[c]['log_sigma'], float)).max())
                if not (bool(S[f'{c}__used_replay'].all()) and ad < REPLAY_ASK_TOL and d1 < 1e-12 and d2 < 1e-9):
                    hard.append(f'{c}: replay not exact (asks equal {int(S[f"{c}__used_replay"].sum())}/60, max ask diff {ad:.1e} (tolerance {REPLAY_ASK_TOL:g}), '
                                f'sigma asked {d1:.1e}, after tell {d2:.1e})')
        if fls['V'].get('random_fitness') or fls['V'].get('sigma_path'):
            hard.append(f'{tri["V"]}: the search carries random_fitness or a sigma_path')
        vdir = rundir(tri['V']) + '/'
        if not fls['C1'].get('random_fitness') or os.path.basename(os.path.dirname(str(fls['C1'].get('sigma_path') or ''))) != os.path.basename(vdir.rstrip('/')):
            hard.append(f'{tri["C1"]}: random_fitness {fls["C1"].get("random_fitness")}, sigma_path {fls["C1"].get("sigma_path")} (must be random-score and name {vdir})')
        sa2 = np.asarray(Z[tri['C2']]['log_sigma_asked'], np.float64)
        if not fls['C2'].get('random_fitness') or os.path.basename(str(fls['C2'].get('sigma_path'))) != 'c2_sigma_1e-5.npz' or not (np.abs(sa2 - 1e-5) < 1e-15).all():
            hard.append(f'{tri["C2"]}: random_fitness {fls["C2"].get("random_fitness")}, sigma_path {fls["C2"].get("sigma_path")}, asked steps {np.unique(sa2).tolist()[:3]} '
                        f'(must be random-score at 1e-5)')
        sv, s1 = (np.asarray(Z[tri[k]]['log_sigma_asked'], np.float64) for k in ('V', 'C1'))
        if not np.array_equal(sv, s1):
            hard.append(f's{n}: the random-score loop asked step sizes are not the search ones exactly')
        if (len({int(Z[c]['start_idx']) for c in tri.values()}) != 1 or len({FACT[c]['hash'] for c in tri.values()}) != 1
                or len({float(fl['fidelity_max']) for fl in fls.values()}) != 1):
            hard.append(f's{n}: the three arms do not share one start row, start picture and fidelity_max')
    for h in hard:
        say(f'   HARD CHECK FAILED: {h}')
    if hard:
        stop(f'{len(hard)} hard checks failed')
    say('hard checks passed: arms match their roles (search: no random score, no step path; random-score loop: random score, step path in the search run of the same seed, '
        'asked steps equal the search ones exactly; no-step loop: random score at 1e-5), seed = tag number, 60 x 32, re-anchor 4, one start row, start picture and fidelity_max per start, '
        + 'the loop reference = the given file and its stored null q99 = the reference bars (1e-6), '
        + f'replay exact in every cell (asks within {REPLAY_ASK_TOL:g}, sigmas to 1e-12 / 1e-9), every expected array present')
    say()

    cellv = {}
    for arm, n, cell in cells_ok:
        z, x, f = Z[cell], X[cell], FACT[cell]
        m = z['iteration'] >= 0
        inref = R['id'] == f['wid']
        keep = ~inref                                  # the start work's own row only (controls_evaluate.without(); no artist rule)
        Rr = without_rows(R, keep, OBS)
        sR = score(Rr, {f'emb_{o}': z[f'emb_{o}'][m] for o in OBS})
        d = {o: sR[o][0] for o in OBS}
        F = far(sR, Rr)
        F_exact = far(score(Rr, {f'emb_{o}': x[f'emb_{o}'][m] for o in OBS}), Rr)   # side line: the same rule on the exact file's readings
        rk = {o: rank(d[o], own[o]) for o in OBS}
        gap = chord((unit(z['style_embs']) * unit(x['emb_clip-vit-l-14'])).sum(1))[m]
        aimE = np.asarray(Rz[f'emb_{AIM}'])[keep]   # raw rows (cand_scores makes them unit once)
        c = dict(arm=arm, start=n, it=z['iteration'][m].astype(int), d_dino=d[OBS[0]], d_siglip=d[OBS[1]], rank_dino=rk[OBS[0]], rank_siglip=rk[OBS[1]],
                 minrank=np.minimum(rk[OBS[0]], rk[OBS[1]]), gap=gap, infeasible=~z['feasible'][m].astype(bool), F=F,
                 aim=cand_scores(x[f'emb_{AIM}'][m], aimE, K, unit, chord), admitted=z['admitted'][m].astype(bool),
                 passes_q99=z['passes_q99'][m].astype(bool), dropped=int((~keep).sum()), F_exact=F_exact)
        assert beyond([c['d_dino'], c['d_siglip']], [lev[o] for o in OBS])[99] == int(F.sum()), cell   # the p99 column = today_rescore.far
        cellv[cell] = c
    say('rows of the reference left out per cell (the start work\'s own row, only when it is a row of the reference; 0 or 1): '
        + ', '.join(f'{cell} {c["dropped"]}' for cell, c in cellv.items()))
    say()
    arms = [x_ for x_, _ in ARMS]
    tg = dict(ARMS)

    def group(arm):
        return [cellv[f'{a.prefix}{tg[arm]}_s{n}'] for n in starts if f'{a.prefix}{tg[arm]}_s{n}' in cellv]

    def cat(arm, key):
        g = group(arm)
        return np.concatenate([c[key] for c in g]) if g else np.zeros(0)
    rname = 'reference ' + refbase

    for key, title, fmt in (('d_dino', f'far score DINOv3 (mean chord to the 15 nearest reference pictures)', '.4f'),
                            ('d_siglip', f'far score SigLIP (mean chord to the 15 nearest reference pictures)', '.4f'),
                            ('rank_dino', 'rank in the reference\'s own DINOv3 distribution (100 x share of reference works with a smaller own score)', '.1f'),
                            ('rank_siglip', 'rank in the reference\'s own SigLIP distribution', '.1f'),
                            ('minrank', 'smaller rank of the two observers', '.1f')):
        say(f'TABLE 1 {title}; reader {"DINOv3" if "dino" in key else "SigLIP" if "siglip" in key else "DINOv3 and SigLIP"}; {rname}; all candidates')
        for arm in arms:
            for c in group(arm):
                say(f'   {arm:2s} s{c["start"]:<2d} n {len(c[key]):5d} | {pcts(c[key], fmt)}')
            if group(arm):
                say(f'   {arm:2s} pool n {len(cat(arm, key)):5d} | {pcts(cat(arm, key), fmt)}')
        say()
    say(f'TABLE 1e candidates beyond each level of the reference\'s own score in BOTH DINOv3 and SigLIP (count, share); {rname}; the p99 column = today_rescore.far')
    for arm in arms:
        for c in group(arm):
            say(f'   {arm:2s} s{c["start"]:<2d} n 1920 | {bline(beyond([c["d_dino"], c["d_siglip"]], [lev[o] for o in OBS]), 1920)}')
        if group(arm):
            nn = len(cat(arm, 'd_dino'))
            say(f'   {arm:2s} pool n {nn} | {bline(beyond([cat(arm, "d_dino"), cat(arm, "d_siglip")], [lev[o] for o in OBS]), nn)}')
    say()

    say(f'TABLE 1x side line: candidates beyond the reference\'s 99th percentile in BOTH observers, on the run file\'s readings (= TABLE 1e p99) and on the EXACT file\'s readings; {rname}')
    for arm in arms:
        for c in group(arm):
            say(f'   {arm:2s} s{c["start"]:<2d} n 1920 | run-file readings {int(c["F"].sum()):5d} | exact-file readings {int(c["F_exact"].sum()):5d}')
        if group(arm):
            say(f'   {arm:2s} pool n {len(cat(arm, "F"))} | run-file readings {int(cat(arm, "F").sum()):5d} | exact-file readings {int(cat(arm, "F_exact").sum()):5d}')
    say()

    aucs = {}
    say(f'TABLE 2 pair number per start: P(a random V candidate scores above a random control candidate of the same start), ties half; 0.5 = no difference; {rname}')
    for key, lab in (('d_dino', 'far score DINOv3'), ('d_siglip', 'far score SigLIP'), ('minrank', 'smaller rank')):
        for xa in ('C1', 'C2'):
            ss = [n for n in starts if f'{a.prefix}ra_s{n}' in cellv and f'{a.prefix}{tg[xa]}_s{n}' in cellv]
            v = np.array([auc(cellv[f'{a.prefix}ra_s{n}'][key], cellv[f'{a.prefix}{tg[xa]}_s{n}'][key]) for n in ss])
            aucs[f'{key}__V_vs_{xa}'] = v
            say(f'   {lab:17s} V vs {xa}: ' + ', '.join(f's{n} {a_:.4f}' for n, a_ in zip(ss, v)) + (f' | mean {v.mean():.4f}' if len(v) else ' | no valid pair'))
    say()

    keys3 = ('d_dino', 'd_siglip', 'minrank')
    series = np.full((3, len(starts), 60, 3), np.nan)
    for i, arm in enumerate(arms):
        for j, n in enumerate(starts):
            cell = f'{a.prefix}{tg[arm]}_s{n}'
            if cell not in cellv:
                continue
            c = cellv[cell]
            for rr in range(60):
                k_ = c['it'] == rr
                assert k_.sum() == 32
                series[i, j, rr] = [np.median(c[kk][k_]) for kk in keys3]
    mean_series = series.mean(1) if not np.isnan(series).any() else np.nanmean(series, 1)
    say(f'TABLE 3 median per round (32 candidates) of each score, averaged over the valid starts; {rname}. Columns: V, C1, C2 for DINOv3 far score | SigLIP far score | smaller rank')
    for rr in range(60):
        say(f'   round {rr:2d} | ' + ' | '.join(' '.join(f'{mean_series[i, rr, k]:{".4f" if k < 2 else "6.1f"}}' for i in range(3)) for k in range(3)))
    say()

    say(f'TABLE 4 request-to-reading gap = chord(request, exact CLIP reading); reader CLIP (exact file). Counted above FID {FID:.6f}, above the literal {GAP_LITERAL}, '
        f'above this reference\'s own p99 gap {g99:.6f}; infeasible = the loop\'s own flag (~feasible: coherence under its gate or fidelity_clip above fidelity_max)')
    for arm in arms:
        rows4 = [(f's{c["start"]:<2d}', c['gap'], c['infeasible']) for c in group(arm)] + ([('pool', cat(arm, 'gap'), cat(arm, 'infeasible'))] if group(arm) else [])
        for lab, g, inf in rows4:
            say(f'   {arm:2s} {lab} n {len(g):4d} | {pcts(g)} | above FID {FID:.6f}: {int((g > FID).sum()):4d} | above {GAP_LITERAL}: {int((g > GAP_LITERAL).sum()):4d} | '
                f'above ref p99 {g99:.6f}: {int((g > g99).sum()):4d} | infeasible (loop) {int(inf.sum()):4d}')
    say()

    ok5 = [(arm, n, cell) for arm, n, cell in cells_ok if rstat.get(cell, '').endswith(' ok')]
    for arm, n, cell in cells_ok:
        if (arm, n, cell) not in ok5:
            say(f'ITEM 5 FAILED for {cell}: replay status {rstat.get(cell, "absent")}; left out of item 5 and of the centre columns of item 8')
    say('TABLE 5.0 (check of the pycma replay): per cell, rounds whose replayed asks equal the saved requests (1e-6); max |replayed sigma asked - log_sigma_asked|; max |replayed sigma after tell - log_sigma|')
    exact_replay = True
    for arm, n, cell in ok5:
        z = Z[cell]
        ur = S[f'{cell}__used_replay']
        d1 = np.abs(S[f'{cell}__s_asked'] - np.asarray(z['log_sigma_asked'], float)).max()
        d2 = np.abs(S[f'{cell}__s_after'] - np.asarray(z['log_sigma'], float)).max()
        exact_replay &= bool(ur.all()) and d1 < 1e-12 and d2 < 1e-9
        say(f'   {arm:2s} s{n:<2d} asks equal {int(ur.sum())}/60 (max abs diff {S[f"{cell}__askdiff"].max():.1e}) | sigma asked {d1:.1e} | sigma after tell {d2:.1e}')
    say(f'   replay exact in every cell: {exact_replay}')
    say()
    say('TABLE 5.1 step sizes asked (log_sigma_asked, run files): C1 against V of the same start over all 60 rounds; C2\'s values')
    for n in starts:
        tri = [f'{a.prefix}{t}_s{n}' for t in ('ra', 'pin', 'c2')]
        if not all(t in Z for t in tri):
            say(f'   s{n:<2d} not shown: a cell of this start is invalid')
            continue
        v, c1, c2 = (np.asarray(Z[t]['log_sigma_asked'], np.float64) for t in tri)
        eq = len(v) == len(c1) and bool(np.array_equal(v, c1))
        say(f'   s{n:<2d} C1 == V exactly: {eq} (rounds {len(v)}, {len(c1)}; max |diff| {np.abs(v - c1).max() if len(v) == len(c1) else float("nan"):.3e}; '
            f'rounds differing {int((v != c1).sum()) if len(v) == len(c1) else -1}) | V asked min {v.min():.5f} median {np.median(v):.5f} max {v.max():.5f} | C2 distinct values {np.unique(c2).tolist()}')
    say()
    say('TABLE 5.3 what moves the centre (replay; centre = the unit mean pycma samples around). Per round: (a) angle from pycma\'s own centre after tell (made unit, as the loop does '
        'before the re-anchor) to the previous centre; (b) angle from the re-anchored centre (unit mean of render_clip of the 4 feasible pictures lowest in told score) to the previous '
        'centre; (a\') unit mean of the best 16 told requests to the previous centre; (c) angle between (a) and (b). Medians over the 60 rounds. C1 and C2 told scores are random draws.')
    drift = np.full((3, len(starts), 61), np.nan)
    res5 = {}
    for ia, (arm, tga) in enumerate(ARMS):
        pa, pb, pf, pc = [], [], [], []
        for j, n in enumerate(starts):
            cell = f'{a.prefix}{tga}_s{n}'
            if (arm, n, cell) not in ok5:
                continue
            z = Z[cell]
            prev, cm, rea, fits = S[f'{cell}__prev'], S[f'{cell}__cma'], S[f'{cell}__rea'], S[f'{cell}__fits']
            it = z['iteration'].astype(int)
            fb = np.array([unit64(np.asarray(z['style_embs'][np.flatnonzero(it == r)], np.float64)[np.argsort(fits[r], kind='stable')[:16]].mean(0)) for r in range(60)])
            a_, b_, f_, c_ = ang(cm, prev), ang(rea, prev), ang(fb, prev), ang(cm, rea)
            pa.append(a_); pb.append(b_); pf.append(f_); pc.append(c_)
            x0 = np.asarray(z['style_embs'][0], np.float64)
            drift[ia, j] = ang(np.vstack([prev, rea[-1:]]), np.repeat(x0[None], 61, 0))
            res5[f'{cell}__angle_cma'], res5[f'{cell}__angle_reanchor'], res5[f'{cell}__angle_fallback16'] = a_, b_, f_
            if arm != 'C2':
                say(f'   {arm:2s} s{n:<2d} (a) {np.median(a_):7.3f} | (b) {np.median(b_):7.3f} | (a\') {np.median(f_):7.3f} | (c) {np.median(c_):7.3f}')
        if not pa:
            say(f'   {arm} no valid replay')
        elif arm != 'C2':
            say(f'   {arm:2s} pool (a) {np.median(np.concatenate(pa)):7.3f} | (b) {np.median(np.concatenate(pb)):7.3f} | (a\') {np.median(np.concatenate(pf)):7.3f} | (c) {np.median(np.concatenate(pc)):7.3f}')
        else:
            say(f'   C2 (for reference, sigma 1e-5) pool (a) {np.median(np.concatenate(pa)):7.4f} | (b) {np.median(np.concatenate(pb)):7.3f} | (a\') {np.median(np.concatenate(pf)):7.4f} | (c) {np.median(np.concatenate(pc)):7.3f}')
    say('TABLE 5.3 drift: angle (degrees) of each round\'s sampling centre from the start (row 0 request = x0); index 60 = the centre after the last round. Median over the valid starts [min, max]')
    for r in list(range(0, 60, 5)) + [59, 60]:
        say(f'   round {r:2d} | ' + ' | '.join(f'{arm} {np.nanmedian(drift[i, :, r]):6.2f} [{np.nanmin(drift[i, :, r]):6.2f}, {np.nanmax(drift[i, :, r]):6.2f}]'
                                            if not np.isnan(drift[i, :, r]).all() else f'{arm} n/a' for i, (arm, _) in enumerate(ARMS)))
    say()

    say('TABLE 6 the loop\'s own bookkeeping (run file): kept = admitted; passes_q99 = the loop\'s own flag "raw score over its stored bar in both observers", raw score against the '
        'loop\'s index (its --pushforward reference plus its archive), NOT item 1\'s score against the reference alone')
    for arm in arms:
        tot = np.zeros(2, int)
        for c in group(arm):
            pq = int(c['passes_q99'].sum())
            tot += [int(c['admitted'].sum()), pq]
            say(f'   {arm:2s} s{c["start"]:<2d} kept {int(c["admitted"].sum()):4d} | passes_q99 (loop) {pq}')
        if group(arm):
            say(f'   {arm:2s} pool kept {tot[0]:4d} | passes_q99 (loop) {tot[1]}')
    say()

    say(f'TABLE 7 held-out reader {AIM} (reading of the picture): {refbase} emb_{AIM} (repaint readings) against the candidates\' exact readings (exact file emb_{AIM}); '
        f'same score (mean chord to 15 nearest; the reference\'s own leave-one-out, artist out); share beyond each level of the reference\'s own score in this reader alone')
    say('   levels: ' + ', '.join(f'p{qq:g} {lev_aim[qq]:.4f}' for qq in LEVELS))
    for arm in arms:
        for c in group(arm):
            say(f'   {arm:2s} s{c["start"]:<2d} n 1920 | {bline(beyond([c["aim"]], [lev_aim]), 1920)}')
        if group(arm):
            nn = len(cat(arm, 'aim'))
            say(f'   {arm:2s} pool n {nn} | {bline(beyond([cat(arm, "aim")], [lev_aim]), nn)}')
    say()

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['cell', 'arm', 'start', 'round', 'sigma_asked', 'centre_angle_from_start_deg', 'move_after_this_round_deg', 'move_after_previous_round_deg', 'n_infeasible',
                'median_far_dino', 'median_far_siglip', 'median_minrank'])
    nrow = 0
    for ia, (arm, tga) in enumerate(ARMS):
        for j, n in enumerate(starts):
            cell = f'{a.prefix}{tga}_s{n}'
            if cell not in cellv:
                continue
            z = Z[cell]
            sa = np.asarray(z['log_sigma_asked'], float)
            have = (arm, n, cell) in ok5
            if have:
                cen = np.vstack([S[f'{cell}__prev'], S[f'{cell}__rea'][-1:]])
                uu = unit64(cen)
                move = np.degrees(2 * np.arctan2(np.linalg.norm(uu[1:] - uu[:-1], axis=1), np.linalg.norm(uu[1:] + uu[:-1], axis=1)))
            inf = np.bincount(cellv[cell]['it'][cellv[cell]['infeasible']], minlength=60)
            ser = series[ia, j]
            for rr in range(60):
                w.writerow([cell, arm, f's{n}', rr, f'{sa[rr]:.6g}', f'{drift[ia, j, rr]:.6f}' if have else '', f'{move[rr]:.6f}' if have else '',
                            '' if rr == 0 or not have else f'{move[rr - 1]:.6f}', int(inf[rr]), f'{ser[rr, 0]:.6f}', f'{ser[rr, 1]:.6f}', f'{ser[rr, 2]:.2f}'])
                nrow += 1
    say(f'ITEM 8 rounds CSV: {prefix}_rounds.csv ({nrow} rows = valid cells x 60 rounds); sigma_asked = run file log_sigma_asked; '
        'centre angles from the replay; n_infeasible = run file feasible flag per round; medians = TABLE 3 per cell (median of the 32, this reference).')

    arrs = {'series_median_per_round': series, 'starts': np.array(starts), 'FID': np.array(FID), 'ref_gap_p99': np.array(g99), 'drift_from_start': drift}
    for o in OBS:
        arrs[f'ref_own__{o}'], arrs[f'ref_bar__{o}'] = own[o], np.array(R[o]['bar'])
    arrs[f'ref_own__{AIM}'] = own_aim
    for k_, v in aucs.items():
        arrs[f'auc__{k_}'] = v
    arrs.update(res5)
    for cell, c in cellv.items():
        for kk in ('it', 'd_dino', 'd_siglip', 'rank_dino', 'rank_siglip', 'minrank', 'gap', 'infeasible', 'F', 'F_exact', 'aim', 'admitted', 'passes_q99'):
            arrs[f'{cell}__{kk}'] = c[kk]
    with open(prefix + '.npz', 'xb') as fh:
        np.savez_compressed(fh, **arrs)
    open(prefix + '_rounds.csv', 'x').write(buf.getvalue())
    say(f'wrote {prefix}.txt, {prefix}.npz, {prefix}_rounds.csv')
    FH[0].close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['replay', 'readout'])
    p.add_argument('out', help='replay: OUT.npz; readout: OUTPREFIX')
    p.add_argument('--prefix', required=True)
    p.add_argument('--starts', type=int, nargs='+')
    p.add_argument('--ref')
    p.add_argument('--exact')
    p.add_argument('--replay')
    p.add_argument('--label', default='')
    a = p.parse_args()
    starts = tuple(a.starts) if a.starts else STARTS_OF.get(a.prefix)
    assert starts, f'no default starts for prefix {a.prefix!r}: give --starts'
    cells = cell_list(a.prefix, starts)
    if a.mode == 'replay':
        return replay_main(a, cells)
    assert a.ref and a.exact and a.replay, 'readout needs --ref, --exact and --replay'
    assert '{tag}' in a.exact, 'the exact-file pattern must contain {tag}'
    a.outprefix = a.out
    return readout_main(a, cells, starts)


if __name__ == '__main__':
    main()
