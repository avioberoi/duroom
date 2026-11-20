"""Loop facts on the 15 main runs: pycma replay checks, step sizes, readers, what moves the centre, drift, no-step runs, feasibility gate and replay without it, coordinates 644 and 413, file checks. Counts and distributions only.
Usage (CPU): python -B lf_main_v2.py OUTPREFIX REPLAY.npz (the lf_cma_v2.py output) -> OUTPREFIX.txt and OUTPREFIX.npz, each refused if it exists."""
import importlib.util
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
EV_FILE = f'{C}/controls_evaluate.py'
ARMS_NPZ = f'{B}/tests12/arms/arms_barfree_v1.npz'
ARMS = (('V', 'ra'), ('C1', 'pin'), ('C2', 'c2'))
STARTS = (6, 7, 8, 9, 10)
ODD = (644, 413)
OUT = []


def say(s=''):
    OUT.append(s)


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ang(a, b):
    """angle in degrees between rows of a and b (unit-normalised first): 2 arctan2(|a - b|, |a + b|), exact near 0"""
    a, b = unit(a), unit(b)
    return np.degrees(2 * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1)))


def chord(c):
    return np.sqrt(np.clip(2 - 2 * np.asarray(c, np.float64), 0, None))


def pairwise_median(E):
    U = unit(E)
    G = U @ U.T
    return float(np.median(chord(G[np.triu_indices(len(U), 1)])))


def main(prefix, stepa):
    assert not os.path.exists(prefix + '.txt') and not os.path.exists(prefix + '.npz')
    spec = importlib.util.spec_from_file_location('controls_evaluate', EV_FILE)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    P, T, Pz = ev.references(O)
    A, S = np.load(ARMS_NPZ), np.load(stepa)
    FID = float(ev.FID)
    say(f'lf_main_v2.py; replay {stepa} (pycma {S["cma_version"]}, numpy {S["numpy_version"]})')
    say('Runs: the 15 main runs. V = qgc_ra_s6..10 (search), C1 = qgc_pin_s6..10 (random score, V\'s step sizes), C2 = qgc_c2_s6..10 (random score, no step). Counts and distributions only.')
    say()
    Z, X = {}, {}
    for arm, tg in ARMS:
        for n in STARTS:
            cell = f'qgc_{tg}_s{n}'
            Z[cell] = np.load(f'{O}/runs/{cell}_committee_euclidean_min_seedrow/all_candidates.npz', allow_pickle=True)
            X[cell] = np.load(f'{O}/eval/{cell}_emb.npz', allow_pickle=True)
            assert all(str(a) == str(b) for a, b in zip(Z[cell]['paths'], X[cell]['paths']))
    cells = [(arm, n, f'qgc_{tg}_s{n}') for arm, tg in ARMS for n in STARTS]
    res = {}

    say('ITEM 0 (check of the pycma replay): per cell, rounds whose replayed asks equal the saved requests (1e-6); max |replayed sigma asked - log_sigma_asked|; max |replayed sigma after tell - log_sigma|')
    exact_replay = True
    for arm, n, cell in cells:
        z = Z[cell]
        ur = S[f'{cell}__used_replay']
        d1 = np.abs(S[f'{cell}__s_asked'] - np.asarray(z['log_sigma_asked'], float)).max()
        d2 = np.abs(S[f'{cell}__s_after'] - np.asarray(z['log_sigma'], float)).max()
        exact_replay &= bool(ur.all()) and d1 < 1e-12 and d2 < 1e-9
        say(f'   {arm:2s} s{n:<2d} asks equal {int(ur.sum())}/60 (max abs diff {S[f"{cell}__askdiff"].max():.1e}) | sigma asked {d1:.1e} | sigma after tell {d2:.1e}')
    say(f'   replay exact in every cell: {exact_replay}')
    say()

    say('ITEM 1 step sizes asked (log_sigma_asked, run files): C1 against V of the same seed over all 60 rounds; C2\'s values')
    for n in STARTS:
        v, c1, c2 = (np.asarray(Z[f'qgc_{t}_s{n}']['log_sigma_asked'], np.float64) for t in ('ra', 'pin', 'c2'))
        say(f'   s{n:<2d} C1 == V exactly: {bool(np.array_equal(v, c1))} (rounds {len(v)}, {len(c1)}; max |diff| {np.abs(v - c1).max():.3e}; rounds differing {int((v != c1).sum())}) | '
            f'V asked min {v.min():.5f} median {np.median(v):.5f} max {v.max():.5f} | C2 distinct values {np.unique(c2).tolist()}')
    say()

    say('ITEM 2 readers: row cosine (rows 1..1920) between the run file\'s own reading and the exact file\'s; min | median')
    for arm, n, cell in cells:
        z, x = Z[cell], X[cell]
        m = z['iteration'] >= 0
        cs = {k: (unit(z[a][m]) * unit(x[b][m])).sum(1) for k, a, b in (('DINOv3', 'emb_dinov3-l', 'emb_dinov3-l'), ('SigLIP', 'emb_siglip-so400m', 'emb_siglip-so400m'),
                                                                       ('CLIP (render_clip vs exact)', 'render_clip', 'emb_clip-vit-l-14'))}
        say(f'   {arm:2s} s{n:<2d} ' + ' | '.join(f'{k} {v.min():.6f} {np.median(v):.6f}' for k, v in cs.items())
            + f' | CLIP pictures under 0.999: {int((cs["CLIP (render_clip vs exact)"] < 0.999).sum())}')
    say()

    say('ITEM 3 what moves the centre (pycma replay; centre = the unit mean pycma samples around). Per round: (a) angle from pycma\'s own centre after tell (made unit, '
        'as the loop does before the re-anchor) to the previous centre; (b) angle from the re-anchored centre (unit mean of render_clip of the 4 feasible pictures lowest in told score) '
        'to the previous centre; (a\') the unit mean of the best 16 told requests, to the previous centre; (c) angle between (a) and (b). Medians over the 60 rounds.')
    drift = np.zeros((3, 5, 61))
    for ia, (arm, tg) in enumerate(ARMS):
        pa, pb, pf, pc = [], [], [], []
        for j, n in enumerate(STARTS):
            cell = f'qgc_{tg}_s{n}'
            z = Z[cell]
            prev, cm, rea, fits = S[f'{cell}__prev'], S[f'{cell}__cma'], S[f'{cell}__rea'], S[f'{cell}__fits']
            it = z['iteration'].astype(int)
            fb = np.array([unit(np.asarray(z['style_embs'][np.flatnonzero(it == r)], np.float64)[np.argsort(fits[r], kind='stable')[:16]].mean(0)) for r in range(60)])
            a, b, f, c = ang(cm, prev), ang(rea, prev), ang(fb, prev), ang(cm, rea)
            pa.append(a); pb.append(b); pf.append(f); pc.append(c)  # noqa: E702
            x0 = np.asarray(z['style_embs'][0], np.float64)
            drift[ia, j] = ang(np.vstack([prev, rea[-1:]]), np.repeat(x0[None], 61, 0))
            res[f'{cell}__angle_cma'], res[f'{cell}__angle_reanchor'], res[f'{cell}__angle_fallback16'] = a, b, f
            if arm != 'C2':
                say(f'   {arm:2s} s{n:<2d} (a) {np.median(a):7.3f} | (b) {np.median(b):7.3f} | (a\') {np.median(f):7.3f} | (c) {np.median(c):7.3f}')
        if arm != 'C2':
            say(f'   {arm:2s} pool (a) {np.median(np.concatenate(pa)):7.3f} | (b) {np.median(np.concatenate(pb)):7.3f} | (a\') {np.median(np.concatenate(pf)):7.3f} | (c) {np.median(np.concatenate(pc)):7.3f}')
        else:
            say(f'   C2 (for reference, sigma 1e-5) pool (a) {np.median(np.concatenate(pa)):7.4f} | (b) {np.median(np.concatenate(pb)):7.3f} | (a\') {np.median(np.concatenate(pf)):7.4f} | (c) {np.median(np.concatenate(pc)):7.3f}')
    res['drift_from_start'] = drift
    say('ITEM 3 drift: angle (degrees) of each round\'s sampling centre from the start (row 0 request = x0); index 60 = the centre after the last round. Median over the 5 starts [min, max]')
    for r in list(range(0, 60, 5)) + [59, 60]:
        say(f'   round {r:2d} | ' + ' | '.join(f'{arm} {np.median(drift[i, :, r]):6.2f} [{drift[i, :, r].min():6.2f}, {drift[i, :, r].max():6.2f}]' for i, (arm, _) in enumerate(ARMS)))
    say('   (all 61 values per cell in the npz: drift_from_start, axes arm V/C1/C2, start s6..s10, round 0..60)')
    say()

    say('ITEM 4 no-step runs (C2): per round, the 32 pictures of one request. Max pairwise chord among the 32 requests; median pairwise chord among the 32 readings '
        '(exact file) in CLIP, DINOv3, SigLIP; gap = chord(request, exact CLIP reading): std and max - min over the 32. Pooled median over the 60 rounds per cell [min, max over rounds]')
    for n in STARTS:
        cell = f'qgc_c2_s{n}'
        z, x = Z[cell], X[cell]
        it = z['iteration'].astype(int)
        rq, pc, pd, ps, gs, gr = [], [], [], [], [], []
        for r in range(60):
            rows = np.flatnonzero(it == r)
            U = unit(z['style_embs'][rows])
            rq.append(float(chord((U @ U.T)[np.triu_indices(32, 1)]).max()))
            pc.append(pairwise_median(x['emb_clip-vit-l-14'][rows]))
            pd.append(pairwise_median(x['emb_dinov3-l'][rows]))
            ps.append(pairwise_median(x['emb_siglip-so400m'][rows]))
            g = chord((U * unit(x['emb_clip-vit-l-14'][rows])).sum(1))
            gs.append(float(g.std()))
            gr.append(float(g.max() - g.min()))
        q = lambda v: f'{np.median(v):.4f} [{np.min(v):.4f}, {np.max(v):.4f}]'  # noqa: E731
        say(f'   C2 s{n:<2d} requests max chord {np.median(rq):.2e} [max {np.max(rq):.2e}] | CLIP {q(pc)} | DINOv3 {q(pd)} | SigLIP {q(ps)} | gap std {q(gs)} | gap range {q(gr)}')
        res[f'{cell}__c2_pair_clip'], res[f'{cell}__c2_pair_dino'], res[f'{cell}__c2_pair_siglip'], res[f'{cell}__c2_gap_std'] = np.array(pc), np.array(pd), np.array(ps), np.array(gs)
    say('   for scale: median pairwise chord among P\'s 2,048 repaints, CLIP {:.4f}, DINOv3 {:.4f}, SigLIP {:.4f}'.format(
        *[pairwise_median(Pz[k]) for k in ('emb_clip-vit-l-14', 'emb_dinov3-l', 'emb_siglip-so400m')]))
    say()

    say('ITEM 5 feasibility gate (infeasible = the run file\'s feasible flag: loop CLIP reading over 0.7788). Per cell: infeasible total; rounds with >= 1 and >= 2; told rank of the '
        'infeasible pictures in their round (1 = best told score; replayed told scores); how many in the best 16; how many beyond P\'s 99th percentile in both observers (F of the arms npz); '
        'REPLAY ON SAVED PICTURES, NOT A RERUN OF THE LOOP: evaluator replay of admission (controls_evaluate.replay, full P, same order, same spacing) with the fidelity condition '
        'switched off: admitted with the gate as run, admitted with it off, the extra pictures, and how many extra are over FID on the exact reading. '
        'The gate-as-run replay is asserted equal to the arms npz and to the run files\' own admitted flags. In C1 and C2 the told order is the random draw, so their ranks are random-draw ranks.')
    for arm, n, cell in cells:
        z = Z[cell]
        m = z['iteration'] >= 0
        it = z['iteration'][m].astype(int)
        inf = ~z['feasible'][m].astype(bool)
        per = np.bincount(it[inf], minlength=60)
        fits = S[f'{cell}__fits']
        ranks = []
        for r in range(60):
            rk = np.empty(32, int)
            rk[np.array(sorted(range(32), key=lambda i: fits[r][i]))] = np.arange(1, 33)
            ranks += rk[inf[it == r]].tolist()
        ranks = np.array(ranks, int)
        F = A[f'{cell}__F'].astype(bool)
        adm_on = ev.replay(z, P)[0][m]
        assert (adm_on == A[f'{cell}__EV'].astype(bool)).all(), cell
        assert (adm_on == z['admitted'][m].astype(bool)).all(), cell
        zoff = {k: z[k] for k in ('iteration', 'coherence', 'flags', 'emb_dinov3-l', 'emb_siglip-so400m')}
        zoff['fidelity_clip'] = np.zeros(len(z['iteration']))
        adm_off = ev.replay(zoff, P)[0][m]
        gap = A[f'{cell}__gap']
        extra = adm_off & ~adm_on
        say(f'   {arm:2s} s{n:<2d} infeasible {int(inf.sum()):4d} | rounds >=1 {int((per >= 1).sum()):2d}, >=2 {int((per >= 2).sum()):2d}, max in a round {int(per.max()):2d} | '
            f'told rank{"" if arm == "V" else " (random draw)"} median {np.median(ranks) if len(ranks) else float("nan"):5.1f}, in best 16 {int((ranks <= 16).sum()):4d} | far99 both {int((inf & F).sum()):4d} | '
            f'replay admitted gate as run {int(adm_on.sum()):3d}, gate off {int(adm_off.sum()):3d}, extra {int(extra.sum()):3d} (over FID exact {int((extra & (gap > FID)).sum()):3d}); '
            f'lost by the gate-off replay {int((adm_on & ~adm_off).sum())}')
        res[f'{cell}__infeasible_per_round'] = per
        res[f'{cell}__infeasible_told_rank'] = ranks
    say()

    say('ITEM 6 coordinates 644 and 413 (V and C1): per round, median over the 32 TOLD requests (the saved requests, made unit by the loop, which includes the projection) of |request_k - centre_k| / sigma_asked, '
        'centre = the replayed sampling centre; other = the same median over the other 766 coordinates pooled. Medians over the 60 rounds [max over rounds]; RAW ASK column = the same from pycma\'s raw asks before '
        'the projection (pycma replay); and the centre\'s own value in those coordinates (median over rounds)')
    for arm, n, cell in [c for c in cells if c[0] != 'C2']:
        z = Z[cell]
        it = z['iteration'].astype(int)
        prev, sa = S[f'{cell}__prev'], S[f'{cell}__s_asked']
        oth = np.setdiff1d(np.arange(768), ODD)
        r644, r413, rot = [], [], []
        for r in range(60):
            dv = np.abs(np.asarray(z['style_embs'][np.flatnonzero(it == r)], np.float64) - prev[r]) / sa[r]
            r644.append(np.median(dv[:, 644])); r413.append(np.median(dv[:, 413])); rot.append(np.median(dv[:, oth]))  # noqa: E702
        q = lambda v: f'{np.median(v):6.2f} [{np.max(v):6.2f}]'  # noqa: E731
        raw = S[f'{cell}__odd_raw']
        assert np.allclose(S[f'{cell}__odd_told'], np.array([r644, r413, rot]).T, rtol=0, atol=1e-5), cell   # the replayed told medians equal this script's (float32 storage floor)
        say(f'   {arm:2s} s{n:<2d} told: dim 644 {q(r644)} | dim 413 {q(r413)} | other {q(rot)} || raw ask: dim 644 {q(raw[:, 0])} | dim 413 {q(raw[:, 1])} | other {q(raw[:, 2])} '
            f'| centre value 644 {np.median(prev[:, 644]):+.3f}, 413 {np.median(prev[:, 413]):+.3f}')
    say()

    say('ITEM 7 file checks: n_reanchored; NaN count and row per raw_* array; n_sigma_capped where stored')
    for arm, n, cell in cells:
        z = Z[cell]
        nan = {k: np.flatnonzero(np.isnan(np.asarray(z[k], float))).tolist() for k in z.files if k.startswith('raw_')}
        say(f'   {arm:2s} s{n:<2d} n_reanchored {int(z["n_reanchored"])} | ' + ' | '.join(f'{k} NaN rows {v}' for k, v in nan.items())
            + (f' | n_sigma_capped {int(z["n_sigma_capped"])}' if 'n_sigma_capped' in z.files else ''))
    with open(prefix + '.npz', 'xb') as fh:
        np.savez_compressed(fh, **res)
    open(prefix + '.txt', 'x').write('\n'.join(OUT) + '\n')
    print('\n'.join(OUT), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
