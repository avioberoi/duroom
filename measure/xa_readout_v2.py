"""Readout of the extra arms (no-step command told the real score, repeats, other re-anchor sizes) against the main runs, from run files only: far, spaced, kept, pair chances, gap, per-round moves.
Usage: python -B xa_readout_v2.py replay OUT.npz [--arms ..] (search environment, pycma 4.4.4) | readout OUTPREFIX --replay OUT.npz [--arms ..] -> OUTPREFIX.txt, _rounds.csv, .npz, each refused if it exists."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import os
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
QV3 = f'{C}/arms_barfree_qgt_v3.py'
IDS = os.environ['DUROOM_EMB'] + '/clip-vit-l-14.npz'
CONF_START = {6: 182034, 7: 126645, 8: 56762, 9: 56179, 10: 71536}
STARTS = (6, 7, 8, 9, 10)
C2CFG, VCFG = 'config_sphere15_c2.yaml', 'config_sphere15.yaml'
# arm -> (tag prefix, card, random_fitness, sigma_path basename, reanchor, config basename, what)
ARMS = {'SC': ('qgx_sc', 'A100', False, 'c2_sigma_1e-5.npz', 4, C2CFG, 'no-step command told the real score (A100)'),
        'SCh': ('qgh_sc', 'H200', False, 'c2_sigma_1e-5.npz', 4, C2CFG, 'no-step command told the real score (H200)'),
        'NSa': ('qgx_c2', 'A100', True, 'c2_sigma_1e-5.npz', 4, C2CFG, 'no-step command unchanged, repeat (A100)'),
        'NSh': ('qgc_c2', 'H200', True, 'c2_sigma_1e-5.npz', 4, C2CFG, 'no-step loop (H200)'),
        'Vh': ('qgc_ra', 'H200', False, None, 4, VCFG, 'search, re-anchor 4 (H200)'),
        'M1': ('qgm_m1', 'H200', False, None, 1, VCFG, 'search command, re-anchor 1 (H200)'),
        'M16': ('qgm_m16', 'H200', False, None, 16, VCFG, 'search command, re-anchor 16 (H200)'),
        'M32': ('qgm_m32', 'H200', False, None, 32, VCFG, 'search command, re-anchor 32 (H200)')}
GROUPS = {'A': ('SC', 'NSa', 'NSh', 'Vh'), 'B': ('M1', 'M16', 'M32', 'Vh'), 'C': ('SCh', 'NSh', 'Vh')}
PAIRS = (('SCh', 'NSh'), ('SCh', 'Vh'), ('SC', 'SCh'), ('SC', 'NSa'), ('NSa', 'NSh'), ('SC', 'NSh'), ('Vh', 'NSh'), ('Vh', 'NSa'), ('Vh', 'SC'), ('M1', 'Vh'), ('M16', 'Vh'), ('M32', 'Vh'))
LEVELS = (50, 90, 99, 99.9)
QS = (1, 10, 25, 50, 75, 90, 99)
FH = [None]


def say(s=''):
    print(s, flush=True)
    if FH[0] is not None:
        FH[0].write(s + '\n')
        FH[0].flush()


def stop(msg):
    say(f'STOP: {msg}. Nothing after this line was computed.')
    sys.exit(3)


def tag_of(arm, n):
    t = f'{ARMS[arm][0]}_s{n}'
    assert 'qgt_' not in t
    return t


def rundir(tag):
    return f'{O}/runs/{tag}_committee_euclidean_min_seedrow'


def logs_of(tag):
    return f'{B}/logs/{tag}.out'


def status(tag):
    """'complete' or the reason a cell is skipped"""
    out = logs_of(tag)
    if not (os.path.exists(f'{rundir(tag)}/all_candidates.npz') and os.path.exists(out)):
        return 'not run'
    return 'complete' if ' done task=' in open(out, errors='replace').read() else 'incomplete (no completion line in the log)'


def cells_of(arms):
    out = []
    for arm in arms:
        for n in STARTS:
            t = tag_of(arm, n)
            out.append((arm, n, t, status(t)))
    return out


def unit64(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ang(a, b):
    a, b = unit64(a), unit64(b)
    return np.degrees(2 * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1)))


# replay (search environment)
def replay_main(a):
    spec = importlib.util.spec_from_file_location('abq3', QV3)
    abq3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abq3)
    import cma
    import yaml
    assert cma.__version__ == '4.4.4', cma.__version__
    assert not os.path.exists(a.out), a.out
    res, st = {'cma_version': np.array(cma.__version__)}, []
    for arm, n, t, s in cells_of(a.arms):
        if s != 'complete':
            st.append(f'{t} skipped {s}')
            continue
        try:
            res.update({f'{t}__{k}': v for k, v in abq3.replay_cell(t, cma, yaml).items()})
            st.append(f'{t} ok')
        except Exception as e:
            st.append(f'{t} FAILED {e!r}')
            print(f'{t}: REPLAY FAILED {e!r}', flush=True)
    res['status'] = np.array(st)
    with open(a.out, 'xb') as fh:
        np.savez_compressed(fh, **res)
    print('\n'.join(st))
    print('wrote', a.out)
    if any(' FAILED ' in s for s in st):
        sys.exit(4)


# readout
def load_helpers():
    sys.path.insert(0, C)
    import local_rescore as lr
    import today_rescore as tr
    spec = importlib.util.spec_from_file_location('controls_evaluate', f'{C}/controls_evaluate.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return lr, tr, ev


def validate(arm, n, t, z, obs_ids, lr):
    """reasons this complete cell does not match its arm (empty = valid), and facts"""
    pre, card, rf, sp, re_, cfgb, _ = ARMS[arm]
    bad, f = [], {}
    it = np.asarray(z['iteration']).astype(np.int64)
    if len(it) != 1921 or it[0] >= 0 or not (it[1:] >= 0).all() or it.max() != 59 or not (np.bincount(it[1:], minlength=60) == 32).all():
        return [f'rows {len(it)}: not the start row then 60 x 32'], f
    if not np.all(np.diff(it * 1000 + np.asarray(z['index']).astype(np.int64)) > 0):
        bad.append('rows not in loop order')
    fl = ast.literal_eval(str(z['flags']))
    want = {'tag': t, 'seed': n, 'maxiter': 60, 'popsize': 32, 'reanchor': re_, 'random_fitness': rf, 'fidelity_max': 0.7788, 'delta': 0.575, 'warmup': 1,
            'admit_q99': True, 'gate_threshold': 0.0, 'band': None, 'mode': 'cma', 'space': 'sphere', 'start': 'seedrow', 'aggregation': 'min', 'sigma_cap': 2.0,
            'norm_from_push': True, 'render_seeded': True, 'no_early_stop': True}
    for k, v in want.items():
        if fl.get(k) != v:
            bad.append(f'flag {k} = {fl.get(k)!r}, expected {v!r}')
    for k, v in (('sigma_path', sp), ('pushforward', 'push_emb.npz'), ('start_pool', 'starts_pre1900.npz'), ('config', cfgb)):
        got = None if fl.get(k) is None else os.path.basename(str(fl.get(k)))
        if got != v:
            bad.append(f'flag {k} basename {got!r}, expected {v!r}')
    wid = int(obs_ids[int(z['start_idx'])])
    if wid != CONF_START[n]:
        bad.append(f'start work {wid} is not the evaluator table id {CONF_START[n]} of seed {n}')
    for o, b in zip(lr.OBS, lr.BARS):
        if abs(float(z[f'norm_q99_{o}']) - b) >= 1e-6:
            bad.append(f'stored bar {o} {float(z[f"norm_q99_{o}"]):.6f} is not P\'s {b:.6f}')
    out = logs_of(t)
    head = open(out, errors='replace').read()
    g = re.search(r'^NVIDIA ([^,\n]+),', head, re.M)
    f['card'] = g.group(1) if g else '?'
    if 'CLIP loader: open_clip ViT-L-14-quickgelu' not in head:
        bad.append('log lacks the quickgelu loader line')
    f['n_stop_lines'] = head.count('CMA-ES stopped at iter')
    f['wid'], f['start_idx'] = wid, int(z['start_idx'])
    f['hash'] = hashlib.md5(np.asarray(z['emb_dinov3-l'][0], np.float32).tobytes()).hexdigest()
    return bad, f


def readout_main(a):
    pre = a.out
    for ext in ('.txt', '_rounds.csv', '.npz'):
        assert not os.path.exists(pre + ext), f'{pre + ext} exists: refusing to overwrite'
    FH[0] = open(pre + '.txt', 'x')
    lr, tr, ev = load_helpers()
    OBS, K, unit, chord, auc = lr.OBS, lr.K, lr.unit, lr.chord, lr.auc
    assert auc(np.array([2.0]), np.array([1.0])) == 1.0 and auc(np.array([1.0]), np.array([1.0])) == 0.5
    Pz = np.load(f'{O}/push_emb.npz', allow_pickle=True)
    P = tr.reference(Pz)
    Pid = tr.pid(Pz['paths'])
    for o, b in zip(OBS, lr.BARS):
        assert abs(P[o]['bar'] - b) < 1e-9, o
    lev = {o: {q: float(np.percentile(P[o]['own'], q)) for q in LEVELS} for o in OBS}
    obs_ids = np.load(IDS, allow_pickle=True)['ids'].astype(np.int64)
    S = np.load(a.replay, allow_pickle=True)
    rstat = {s.split(' ')[0]: s for s in S['status'].tolist()}
    say(f'replay file {a.replay} (pycma {S["cma_version"]}, via arms_barfree_qgt_v3.replay_cell)')
    say('DESCRIPTIVE: counts and distributions only. RUN FILES only: far scores use the run file\'s own DINOv3 / SigLIP readings; the gap is the loop\'s fidelity_clip '
        '(chord of the request to the loop reader\'s CLIP reading, render_clip), not an exact-file gap.')
    say(f'Reference P = push_emb.npz ({len(Pid)} pre-1900 repaints); P\'s own score = leave-one-out, artist rows out (today_rescore.reference); a candidate\'s far score = mean '
        f'chord to its {K} nearest P pictures, P alone (today_rescore.score). Levels: ' + '; '.join(f'{o} ' + ', '.join(f'p{q:g} {lev[o][q]:.4f}' for q in LEVELS) for o in OBS))
    say('Arms: ' + ' | '.join(f'{k} = {v[0]}_sN: {v[6]}' for k, v in ARMS.items() if k in a.arms))
    say()
    say('CELLS (status read from the log before any run file is opened)')
    cells = cells_of(a.arms)
    for arm, n, t, s in cells:
        say(f'   {arm:4s} s{n:<2d} {t:14s} {s}')
    ok = [(arm, n, t) for arm, n, t, s in cells if s == 'complete']
    skipped = [t for arm, n, t, s in cells if s != 'complete']
    say(f'complete {len(ok)} of {len(cells)}; skipped: {", ".join(skipped) if skipped else "none"}')
    say()
    say('VALIDITY per complete cell (any failure stops the readout before any table)')
    Z, FACT, hard = {}, {}, []
    for arm, n, t in ok:
        z = np.load(f'{rundir(t)}/all_candidates.npz', allow_pickle=True)
        bad, f = validate(arm, n, t, z, obs_ids, lr)
        if not bad and not rstat.get(t, '').endswith(' ok'):
            bad.append(f'replay status {rstat.get(t, "absent")}')
        if bad:
            hard.append(f'{t}: ' + '; '.join(bad))
            say(f'   INVALID {t}: ' + '; '.join(bad))
            continue
        Z[t], FACT[t] = z, f
        R = S
        ad = float(R[f'{t}__askdiff'].max())
        d1 = float(np.abs(R[f'{t}__s_asked'] - np.asarray(z['log_sigma_asked'], float)).max())
        d2 = float(np.abs(R[f'{t}__s_after'] - np.asarray(z['log_sigma'], float)).max())
        rep_ok = bool(R[f'{t}__used_replay'].all()) and ad < 1e-6 and d1 < 1e-12 and d2 < 1e-9
        if not rep_ok:
            hard.append(f'{t}: replay not exact (asks {int(R[f"{t}__used_replay"].sum())}/60, max ask diff {ad:.1e}, sigma asked {d1:.1e}, after tell {d2:.1e})')
        say(f'   {t}: card {f["card"]} | start idx {f["start_idx"]} work {f["wid"]} (evaluator table id) | start DINOv3 hash {f["hash"][:12]} | flags as the arm | stored bars = P\'s | '
            f'"CMA-ES stopped" lines {f["n_stop_lines"]} | replay asks equal {int(R[f"{t}__used_replay"].sum())}/60, max diff {ad:.1e}')
    if (Pid[:, None] == np.array([FACT[t]['wid'] for t in Z])[None, :]).any():
        hard.append('a start work is a row of P')
    for h in hard:
        say(f'   HARD CHECK FAILED: {h}')
    if hard:
        stop(f'{len(hard)} checks failed')
    for n in STARTS:
        hs = {t: FACT[t]['hash'][:12] for t in Z if t.endswith(f'_s{n}')}
        cards = {t: FACT[t]['card'] for t in hs}
        if hs:
            say(f'   s{n}: start picture DINOv3 hash by cell ' + ', '.join(f'{t} {h} ({cards[t]})' for t, h in hs.items()))
    say('checks passed: every complete cell carries its arm\'s flags, seed, the evaluator table start work, P\'s stored bars, 60 x 32 rows, the quickgelu loader line, and an exact pycma replay')
    say()

    V = {}
    for t, z in Z.items():
        m = z['iteration'] >= 0
        s = tr.score(P, {f'emb_{o}': z[f'emb_{o}'][m] for o in OBS})
        F = tr.far(s, P)
        dino_all = unit(z['emb_dinov3-l'])
        V[t] = dict(it=z['iteration'][m].astype(int), d=[s[o][0] for o in OBS], F=F, feas=z['feasible'][m].astype(bool), kept=z['admitted'][m].astype(bool),
                    sp=int(ev.spacing(dino_all, 1 + np.flatnonzero(F))), gap=np.asarray(z['fidelity_clip'], np.float64)[m],
                    gap_chk=float(np.abs(chord((unit(z['style_embs']) * unit(z['render_clip'])).sum(1))[m] - np.asarray(z['fidelity_clip'], np.float64)[m]).max()))
        assert int(np.all([V[t]['d'][i] > lev[o][99] for i, o in enumerate(OBS)], axis=0).sum()) == int(F.sum())

    def starts_of(group):
        return [n for n in STARTS if all(tag_of(arm, n) in Z for arm in group if arm in a.arms)]

    def pooled(arm, ns, key):
        return np.concatenate([V[tag_of(arm, n)][key] for n in ns]) if ns else np.zeros(0)

    say('TABLE 1 cascade per cell: candidates -> feasible (loop flag) | far (beyond P\'s own 99th percentile in BOTH DINOv3 and SigLIP, each candidate against P alone) '
        '-> far and spaced (controls_evaluate.spacing on the far ones alone, start first: the evaluator table\'s spacing-only count) | kept by the loop (admitted)')
    for arm in a.arms:
        for n in STARTS:
            t = tag_of(arm, n)
            if t in V:
                c = V[t]
                say(f'   {arm:4s} s{n:<2d} 1920 -> feasible {int(c["feas"].sum()):4d} | far {int(c["F"].sum()):4d} -> far and spaced {c["sp"]:4d} | kept {int(c["kept"].sum()):4d}')
    for g, members in GROUPS.items():
        ns = starts_of(members)
        say(f'   pooled, group {g} ({", ".join(m for m in members if m in a.arms)}), over the starts where every arm of the group is complete: {ns if ns else "none"}')
        for arm in members:
            if arm in a.arms and ns:
                ts = [V[tag_of(arm, n)] for n in ns]
                say(f'      {arm:4s} {1920 * len(ns):5d} -> feasible {sum(int(c["feas"].sum()) for c in ts):5d} | far {sum(int(c["F"].sum()) for c in ts):5d} -> far and spaced '
                    f'{sum(c["sp"] for c in ts):5d} | kept {sum(int(c["kept"].sum()) for c in ts):5d}')
    say()

    say('TABLE 2 share of candidates beyond P\'s own p50, p90, p99, p99.9 in BOTH readers (count / share)')
    for arm in a.arms:
        for n in STARTS:
            t = tag_of(arm, n)
            if t in V:
                c = V[t]
                say(f'   {arm:4s} s{n:<2d} ' + ' | '.join(f'>p{q:g} {int(np.all([c["d"][i] > lev[o][q] for i, o in enumerate(OBS)], axis=0).sum()):4d} '
                                                        f'({np.all([c["d"][i] > lev[o][q] for i, o in enumerate(OBS)], axis=0).mean():.4f})' for q in LEVELS))
    say()

    say('TABLE 3 pair chance per start and reader: P(a random candidate of the first arm has a larger far score than a random candidate of the second arm, same start), '
        'ties half (local_rescore.auc). Only starts where both cells are complete.')
    pairres = {}
    for x, y in PAIRS:
        if x not in a.arms or y not in a.arms:
            continue
        ns = [n for n in STARTS if tag_of(x, n) in V and tag_of(y, n) in V]
        for i, o in enumerate(OBS):
            v = [auc(V[tag_of(x, n)]['d'][i], V[tag_of(y, n)]['d'][i]) for n in ns]
            pairres[(x, y, o)] = dict(zip(ns, v))
            cx = {FACT[tag_of(x, n)]['card'] for n in ns} | {FACT[tag_of(y, n)]['card'] for n in ns}
            say(f'   {x:4s} vs {y:4s} {o:15s}: ' + (', '.join(f's{n} {u:.4f}' for n, u in zip(ns, v)) if ns else 'no start with both complete') + f' | cards {sorted(cx)}')
    say()

    say(f'TABLE 4 the loop\'s gap (fidelity_clip, run file; checked = chord(request, render_clip) per row) percentiles {QS}; above the loop gate 0.7788; infeasible (~feasible)')
    for arm in a.arms:
        for n in STARTS:
            t = tag_of(arm, n)
            if t in V:
                c = V[t]
                say(f'   {arm:4s} s{n:<2d} ' + ' '.join(f'p{q:g} {x:.4f}' for q, x in zip(QS, np.percentile(c['gap'], QS))) + f' | max {c["gap"].max():.4f} | above 0.7788 '
                    f'{int((c["gap"] > 0.7788).sum()):4d} | infeasible {int((~c["feas"]).sum()):4d} | check max|d| {c["gap_chk"]:.1e}')
    say()

    say('TABLE 5 per round (pycma replay; centre = the unit mean the round\'s 32 requests are drawn around): CMA-ES\'s own move = angle (deg) of pycma\'s centre after tell '
        '(made unit) from the round\'s centre; the restart\'s move = angle of the re-anchored centre (unit mean of render_clip of the re-anchor\'s feasible picks) from pycma\'s '
        'centre after tell; net move = re-anchored centre from the round\'s centre; spread = largest chord of the 32 unit requests from the centre, and largest pairwise chord. '
        'Per cell: median over the 60 rounds [min, max]; the centre\'s angle from the start after round 59.')
    rows_csv = []
    for arm in a.arms:
        for n in STARTS:
            t = tag_of(arm, n)
            if t not in V:
                continue
            z = Z[t]
            prev, cm, rea = S[f'{t}__prev'], S[f'{t}__cma'], S[f'{t}__rea']
            it = z['iteration'].astype(int)
            x0 = np.asarray(z['style_embs'][0], np.float64)
            a_, b_, c_ = ang(cm, prev), ang(rea, cm), ang(rea, prev)
            spc, spp = np.zeros(60), np.zeros(60)
            for r in range(60):
                Xr = unit64(z['style_embs'][np.flatnonzero(it == r)])
                spc[r] = float(chord(Xr @ unit64(prev[r])).max())
                spp[r] = float(chord(Xr @ Xr.T).max())
            drift = ang(np.vstack([prev, rea[-1:]]), np.repeat(x0[None], 61, 0))
            V[t].update(cma_move=a_, restart_move=b_, net_move=c_, spread_c=spc, spread_p=spp, drift=drift)
            q = lambda v, f='.5f': f'{np.median(v):{f}} [{v.min():{f}}, {v.max():{f}}]'  # noqa: E731
            say(f'   {arm:4s} s{n:<2d} CMA-ES own move {q(a_)} | restart\'s move {q(b_, ".3f")} | net move {q(c_, ".3f")} | spread from centre {q(spc, ".2e")} | '
                f'pairwise {q(spp, ".2e")} | centre from start after round 59: {drift[60]:.3f} deg')
            for r in range(60):
                rows_csv.append([t, arm, n, FACT[t]['card'], r, f'{float(z["log_sigma_asked"][r]):.6g}', f'{a_[r]:.6f}', f'{b_[r]:.6f}', f'{c_[r]:.6f}', f'{spc[r]:.6e}',
                                 f'{spp[r]:.6e}', f'{drift[r]:.6f}', f'{drift[r + 1]:.6f}', int((~V[t]['feas'][V[t]['it'] == r]).sum()), int(V[t]['F'][V[t]['it'] == r].sum()),
                                 int(V[t]['kept'][V[t]['it'] == r].sum())])
    say()


    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['cell', 'arm', 'start', 'card', 'round', 'sigma_asked', 'cma_own_move_deg', 'restart_move_deg', 'net_move_deg', 'spread_from_centre_chord',
                'spread_pairwise_chord', 'centre_from_start_deg_before', 'centre_from_start_deg_after', 'n_infeasible', 'n_far', 'n_kept'])
    w.writerows(rows_csv)
    open(pre + '_rounds.csv', 'x').write(buf.getvalue())
    arrs = {}
    for t, c in V.items():
        for k in ('it', 'F', 'feas', 'kept', 'gap', 'cma_move', 'restart_move', 'net_move', 'spread_c', 'spread_p', 'drift'):
            if k in c:
                arrs[f'{t}__{k}'] = c[k]
        arrs[f'{t}__d_dino'], arrs[f'{t}__d_siglip'] = c['d'][0], c['d'][1]
    with open(pre + '.npz', 'xb') as fh:
        np.savez_compressed(fh, **arrs)
    say(f'wrote {pre}.txt, {pre}_rounds.csv ({len(rows_csv)} rows), {pre}.npz')
    FH[0].close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['replay', 'readout'])
    p.add_argument('out')
    p.add_argument('--arms', nargs='+', default=list(ARMS))
    p.add_argument('--replay')
    a = p.parse_args()
    assert all(x in ARMS for x in a.arms), a.arms
    if a.mode == 'replay':
        return replay_main(a)
    assert a.replay, 'readout needs --replay'
    return readout_main(a)


if __name__ == '__main__':
    main()
