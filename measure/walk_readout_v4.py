"""Counts per cell of the single walk and the three loops from run files only: far, faithful, far and faithful, far and spaced, kept; search against walk by the rule words.
Usage (CPU, cwd = the search folder): python -B walk_readout_v4.py OUTPREFIX [--arms W V C1 C2] -> OUTPREFIX.txt and OUTPREFIX.npz, each refused if it exists."""
import argparse
import ast
import hashlib
import importlib.util
import os
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
TR_FILE = f'{C}/tr_readout_v2.py'
THETA_BASE = 'walk_theta_v1.npz'
IDS = os.environ['DUROOM_EMB'] + '/clip-vit-l-14.npz'
CONF_START = {6: 182034, 7: 126645, 8: 56762, 9: 56179, 10: 71536}
STARTS = (6, 7, 8, 9, 10)
GAP = 0.7788
# arm -> tag prefix, mode, reanchor, random_fitness, config basename, what
ARMS = {'W': ('qgw_walk', 'walk', None, False, 'config_sphere15.yaml', 'single walk with selection'),
        'V': ('qgc_ra', 'cma', 4, False, 'config_sphere15.yaml', 'search'),
        'C1': ('qgc_pin', 'cma', 4, True, 'config_sphere15.yaml', 'random-score loop (the search\'s step sizes)'),
        'C2': ('qgc_c2', 'cma', 4, True, 'config_sphere15_c2.yaml', 'no-step loop')}
WALK_ARRAYS = ('accepted', 'parent_row', 'theta', 'sel_novelty', 'parent_novelty', 'parent_novelty_plain', 'index_version')
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
    p = f'{O}/runs/{tag}_committee_euclidean_min_seedrow'
    assert 'qgt_' not in p
    return p


def logs_of(tag):
    return f'{B}/logs/{tag}.out'


def status(tag):
    out = logs_of(tag)
    if not (os.path.exists(f'{rundir(tag)}/all_candidates.npz') and os.path.exists(out)):
        return 'not run'
    return 'complete' if ' done task=' in open(out, errors='replace').read() else 'incomplete (no completion line in the log)'


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate(arm, n, t, z, obs_ids, lr):
    pre, mode, re_, rf, cfgb, _ = ARMS[arm]
    bad, f = [], {}
    it = np.asarray(z['iteration']).astype(np.int64)
    if len(it) != 1921 or it[0] >= 0 or not (it[1:] >= 0).all() or it.max() != 59 or not (np.bincount(it[1:], minlength=60) == 32).all():
        return [f'rows {len(it)}: not the start row then 60 x 32'], f
    if not np.all(np.diff(it * 1000 + np.asarray(z['index']).astype(np.int64)) > 0):
        bad.append('rows not in loop order')
    fl = ast.literal_eval(str(z['flags']))
    want = {'tag': t, 'seed': n, 'mode': mode, 'reanchor': re_, 'random_fitness': rf, 'maxiter': 60, 'popsize': 32, 'fidelity_max': GAP, 'delta': 0.575, 'warmup': 1,
            'admit_q99': True, 'gate_threshold': 0.0, 'band': None, 'space': 'sphere', 'start': 'seedrow', 'aggregation': 'min', 'sigma_cap': 2.0,
            'norm_from_push': True, 'render_seeded': True, 'no_early_stop': True, 'log_candidates': True, 'noise_seed': None, 'k': 15, 'metric': 'euclidean'}
    for k, v in want.items():
        if fl.get(k) != v:
            bad.append(f'flag {k} = {fl.get(k)!r}, expected {v!r}')
    for k, v in (('pushforward', 'push_emb.npz'), ('start_pool', 'starts_pre1900.npz'), ('config', cfgb)):
        got = None if fl.get(k) is None else os.path.basename(str(fl.get(k)))
        if got != v:
            bad.append(f'flag {k} basename {got!r}, expected {v!r}')
    sp = fl.get('sigma_path')
    want_sp = {'W': None, 'V': None, 'C1': f'{O}/runs/qgc_ra_s{n}_committee_euclidean_min_seedrow/all_candidates.npz', 'C2': 'c2_sigma_1e-5.npz'}[arm]
    if (os.path.basename(str(sp)) if (arm == 'C2' and sp) else sp) != want_sp:
        bad.append(f'flag sigma_path {sp!r}, expected {want_sp!r}')
    if arm == 'W':
        if os.path.basename(str(fl.get('walk_theta'))) != THETA_BASE:
            bad.append(f'flag walk_theta {fl.get("walk_theta")!r}, expected basename {THETA_BASE}')
        miss = [k for k in WALK_ARRAYS if k not in z.files]
        if miss:
            bad.append(f'walk arrays missing {miss}')
    elif fl.get('walk_theta') is not None:
        bad.append('a loop cell carries walk_theta')
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
    f['n_interrupted'] = len(re.findall(r'^Interrupted: ', head, re.M))
    if f['n_interrupted']:
        bad.append('log has an "Interrupted: " line (the loop\'s outer except: the run ended early)')
    if arm == 'W':
        rr = sorted(int(x) for x in re.findall(r'^  walk round (\d+): ', head, re.M))
        if rr != list(range(60)):
            bad.append(f'walk log has round lines for {len(rr)} rounds, not rounds 0..59 once each')
    f['wid'], f['start_idx'] = wid, int(z['start_idx'])
    f['hash'] = hashlib.md5(np.asarray(z['emb_dinov3-l'][0], np.float32).tobytes()).hexdigest()
    return bad, f


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--arms', nargs='+', default=list(ARMS))
    a = p.parse_args()
    assert all(x in ARMS for x in a.arms) and 'qgt_' not in a.out, a.arms
    pre = a.out
    for ext in ('.txt', '.npz'):
        assert not os.path.exists(pre + ext), f'{pre + ext} exists: refusing to overwrite'
    FH[0] = open(pre + '.txt', 'x')
    sys.path.insert(0, C)
    import local_rescore as lr
    import today_rescore as tr_
    ev = load_mod('controls_evaluate', f'{C}/controls_evaluate.py')
    trr = load_mod('tr_readout_v2', TR_FILE)
    assert trr.rule([6] * 5, [5] * 5) == 'cannot tell' and trr.rule([20] * 5, [1] * 5) == 'ahead' and trr.rule([1] * 5, [20] * 5) == 'behind'
    OBS, unit = lr.OBS, lr.unit
    Pz = np.load(f'{O}/push_emb.npz', allow_pickle=True)
    P = tr_.reference(Pz)
    Pid = tr_.pid(Pz['paths'])
    for o, b in zip(OBS, lr.BARS):
        assert abs(P[o]['bar'] - b) < 1e-9, o
    obs_ids = np.load(IDS, allow_pickle=True)['ids'].astype(np.int64)
    say('STEP LENGTH: the walk\'s step angle for round r is the median angle between the search\'s 32 requests of round r and their centre at the same start '
        f'({THETA_BASE}). It matches one summary of the search\'s spread, not its full step-length distribution or its per-coordinate scaling. '
        'Step-size columns (log_sigma, log_sigma_asked) do not apply to the walk: its run files carry log_sigma as NaN and no log_sigma_asked; none is read here.')
    say('SCOPE: run-file readings only (the loop\'s own DINOv3 / SigLIP readings and its gap fidelity_clip). The far-and-faithful count here is not the evaluator\'s, '
        'which uses exact readings and the evaluator\'s faithful flag. Counts only.')
    say('COMPLETION: a cell whose log has a line starting "Interrupted: " is rejected and stops the readout; a walk cell must also carry all 60 "  walk round <r>: " lines.')
    say(f'Reference P = push_emb.npz ({len(Pid)} pre-1900 repaints); far = beyond P\'s own p99 in both readers (bars {", ".join(f"{o} {b:.6f}" for o, b in zip(OBS, lr.BARS))}); '
        f'faithful = fidelity_clip <= {GAP} (the loop\'s gap bar).')
    say('Arms: ' + ' | '.join(f'{k} = {v[0]}_sN: {v[5]}' for k, v in ARMS.items() if k in a.arms))
    say()
    say('CELLS (status read once, before any run file is opened)')
    cells = [(arm, n, tag_of(arm, n)) for arm in a.arms for n in STARTS]
    st = {t: status(t) for _, _, t in cells}
    for arm, n, t in cells:
        say(f'   {arm:3s} s{n:<2d} {t:14s} {st[t]}')
    ok = [(arm, n, t) for arm, n, t in cells if st[t] == 'complete']
    say(f'complete {len(ok)} of {len(cells)}; skipped: {", ".join(t for _, _, t in cells if st[t] != "complete") or "none"}')
    say()
    say('VALIDITY per complete cell (any failure stops the readout before any table)')
    Z, FACT, hard = {}, {}, []
    for arm, n, t in ok:
        z = np.load(f'{rundir(t)}/all_candidates.npz', allow_pickle=True)
        bad, f = validate(arm, n, t, z, obs_ids, lr)
        if bad:
            hard.append(f'{t}: ' + '; '.join(bad))
            say(f'   INVALID {t}: ' + '; '.join(bad))
            continue
        Z[t], FACT[t] = z, f
        say(f'   {t}: card {f["card"]} | start idx {f["start_idx"]} work {f["wid"]} (evaluator table id) | start DINOv3 hash {f["hash"][:12]} | flags as the arm | stored bars = P\'s')
    if (Pid[:, None] == np.array([FACT[t]['wid'] for t in Z] or [-1])[None, :]).any():
        hard.append('a start work is a row of P')
    for h in hard:
        say(f'   HARD CHECK FAILED: {h}')
    if hard:
        stop(f'{len(hard)} checks failed')
    say('checks passed')
    say()

    V = {}
    for t, z in Z.items():
        m = z['iteration'] >= 0
        s = tr_.score(P, {f'emb_{o}': z[f'emb_{o}'][m] for o in OBS})
        F = tr_.far(s, P)
        gap = np.asarray(z['fidelity_clip'], np.float64)[m]
        faith = gap <= GAP
        feas = z['feasible'][m].astype(bool)
        if not (faith == feas).all():
            stop(f'{t}: faithful by the gap differs from the loop\'s feasible flag at {int((faith != feas).sum())} rows')
        c = dict(it=z['iteration'][m].astype(int), F=F, faith=faith, FF=F & faith, kept=z['admitted'][m].astype(bool),
                 sp=int(ev.spacing(unit(z['emb_dinov3-l']), 1 + np.flatnonzero(F))))
        if t.startswith('qgw_'):
            c['acc'] = z['accepted'][m].astype(bool)
        V[t] = c
    say('TABLE 1 per cell: candidates | faithful (gap <= 0.7788) | far (both readers, P alone) | far and faithful | far and spaced (spacing on the far ones, start first) | '
        'kept (the loop\'s admissions) | walk: moves kept')
    for arm in a.arms:
        for n in STARTS:
            t = tag_of(arm, n)
            if t in V:
                c = V[t]
                say(f'   {arm:3s} s{n:<2d} 1920 | faithful {int(c["faith"].sum()):4d} | far {int(c["F"].sum()):4d} | far and faithful {int(c["FF"].sum()):4d} | far and spaced '
                    f'{c["sp"]:4d} | kept {int(c["kept"].sum()):4d}' + (f' | moves kept {int(c["acc"].sum()):4d}' if 'acc' in c else ''))
        ns = [n for n in STARTS if tag_of(arm, n) in V]
        if ns:
            ts = [V[tag_of(arm, n)] for n in ns]
            say(f'   {arm:3s} sum over {ns}: {1920 * len(ns)} | faithful {sum(int(c["faith"].sum()) for c in ts)} | far {sum(int(c["F"].sum()) for c in ts)} | far and faithful '
                f'{sum(int(c["FF"].sum()) for c in ts)} | far and spaced {sum(c["sp"] for c in ts)} | kept {sum(int(c["kept"].sum()) for c in ts)}'
                + (f' | moves kept {sum(int(c["acc"].sum()) for c in ts)}' if 'acc' in ts[0] else ''))
    say()
    if 'W' in a.arms:
        say('TABLE 2 the walk per round (60 rounds of 32 steps): moves kept | admitted')
        for n in STARTS:
            t = tag_of('W', n)
            if t in V:
                c = V[t]
                say(f'   W s{n:<2d} kept moves per round ' + ' '.join(str(int(c['acc'][c['it'] == r].sum())) for r in range(60)))
                say(f'   W s{n:<2d} admitted per round   ' + ' '.join(str(int(c['kept'][c['it'] == r].sum())) for r in range(60)))
                say(f'   W s{n:<2d} rounds with no kept move: {int(sum(c["acc"][c["it"] == r].sum() == 0 for r in range(60)))}; last round with a kept move: '
                    f'{max([r for r in range(60) if c["acc"][c["it"] == r].any()], default=-1)}')
        say()

    def per_start(arm, key):
        return [int(V[tag_of(arm, n)][key].sum()) if key != 'sp' else V[tag_of(arm, n)]['sp'] for n in STARTS]

    def full(arm):
        return all(tag_of(arm, n) in V for n in STARTS)

    say('RULE (tr_readout_v2.rule): "ahead" = more at all five starts and 4 x sum_search >= 5 x sum_other and, when the other\'s sum is 0, sum_search >= 10; '
        '"behind" the same the other way round; otherwise "cannot tell" (not "equal").')
    words = {}
    if 'W' in a.arms and 'V' in a.arms and full('W') and full('V'):
        for key, name in (('FF', 'far and faithful'), ('kept', 'kept by the loop\'s own admissions')):
            v, x = per_start('V', key), per_start('W', key)
            words[key] = trr.rule(v, x)
            say(f'   search vs walk on {name}: search {v} (sum {sum(v)}) | walk {x} (sum {sum(x)}) -> search is "{words[key]}"')
    else:
        say('   search vs walk: not computed (the walk or the search is not complete at all five starts)')
    arrs = {f'{t}__{k}': c[k] for t, c in V.items() for k in ('it', 'F', 'faith', 'FF', 'kept', 'acc') if k in c}
    with open(pre + '.npz', 'xb') as fh:
        np.savez_compressed(fh, **arrs)
    say(f'wrote {pre}.txt, {pre}.npz')
    FH[0].close()


if __name__ == '__main__':
    main()
