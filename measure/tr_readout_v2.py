"""Leftover-share readout of the search against the random-score and no-step loops: counts over the leftover line per arm and start, and the ahead / behind rule.
Usage (CPU): python -B tr_readout_v2.py OUTPREFIX --prefix qgc_|qgt_ --ref REF.npz --basis BASIS.npz [--starts N ...] [--label TEXT] -> OUTPREFIX.txt, OUTPREFIX.npz"""
import argparse
import ast
import hashlib
import importlib.util
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
V3_FILE = f'{C}/arms_barfree_qgt_v3.py'
LF_FILE = f'{C}/leftover_fn.py'
IDS = os.environ['DUROOM_EMB'] + '/clip-vit-l-14.npz'
DINO, SIG = 'dinov3-l', 'siglip-so400m'
md5 = lambda f: hashlib.md5(open(f, 'rb').read()).hexdigest()
RD = (DINO, SIG)
RN = {DINO: 'DINOv3', SIG: 'SigLIP'}
ROLE = {'V': 'search', 'C1': 'random-score loop', 'C2': 'no-step loop'}
FH = [None]


def say(s=''):
    print(s, flush=True)
    FH[0].write(s + '\n')
    FH[0].flush()


def stop(msg):
    say(f'STOP: {msg}. No count was computed or printed.')
    FH[0].close()
    sys.exit(3)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rule(v, x):
    """the rule on the both-readers count: v, x = per-start counts (five starts) of the search and of a control"""
    if len(v) != 5 or len(x) != 5:
        return 'cannot tell (not five starts)'
    sv, sx = sum(v), sum(x)
    if all(a > b for a, b in zip(v, x)) and 4 * sv >= 5 * sx and (sx > 0 or sv >= 10):
        return 'ahead'
    if all(b > a for a, b in zip(v, x)) and 4 * sx >= 5 * sv and (sv > 0 or sx >= 10):
        return 'behind'
    return 'cannot tell'


def check_cell(cell, n, z, obs_ids, refbase, R):
    """Validity of one cell from the run file alone. Returns (reasons, facts)."""
    bad, f = [], {}
    need = ['flags', 'iteration', 'index', 'start_idx', 'style_embs', 'feasible', 'paths', 'log_sigma_asked'] + [f'emb_{o}' for o in RD] + [f'norm_q99_{o}' for o in RD]
    miss = [k for k in need if k not in z.files]
    if miss:
        return [f'missing arrays {miss}'], f
    it = np.asarray(z['iteration']).astype(np.int64)
    f['rows'] = len(it)
    if len(it) != 1921 or it[0] >= 0 or not (it[1:] >= 0).all() or it.max() != 59 or not (np.bincount(it[1:], minlength=60) == 32).all():
        return [f'rows {len(it)}, start row first {bool(it[0] < 0)}, not 60 x 32 after the start'], f
    if not np.all(np.diff(it * 1000 + np.asarray(z['index']).astype(np.int64)) > 0):
        bad.append('rows not in loop order')
    fl = ast.literal_eval(str(z['flags']))
    f['fl'] = fl
    if fl.get('tag') != cell:
        bad.append(f'flags tag {fl.get("tag")!r} is not {cell}')
    if int(fl.get('seed', -1)) != n:
        bad.append(f'stored seed {fl.get("seed")} is not the tag number {n}')
    if (int(fl.get('maxiter', -1)), int(fl.get('popsize', -1)), int(fl.get('reanchor', -1))) != (60, 32, 4):
        bad.append(f'maxiter / popsize / reanchor {fl.get("maxiter")} / {fl.get("popsize")} / {fl.get("reanchor")}, not 60 / 32 / 4')
    wid = int(obs_ids[int(z['start_idx'])])
    pool = np.load(fl['start_pool'], allow_pickle=True)
    j = np.flatnonzero(pool['rows_full'] == int(z['start_idx']))
    if len(j) != 1 or int(pool['ids'][j[0]]) != wid:
        bad.append(f'start_idx {int(z["start_idx"])} is not one row of {os.path.basename(fl["start_pool"])} with id {wid}')
    f['wid'] = wid
    lref = os.path.basename(str(fl.get('pushforward')))
    if lref != refbase:
        bad.append(f'the loop reference {lref} is not the given {refbase}')
    else:
        for o in RD:
            if abs(float(z[f'norm_q99_{o}']) - R[o]['bar']) >= 1e-6:
                bad.append(f'stored bar {o} {float(z[f"norm_q99_{o}"]):.6f} differs from the reference bar {R[o]["bar"]:.6f}')
    m = it >= 0
    for o in RD:
        e = np.asarray(z[f'emb_{o}'][m], np.float64)
        if not np.isfinite(e).all():
            bad.append(f'non-finite {o} readings')
    f['hash'] = hashlib.md5(z[f'emb_{DINO}'][0].astype(np.float32).tobytes()).hexdigest()
    f['norm_dev'] = {o: float(np.abs(np.linalg.norm(np.asarray(z[f'emb_{o}'][m], np.float64), axis=1) - 1).max()) for o in RD}
    return bad, f


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--prefix', required=True)
    p.add_argument('--ref', required=True)
    p.add_argument('--basis', required=True)
    p.add_argument('--starts', type=int, nargs='+')
    p.add_argument('--label', default='')
    a = p.parse_args()
    v3 = load_mod('arms_barfree_qgt_v3', V3_FILE)
    starts = tuple(a.starts) if a.starts else v3.STARTS_OF.get(a.prefix)
    assert starts, f'no default starts for prefix {a.prefix!r}: give --starts'
    cells = v3.cell_list(a.prefix, starts)
    refpath = a.ref if os.path.isabs(a.ref) else f'{O}/{a.ref}'
    for ext in ('.txt', '.npz'):
        assert not os.path.exists(a.out + ext), f'{a.out + ext} exists: refusing to overwrite'
    FH[0] = open(a.out + '.txt', 'x')
    lf = load_mod('leftover_fn', LF_FILE)
    sys.path.insert(0, C)
    from local_rescore import auc
    from today_rescore import far, pid, reference, score
    assert auc(np.array([2.0]), np.array([1.0])) == 1.0 and auc(np.array([1.0]), np.array([2.0])) == 0.0 and auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert (rule([5] * 5, [4] * 5) == 'ahead' and rule([6] * 5, [5] * 5) == 'cannot tell' and rule([4] * 5, [5] * 5) == 'behind' and rule([5] * 5, [6] * 5) == 'cannot tell'
            and rule([9, 9, 9, 9, 4], [4] * 5) == 'cannot tell' and rule([5] * 5, [5] * 5) == 'cannot tell'
            and rule([2, 2, 2, 2, 1], [0] * 5) == 'cannot tell' and rule([2] * 5, [0] * 5) == 'ahead'
            and rule([0] * 5, [2, 2, 2, 2, 1]) == 'cannot tell' and rule([0] * 5, [2] * 5) == 'behind'), 'rule self-test'
    refbase = os.path.basename(refpath)
    BZ = np.load(a.basis, allow_pickle=True)
    assert str(BZ['p_md5']) == md5(refpath), f'the basis was fitted on another reference than {refbase}'
    assert int(BZ['K']) == 32 and int(BZ['folds']) == 5 and int(BZ['seed']) == 0
    line = {o: float(BZ[f'null_q99_{o}']) for o in RD}
    for o in RD:
        assert abs(float(np.percentile(BZ[f'crossfit_share_{o}'], 99)) - line[o]) < 1e-15
    basis = {o: (np.asarray(BZ[f'mean_{o}']), np.asarray(BZ[f'V_{o}'])) for o in RD}
    Rz = np.load(refpath, allow_pickle=True)
    R = reference(Rz)
    R['id'] = pid(Rz['paths'])
    if a.label:
        say(f'LABEL: {a.label}')
    say('LEFTOVER READOUT: leftover share against the given reference; counts only.')
    say(f'Cells: prefix {a.prefix}, starts {list(starts)}; V = search ({a.prefix}ra_s*), C1 = random-score loop ({a.prefix}pin_s*), C2 = no-step loop ({a.prefix}c2_s*); 1,920 candidates per cell.')
    say(f'Reference {refbase}, {len(R["id"])} rows; basis {a.basis}, K 32, 5 artist folds, seed 0.')
    say('Line (null 99th) ' + ', '.join(f'{RN[o]} {line[o]:.6f}' for o in RD) + '; null median / MAD ' + ', '.join(f'{RN[o]} {float(BZ[f"null_median_{o}"]):.6f} / {float(BZ[f"null_mad_{o}"]):.6f}' for o in RD)
        + '; distance bars (reference own 99th) ' + ', '.join(f'{RN[o]} {R[o]["bar"]:.6f}' for o in RD))
    say('READINGS: the run file\'s own DINOv3 and SigLIP readings, for the leftover and the distance.')
    say()

    say('VALIDITY per cell (any failure stops the readout before any count)')
    obs_ids = np.load(IDS, allow_pickle=True)['ids'].astype(np.int64)
    Z, FACT, bad_all = {}, {}, []
    for arm, n, cell in cells:
        f_ = f'{v3.rundir(cell)}/all_candidates.npz'
        if not os.path.exists(f_):
            say(f'   INCOMPLETE {cell}: no {f_}')
            bad_all.append(cell)
            continue
        try:
            z = np.load(f_, allow_pickle=True)
            bad, f = check_cell(cell, n, z, obs_ids, refbase, R)
        except Exception as e:   # an unreadable or malformed file is an invalid cell, never skipped silently
            bad, f, z = [f'could not be read: {e!r}'], {}, None
        if bad:
            say(f'   INVALID {cell}: ' + '; '.join(bad))
            bad_all.append(cell)
            continue
        fl = f['fl']
        say(f'   {cell}: rows {f["rows"]} (start row first, 60 x 32, loop order) | start idx {int(z["start_idx"])} work {f["wid"]} | start DINOv3 md5 {f["hash"]} | seed {fl["seed"]}, '
            f'random_fitness {fl.get("random_fitness")}, sigma_path {os.path.basename(str(fl.get("sigma_path")))}, pushforward {os.path.basename(str(fl.get("pushforward")))} | '
            f'max | |e| - 1 | ' + ', '.join(f'{RN[o]} {v:.1e}' for o, v in f['norm_dev'].items()))
        Z[cell], FACT[cell] = z, f
    if bad_all:
        stop(f'{len(bad_all)} of {len(cells)} cells incomplete or invalid')
    hard = []
    tg = dict(v3.ARMS)
    for n in starts:
        tri = {arm: f'{a.prefix}{tg[arm]}_s{n}' for arm, _ in v3.ARMS}
        fls = {arm: FACT[c]['fl'] for arm, c in tri.items()}
        if fls['V'].get('random_fitness') or fls['V'].get('sigma_path'):
            hard.append(f'{tri["V"]}: the search carries random_fitness or a sigma_path')
        vdir = v3.rundir(tri['V']) + '/'
        if not fls['C1'].get('random_fitness') or os.path.basename(os.path.dirname(str(fls['C1'].get('sigma_path') or ''))) != os.path.basename(vdir.rstrip('/')):
            hard.append(f'{tri["C1"]}: random_fitness {fls["C1"].get("random_fitness")}, sigma_path {fls["C1"].get("sigma_path")} (must be random-score and name {vdir})')
        sa2 = np.asarray(Z[tri['C2']]['log_sigma_asked'], np.float64)
        if not fls['C2'].get('random_fitness') or os.path.basename(str(fls['C2'].get('sigma_path'))) != 'c2_sigma_1e-5.npz' or not (np.abs(sa2 - 1e-5) < 1e-15).all():
            hard.append(f'{tri["C2"]}: random_fitness {fls["C2"].get("random_fitness")}, sigma_path {fls["C2"].get("sigma_path")} (must be random-score at 1e-5)')
        if not np.array_equal(np.asarray(Z[tri['V']]['log_sigma_asked'], np.float64), np.asarray(Z[tri['C1']]['log_sigma_asked'], np.float64)):
            hard.append(f's{n}: the random-score loop asked step sizes are not the search ones exactly')
        if (len({int(Z[c]['start_idx']) for c in tri.values()}) != 1 or len({FACT[c]['hash'] for c in tri.values()}) != 1
                or len({float(fl['fidelity_max']) for fl in fls.values()}) != 1):
            hard.append(f's{n}: the three arms do not share one start row, start picture and fidelity_max')
    for h in hard:
        say(f'   HARD CHECK FAILED: {h}')
    if hard:
        stop(f'{len(hard)} hard checks failed')
    say(f'valid cells: {len(Z)} of {len(cells)}; hard checks passed (roles, step sizes, one start per start number, loop reference = {refbase} with its bars)')
    say()

    cv, arrs = {}, {}
    for arm, n, cell in cells:
        z, f = Z[cell], FACT[cell]
        m = z['iteration'] >= 0
        keep = R['id'] != f['wid']
        Rr = v3.without_rows(R, keep, RD)
        emb = {f'emb_{o}': z[f'emb_{o}'][m] for o in RD}
        F = far(score(Rr, emb), Rr)
        sh = {o: lf.share(z[f'emb_{o}'][m], *basis[o]) for o in RD}
        ov = {o: sh[o] > line[o] for o in RD}
        cv[cell] = dict(F=F, sh=sh, ov=ov, both=ov[DINO] & ov[SIG], dropped=int((~keep).sum()))
        arrs.update({f'{cell}__share_{o}': sh[o] for o in RD})
        arrs[f'{cell}__far'] = F
    say('rows of the reference left out per cell (the start work\'s own row, only if it is a row of the reference): ' + ', '.join(f'{c} {v["dropped"]}' for c, v in cv.items()))
    say()
    say('TABLE 1 per arm and start: n | over the line in both readers | in DINOv3 | in SigLIP | median share DINOv3, SigLIP | four cells far x over (both readers): '
        'FO far & over, FU far & under, NO near & over, NU near & under')
    cnt = {}
    for arm, tgx in v3.ARMS:
        for n in starts + ('pool',):
            ks = starts if n == 'pool' else (n,)
            cs = [cv[f'{a.prefix}{tgx}_s{k}'] for k in ks]
            cat = lambda key: np.concatenate([c[key] for c in cs])
            b, F = cat('both'), cat('F')
            ovd, ovs = np.concatenate([c['ov'][DINO] for c in cs]), np.concatenate([c['ov'][SIG] for c in cs])
            med = {o: float(np.median(np.concatenate([c['sh'][o] for c in cs]))) for o in RD}
            if n != 'pool':
                cnt[(arm, n)] = int(b.sum())
            say(f'   {arm:2s} {("s" + str(n)) if n != "pool" else "pool":4s} n {len(b):5d} | both {int(b.sum()):5d} | DINOv3 {int(ovd.sum()):5d} | SigLIP {int(ovs.sum()):5d} | median {med[DINO]:.4f}, {med[SIG]:.4f} | '
                f'FO {int((F & b).sum()):5d} FU {int((F & ~b).sum()):5d} NO {int((~F & b).sum()):5d} NU {int((~F & ~b).sum()):5d}')
    say()
    say('TABLE 2 per start and reader: chance that a search candidate\'s share is larger than a control candidate\'s (same start; ties half)')
    for o in RD:
        for x, tgx in (('C1', 'pin'), ('C2', 'c2')):
            v = [auc(cv[f'{a.prefix}ra_s{n}']['sh'][o], cv[f'{a.prefix}{tgx}_s{n}']['sh'][o]) for n in starts]
            arrs[f'auc__{o}__V_vs_{x}'] = np.array(v)
            say(f'   {RN[o]:6s} V vs {x} ({ROLE[x]}): ' + ', '.join(f's{n} {u:.4f}' for n, u in zip(starts, v)))
    say()
    say('RULE: "ahead" = more over the line in both readers at all five starts and at least a quarter more in sum, '
        'and at least 10 in sum when the control\'s sum is 0; "behind" the same way round; otherwise "cannot tell"')
    res = {}
    for x in ('C1', 'C2'):
        vv, xx = [cnt[('V', n)] for n in starts], [cnt[(x, n)] for n in starts]
        res[x] = rule(vv, xx)
        say(f'   search vs {x} ({ROLE[x]}): search {vv} = {sum(vv)}, {x} {xx} = {sum(xx)} -> {res[x].upper()}')
    with open(a.out + '.npz', 'xb') as fh:
        np.savez_compressed(fh, **arrs)
    say(f'wrote {a.out}.txt, {a.out}.npz')
    FH[0].close()


if __name__ == '__main__':
    main()
