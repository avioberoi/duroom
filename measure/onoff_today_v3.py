"""Off-the-manifold shares against today's reference: per arm and start, pictures far in both readers from every original (nothing masked), beside their leftover against today's directions.
Usage (CPU, cwd = the search folder): python -B onoff_today_v3.py OUTPREFIX --prefix qgc_|qgt_ [--measure-npz FILE] -> OUTPREFIX.txt and OUTPREFIX.npz, each refused if it exists."""
import argparse
import hashlib
import importlib.util
import os
import sys
import types

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
V3_FILE = f'{C_}/arms_barfree_qgt_v3.py'
TR_FILE = f'{C_}/tr_readout_v2.py'
LF_FILE = f'{C_}/leftover_fn.py'
T3_FILE = f'{C_}/t3_measure_v5.py'
SCALES = f'{B}/tests12/t3/out/scales_v5.npz'
GUARDS = f'{B}/tests12/t3/out/guards_v3.npz'
TBASIS = f'{B}/tests12/tworulers/out/tr_basis_v1_T_basis.npz'
IDS = os.environ['DUROOM_EMB'] + '/clip-vit-l-14.npz'
AFTERRUN = f'{B}/tests12/afterrun'
EXACT = {'qgt_': f'{O}/eval/{{tag}}_emb_h200_v1.npz', 'qgc_': f'{O}/eval/{{tag}}_emb.npz'}
LOOPREF = {'qgt_': 'push_today_emb.npz', 'qgc_': 'push_emb.npz'}   # the reference the loops ran against (their --pushforward), for check_cell only
DINO, SIG = 'dinov3-l', 'siglip-so400m'
RD = (DINO, SIG)
ROLE = {'V': 'search', 'C1': 'random-score loop', 'C2': 'no-step loop'}
md5 = lambda f: hashlib.md5(open(f, 'rb').read()).hexdigest()  # noqa: E731
FH = [None]
c_ = lambda x: int(np.sum(x))  # noqa: E731


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


def main():
    if not __debug__:
        raise SystemExit('onoff_today_v3: asserts are off (python -O); refusing to run')
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--prefix', required=True, choices=('qgc_', 'qgt_'))
    p.add_argument('--measure-npz', default=None)
    a = p.parse_args()
    v3 = load_mod('arms_barfree_qgt_v3', V3_FILE)
    starts = v3.STARTS_OF[a.prefix]
    cells = v3.cell_list(a.prefix, starts)
    run_tmpl, exact_tmpl = f'{O}/runs/{{tag}}_committee_euclidean_min_seedrow/all_candidates.npz', EXACT[a.prefix]
    for ext in ('.txt', '.npz'):
        assert not os.path.exists(a.out + ext), f'{a.out + ext} exists: refusing to overwrite'
    FH[0] = open(a.out + '.txt', 'x')
    say(f'onoff_today_v3.py; prefix {a.prefix}; starts {list(starts)}; measure npz {a.measure_npz}')
    # the measure file is required for qgt_ and must hold the per-arm flags (checked before anything else is read)
    need = [f'{arm}__{k}' for arm, _ in v3.ARMS for k in ('tags', 'cell_E', 'cell_e')]
    mz = None
    if a.measure_npz is None:
        if a.prefix == 'qgt_':
            stop('--prefix qgt_ requires --measure-npz (the measure must have run first)')
    else:
        if not os.path.isfile(a.measure_npz):
            stop(f'measure npz {a.measure_npz} is missing')
        mz = np.load(a.measure_npz, allow_pickle=True)
        miss = [k for k in need if k not in mz.files]
        if miss:
            stop(f'measure npz {a.measure_npz} stores no per-arm flags (missing {miss}; a rule-1 measure output stores no arms)')
        say(f'measure npz {a.measure_npz}: per-arm flags present')
    tr = load_mod('tr_readout_v2', TR_FILE)
    lf = load_mod('leftover_fn', LF_FILE)
    t3 = load_mod('t3_measure_v5', T3_FILE)
    sys.path.insert(0, C_)
    from today_rescore import pid, reference
    assert tr.rule([5] * 5, [4] * 5) == 'ahead' and tr.rule([2, 2, 2, 2, 1], [0] * 5) == 'cannot tell' and tr.rule([0] * 5, [2] * 5) == 'behind' and tr.rule([0] * 5, [0] * 5) == 'cannot tell'
    say('DEFINITION: OFF = no original of the corpus near in either reader, nothing masked (far in both readers at the '
        'measure\'s bars). The leftover against today\'s directions is a degree beside, not a condition.')
    say('RECIPE: chords e from t3_measure_v5 (readings, all corpus rows, the start work and its near-duplicates masked, exact file ' + exact_tmpl + '); '
        'u = min chord to the masked rows; d_all = min(e, u); bars theta_99 x med (scales_v5); far = both > bar, near = both <= bar. LL_T as tr_readout_v2 '
        '(run-file readings, leftover_fn.share, basis as stored, strict >).')

    pt = f'{O}/push_today_emb.npz'
    BZ = np.load(TBASIS, allow_pickle=True)
    assert str(BZ['p_md5']) == md5(pt)
    assert int(BZ['K']) == 32 and int(BZ['folds']) == 5 and int(BZ['seed']) == 0
    line = {o: float(BZ[f'null_q99_{o}']) for o in RD}
    for o in RD:
        assert abs(float(np.percentile(BZ[f'crossfit_share_{o}'], 99)) - line[o]) < 1e-15
    basis = {o: (np.asarray(BZ[f'mean_{o}']), np.asarray(BZ[f'V_{o}'])) for o in RD}
    say(f'today\'s line (null 99th): DINOv3 {line[DINO]:.6f}, SigLIP {line[SIG]:.6f}')

    say()
    say('VALIDITY per cell')
    Rz = np.load(f'{O}/{LOOPREF[a.prefix]}', allow_pickle=True)
    Rloop = reference(Rz)
    Rloop['id'] = pid(Rz['paths'])
    obs_ids = np.load(IDS, allow_pickle=True)['ids'].astype(np.int64)
    Z, FACT, bad_all = {}, {}, []
    for arm, n, cell in cells:
        f_ = run_tmpl.format(tag=cell)
        assert f_ == f'{v3.rundir(cell)}/all_candidates.npz'
        if not os.path.exists(f_):
            say(f'   INCOMPLETE {cell}: no {f_}')
            bad_all.append(cell)
            continue
        try:
            z = np.load(f_, allow_pickle=True)
            bad, f = tr.check_cell(cell, n, z, obs_ids, LOOPREF[a.prefix], Rloop)
        except Exception as e:
            bad, f, z = [f'could not be read: {e!r}'], {}, None
        if bad:
            say(f'   INVALID {cell}: ' + '; '.join(bad))
            bad_all.append(cell)
            continue
        say(f'   {cell}: rows {f["rows"]}, start idx {int(z["start_idx"])} work {f["wid"]}, seed {f["fl"]["seed"]}, random_fitness {f["fl"].get("random_fitness")}, '
            f'sigma_path {os.path.basename(str(f["fl"].get("sigma_path")))}, pushforward {os.path.basename(str(f["fl"].get("pushforward")))}')
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
            hard.append(f'{tri["C1"]}: random_fitness {fls["C1"].get("random_fitness")}, sigma_path {fls["C1"].get("sigma_path")}')
        sa2 = np.asarray(Z[tri['C2']]['log_sigma_asked'], np.float64)
        if not fls['C2'].get('random_fitness') or os.path.basename(str(fls['C2'].get('sigma_path'))) != 'c2_sigma_1e-5.npz' or not (np.abs(sa2 - 1e-5) < 1e-15).all():
            hard.append(f'{tri["C2"]}: random_fitness {fls["C2"].get("random_fitness")}, sigma_path {fls["C2"].get("sigma_path")}')
        if not np.array_equal(np.asarray(Z[tri['V']]['log_sigma_asked'], np.float64), np.asarray(Z[tri['C1']]['log_sigma_asked'], np.float64)):
            hard.append(f's{n}: the random-score loop asked step sizes are not the search ones exactly')
        if (len({int(Z[c]['start_idx']) for c in tri.values()}) != 1 or len({FACT[c]['hash'] for c in tri.values()}) != 1
                or len({float(fl['fidelity_max']) for fl in fls.values()}) != 1):
            hard.append(f's{n}: the three arms do not share one start row, start picture and fidelity_max')
    for h in hard:
        say(f'   HARD CHECK FAILED: {h}')
    if hard:
        stop(f'{len(hard)} hard checks failed')
    ns = types.SimpleNamespace(corpus='real', root=O, rows=1921, rounds=60, per_round=32, scales=SCALES, run_tmpl=run_tmpl, exact_tmpl=exact_tmpl, afterrun=AFTERRUN,
                               acc_prior=f'{B}/tests12/t2/renders_v1/prior256_v1', acc_noise=f'{B}/tests12/t2/renders_v1/noise160_v1', acc_aim_prior=None, acc_aim_noise=None)
    if a.prefix == 'qgt_':
        for _, _, cell in cells:
            say('   provenance (t3_measure_v5.provenance): ' + t3.provenance(ns, cell))
    else:
        say(f'   provenance: no COMPLETE markers for these cells; exact file {exact_tmpl} as the guards read it')
    C, RP, idx = t3.load_common(ns)
    sc = t3.load_scales(ns)
    cal = sc['cal'][99]
    BAR = np.asarray(cal['theta'] * cal['med'], np.float64)
    t3.t_readings(C, RP, idx, sc)
    LC = {}
    for arm, n, cell in cells:
        LC[cell] = t3.load_cell(cell, run_tmpl, exact_tmpl, 1921, C, layout=(60, 32), arm=arm, vtag=f'{a.prefix}ra_s{n}')
        if not LC[cell]['valid']:
            stop(f'{cell}: load_cell: {LC[cell]["why"]}')
        if C['ids'][LC[cell]['start']] != str(FACT[cell]['wid']):
            stop(f'{cell}: the measure\'s start work {C["ids"][LC[cell]["start"]]} is not the run\'s start work {FACT[cell]["wid"]}')
    say(f'valid cells: {len(Z)} of {len(cells)}; hard checks passed; load_cell valid for every cell; start works agree')
    # run-file reading against the exact reading of the same row (printed only)
    for arm, n, cell in cells:
        z, lc = Z[cell], LC[cell]
        m = np.asarray(z['iteration']) >= 0
        cs = {o: (t3.unit(np.asarray(z[f'emb_{o}'])[m]) * lc['E'][o]).sum(1) for o in RD}
        say(f'   run vs exact reading, {cell}: ' + '; '.join(f'{o} cosine min {cs[o].min():.6f} median {np.median(cs[o]):.6f}' for o in RD))
    dups = t3.dup_rows(C, np.concatenate([RP['T']['src'], RP['P2']['src'], np.array([LC[c]['start'] for _, _, c in cells], int)]))
    full = t3.Ctx(C, idx, dups)
    ne = lambda Em, src: np.asarray(t3.readings(dict(E=Em, src=np.asarray(src)), full)['e'], np.float64)  # noqa: E731
    farE = lambda e: np.all(e > BAR, axis=1)  # noqa: E731
    nearE = lambda e: np.all(e <= BAR, axis=1)  # noqa: E731
    g = np.load(GUARDS, allow_pickle=True)
    acc = t3.load_acc(ns, need_aim=False)
    for key, lab, e in (('gno1', 'G-no (i) P2', ne(RP['P2']['E'], RP['P2']['src'])), ('gno3', 'G-no (iii) ACC', ne(acc['E'], acc['src']))):
        ok = np.array_equal(farE(e), np.asarray(g[f'{key}__E'], bool)) and float(np.abs(e - g[f'{key}__e']).max()) < 1e-6
        say(f'   guard check {lab}: FAR {c_(farE(e))} of {len(e)}; flags = guard file picture by picture, chords to 1e-6: {ok}')
        if not ok:
            stop(f'{lab} does not reproduce the guard file')
    say(f'bars D {BAR[0]:.6f} / S {BAR[1]:.6f}')

    cv, sv = {}, {'bars': BAR, 'line_T': np.array([line[DINO], line[SIG]])}
    for arm, n, cell in cells:
        z, lc = Z[cell], LC[cell]
        m = np.asarray(z['iteration']) >= 0
        rp, ep = [str(x) for x in np.asarray(z['paths'])[m]], [str(x) for x in lc['paths'][1:]]
        assert len(rp) == len(ep) == 1920 and rp == ep
        sh = {o: lf.share(z[f'emb_{o}'][m], *basis[o]) for o in RD}
        ov = {o: sh[o] > line[o] for o in RD}
        e = ne(lc['E'], lc['src'])
        rows = full.rows(int(lc['start']))
        u = np.full((1920, 2), np.inf)
        if len(rows):
            for j, o in enumerate(RD):
                u[:, j] = t3.chord((lc['E'][o] @ C['E'][o][rows].T).max(1))
        dall = np.minimum(e, u)
        cv[cell] = dict(e=e, dall=dall, sh=sh, ovD=ov[DINO], ovS=ov[SIG], ll=ov[DINO] & ov[SIG], OFF=farE(dall), Fm=farE(e))
        sv.update({f'{cell}__share_{o}': sh[o] for o in RD})
        sv.update({f'{cell}__LL_T': cv[cell]['ll'], f'{cell}__e': e, f'{cell}__u': u, f'{cell}__dall': dall, f'{cell}__FAR_masked': farE(e), f'{cell}__OFF': farE(dall),
                   f'{cell}__paths': np.array(rp)})
    okm = all((cv[c]['dall'] <= cv[c]['e']).all() and (~cv[c]['OFF'] | cv[c]['Fm']).all() for c in cv)
    say(f'ASSERT d_all <= e and far in both unmasked implies far in both masked, every picture: {okm}')
    if not okm:
        stop('the unmasked-chord asserts failed')

    if mz is not None:
        res = []
        for arm, tgx in v3.ARMS:
            tags = [str(t) for t in mz[f'{arm}__tags']]
            mine = [f'{a.prefix}{tgx}_s{n}' for n in starts]
            if tags != mine:
                stop(f'measure npz {arm} tags {tags} are not {mine}')
            F = np.concatenate([cv[c]['Fm'] for c in mine])
            e = np.concatenate([cv[c]['e'] for c in mine])
            same = bool(np.array_equal(F, np.asarray(mz[f'{arm}__cell_E'], bool)))
            d = float(np.abs(e - np.asarray(mz[f'{arm}__cell_e'], np.float64)).max())
            res.append(same and d < 1e-6)
            say(f'MEASURE NPZ CHECK {arm}: FAR masked = the measure\'s cell_E picture by picture: {same}; max |chord - cell_e| {d:.2e}')
        if not all(res):
            stop('the FAR flags differ from the measure output')
        say('MEASURE NPZ CHECK: SAME in every arm')
    else:
        say('MEASURE NPZ CHECK: not run (qgc_ test without --measure-npz)')
    say()

    three = lambda x: f'{c_(nearE(x))} / {c_(~nearE(x) & ~farE(x))} / {c_(farE(x))}'  # noqa: E731
    cls = lambda x: np.select([nearE(x), farE(x)], [0, 2], 1)  # noqa: E731
    say('TABLE 1 per arm and start: n | near in both / near in one / far in both, MASKED (E) | the same, UNMASKED | pictures whose class differs | OFF (= far in both, unmasked)')
    off = {}
    for arm, tgx in v3.ARMS:
        for n in starts + ('pooled',):
            cs = [cv[f'{a.prefix}{tgx}_s{k}'] for k in (starts if n == 'pooled' else (n,))]
            e, dall = np.concatenate([c['e'] for c in cs]), np.concatenate([c['dall'] for c in cs])
            O_ = farE(dall)
            if n != 'pooled':
                off[(arm, n)] = cs[0]
            say(f'   {arm:2s} {ROLE[arm]:17s} {("s" + str(n)) if n != "pooled" else "pooled":6s} n {len(e):5d} | masked {three(e)} | unmasked {three(dall)} | '
                f'differ {c_(cls(e) != cls(dall))} | OFF {c_(O_)}')
    say()
    say('TABLE 2 the leftover against today\'s directions, a degree beside (not a condition): among OFF pictures and among all: over today\'s line in both readers (LL_T) | '
        'in DINOv3 | in SigLIP | median share DINOv3, SigLIP')
    med = lambda x: f'{np.median(x):.4f}' if len(x) else '-'  # noqa: E731
    for arm, tgx in v3.ARMS:
        for n in starts + ('pooled',):
            cs = [cv[f'{a.prefix}{tgx}_s{k}'] for k in (starts if n == 'pooled' else (n,))]
            O_ = np.concatenate([c['OFF'] for c in cs])
            ll, d_, s_ = (np.concatenate([c[k] for c in cs]) for k in ('ll', 'ovD', 'ovS'))
            shD, shS = np.concatenate([c['sh'][DINO] for c in cs]), np.concatenate([c['sh'][SIG] for c in cs])
            say(f'   {arm:2s} {("s" + str(n)) if n != "pooled" else "pooled":6s} OFF {c_(O_):5d}: LL_T {c_(O_ & ll)} | {c_(O_ & d_)} | {c_(O_ & s_)} | {med(shD[O_])}, {med(shS[O_])}'
                f'   || all {len(O_):5d}: LL_T {c_(ll)} | {c_(d_)} | {c_(s_)} | {med(shD)}, {med(shS)}')
    say()
    say('RULE (tr_readout_v2.rule): "ahead" = more at all five starts and 4 x sum_search >= 5 x sum_control and, when the control\'s sum is 0, sum_search >= 10; '
        '"behind" the same way round; otherwise "cannot tell"')
    for lab, fn in (('OFF', lambda c: c['OFF']), ('OFF and LL_T', lambda c: c['OFF'] & c['ll'])):
        vv = [c_(fn(off[('V', n)])) for n in starts]
        for x in ('C1', 'C2'):
            xx = [c_(fn(off[(x, n)])) for n in starts]
            say(f'   {lab}: search vs {x} ({ROLE[x]}): search {vv} = {sum(vv)}, {x} {xx} = {sum(xx)} -> {tr.rule(vv, xx).upper()}')
    say()

    with open(a.out + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'wrote {a.out}.txt and {a.out}.npz (per cell: share per reader, LL_T, e, u, d_all, FAR_masked, OFF, paths)')

    def stop_places(msg):
        say(f'STOP: {msg}. The tables above stand; the places section is not printed.')
        FH[0].close()
        sys.exit(3)

    if mz is None:
        say('PLACES: no measure npz given; nothing to read')
    elif 'cell_m' not in mz.files or not all(f'{arm}__n_places' in mz.files for arm, _ in v3.ARMS):
        say('PLACES: place membership is not in the measure file')
    elif sum(int(mz[f'{arm}__n_places']) for arm, _ in v3.ARMS) == 0:
        say('PLACES: the measure names no place')
    else:
        mcell = int(mz['cell_m'])
        named = [(arm, k) for arm, _ in v3.ARMS for k in range(int(mz[f'{arm}__n_places'])) if int(mz[f'place__{arm}__{k}__support']) >= mcell]
        # every member of a named place maps to a row 1..1920 that is far in both readers (masked) in this script's own arrays, else stop
        for arm, k in named:
            pre = f'place__{arm}__{k}__'
            tags = [str(t) for t in mz[f'{arm}__tags']]
            for u_, r_ in zip(np.asarray(mz[pre + 'members_unit']).astype(int), np.asarray(mz[pre + 'members_row']).astype(int)):
                if not (1 <= r_ <= 1920):
                    stop_places(f'{arm} place {k}: member (unit {u_}, row {r_}) is outside rows 1..1920')
                if not bool(cv[tags[u_]]['Fm'][r_ - 1]):
                    stop_places(f'{arm} place {k}: member (unit {u_} = {tags[u_]}, row {r_}) is not far in both readers (masked) in this script\'s arrays: the row convention does not match')
        nall = sum(int(mz[f'{arm}__n_places']) for arm, _ in v3.ARMS)
        if not named:
            say(f'PLACES: the measure has {nall} place(s), none with support >= cell m {mcell}')
        else:
            say(f'PLACES named by the measure (support >= cell m {mcell}); members read from the measure npz; OFF = unmasked here; support rule = t3_measure_v5.support_of '
                f'(q {t3.Q}, {t3.MIN_ROUNDS} rounds) among the OFF members, at least {mcell} starts. The verdict is the measure\'s and is not re-derived.')
            say(f'   every member checked: row in 1..1920 and far in both readers (masked) in this script\'s arrays')
        if named and 'verdict_rule' in mz.files:
            say(f'   measure verdict: rule {int(mz["verdict_rule"])}: {str(mz["verdict"])}')
        for arm, k in named:
            tags = [str(t) for t in mz[f'{arm}__tags']]
            if True:
                pre = f'place__{arm}__{k}__'
                mu, mr, mrd = (np.asarray(mz[pre + x]).astype(int) for x in ('members_unit', 'members_row', 'members_round'))
                isoff = np.array([bool(cv[tags[u_]]['OFF'][r_ - 1]) for u_, r_ in zip(mu, mr)])
                isll = np.array([bool(cv[tags[u_]]['ll'][r_ - 1]) for u_, r_ in zip(mu, mr)])
                sup_off = t3.support_of(mu[isoff], mrd[isoff]) if isoff.any() else []
                exc = bool(mz[pre + 'exclusive']) if pre + 'exclusive' in mz.files else None
                say(f'   {arm} place {k}: support {int(mz[pre + "support"])} starts {[tags[u_] for u_ in np.asarray(mz[pre + "units"]).astype(int)]}; exclusive {exc}; members {len(mu)}; '
                    f'OFF {c_(isoff)}; LL_T {c_(isll)}; OFF and LL_T {c_(isoff & isll)}')
                say(f'      OFF members per start: ' + ', '.join(f'{tags[u_]} {c_(isoff & (mu == u_))} (rounds {sorted(set(mrd[isoff & (mu == u_)].tolist()))})' for u_ in np.unique(mu)))
                say(f'      support among OFF members: starts {[tags[u_] for u_ in sup_off]} -> {len(sup_off)} of the {mcell} needed: '
                    + ('HOLDS' if len(sup_off) >= mcell else 'DOES NOT HOLD'))
    say('COMPLETE')
    FH[0].close()


if __name__ == '__main__':
    main()
