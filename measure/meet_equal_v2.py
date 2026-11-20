"""Paired draws, search against the random-score loop, of equal numbers of thinned far-and-faithful SINCE pictures per start: share at works reached from at least 3 starts.
Usage (CPU, cwd = the search folder): python -B meet_equal_v2.py OUTPREFIX -> OUTPREFIX.txt and OUTPREFIX.npz, each refused if it exists."""
import importlib.util
import os
import sys
import time
import types

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
R, L, E = f'{O}/runs', f'{B}/logs', f'{O}/eval'
T3D, OO = f'{B}/tests12/t3', f'{B}/tests12/onoff/out'
R2 = f'{B}/tests12/t2/renders_v1'
EV_FILE = f'{C_}/controls_evaluate.py'
T3_FILE = f'{C_}/t3_measure_v5.py'
SCALES = f'{T3D}/out/scales_v5.npz'
CC_NPZ = f'{OO}/onoff_v3_C.npz'
FF_NPZ = f'{OO}/onoff_ff_v1.npz'
DT_NPZ = f'{OO}/onoff_dated_v2.npz'
STARTS = (6, 7, 8, 9, 10)
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
GAP, SEED, NPERM = 0.575, 20261118, 1000
POOLS = ('PRE', 'POST', 'UNDATED')
c_ = lambda x: int(np.sum(x))  # noqa: E731
FH, OUT, CHK = [None], [], []


def say(s=''):
    OUT.append(s)
    print(s, flush=True)
    if FH[0] is not None:
        FH[0].write(s + '\n')
        FH[0].flush()


def want(label, got, exp):
    CHK.append(got == exp)
    say(f'   GUARD {label}: got {got} | expected {exp} | {"same" if got == exp else "DIFFERENT"}')


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chordm(a, b):
    return np.sqrt(np.clip(2.0 - 2.0 * (a @ b.T), 0.0, None))


def unit64(a):
    a = np.asarray(a, np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def a_stats(st, pp, pw):
    """st: start label (0..4) per picture of the set; pp, pw: unique (picture, work) pairs. Returns per-work start counts and the summary."""
    uw, inv = np.unique(pw, return_inverse=True)
    key = np.unique(inv * 8 + st[pp])
    nst = np.bincount(key // 8, minlength=len(uw))
    good = nst[inv] >= 3
    return dict(uw=uw, inv=inv, nst=nst, n_works=len(uw), by=[c_(nst == k) for k in range(1, 6)], n3=c_(nst >= 3),
                pics3=len(np.unique(pp[good])), pics_reached=len(np.unique(pp)))


def a_perm(st, pp, pw, nperm, seed):
    rng = np.random.default_rng(seed)
    n3, p3 = np.empty(nperm, int), np.empty(nperm, int)
    for i in range(nperm):
        s = a_stats(rng.permutation(st), pp, pw)
        n3[i], p3[i] = s['n3'], s['pics3']
    return n3, p3


def b_stats(XD, XS, st, bars, gap):
    """XD, XS: unit float64 readings; st: start label per picture. Nearest other-start / same-start picture of the same set, links, groups."""
    n = len(st)
    DD, DS = chordm(XD, XD), chordm(XS, XS)
    same = st[:, None] == st[None, :]
    other = ~same
    sameX = same & ~np.eye(n, dtype=bool)
    inf = np.inf
    dDo, dSo = np.where(other, DD, inf).min(1) if n else np.zeros(0), np.where(other, DS, inf).min(1) if n else np.zeros(0)
    dDs, dSs = np.where(sameX, DD, inf).min(1) if n else np.zeros(0), np.where(sameX, DS, inf).min(1) if n else np.zeros(0)
    within = (DD <= bars[0]) & (DS <= bars[1])
    link = other & within
    both_o = link.any(1)
    one_o = (dDo <= bars[0]) | (dSo <= bars[1])
    gap_o = dDo <= gap
    both_s = (sameX & within).any(1)
    one_s = (dDs <= bars[0]) | (dSs <= bars[1])
    gap_s = dDs <= gap
    ncomp, lab = connected_components(sparse.csr_matrix(link), directed=False) if n else (0, np.zeros(0, int))
    key = np.unique(lab * 8 + st) if n else np.zeros(0, int)
    nst = np.bincount(key // 8, minlength=ncomp)
    size = np.bincount(lab, minlength=ncomp)
    groups = {k: (c_(nst == k), int(size[nst == k].sum())) for k in range(2, 6)}
    return dict(dDo=dDo, dSo=dSo, dDs=dDs, dSs=dSs, both_o=both_o, one_o=one_o, gap_o=gap_o, both_s=both_s, one_s=one_s, gap_s=gap_s,
                lab=lab, nst_g=nst, size_g=size, groups=groups, n_links=c_(np.triu(link, 1)))


def med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float('nan')


def main(prefix):
    t0 = time.time()
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    ev = load_mod('controls_evaluate', EV_FILE)
    t3 = load_mod('t3_measure_v5', T3_FILE)
    say(f'meet_equal_v2.py. Counts only; numpy {np.__version__}')
    Cz, Fz, Dz = (np.load(f, allow_pickle=True) for f in (CC_NPZ, FF_NPZ, DT_NPZ))
    P, T, Pz = ev.references(O)
    ids, q = ev.startids(O), ev.qfun(O)
    ns = types.SimpleNamespace(corpus='real', root=O, rows=1921, rounds=60, per_round=32, scales=SCALES,
                               run_tmpl=f'{O}/runs/{{tag}}_committee_euclidean_min_seedrow/all_candidates.npz', exact_tmpl=f'{O}/eval/{{tag}}_emb.npz',
                               acc_prior=f'{R2}/prior256_v1', acc_noise=f'{R2}/noise160_v1', acc_aim_prior=None, acc_aim_noise=None)
    C, RP, idx = t3.load_common(ns)
    sc = t3.load_scales(ns)
    cal = sc['cal'][99]
    BAR = np.asarray(cal['theta'] * cal['med'], np.float64)
    r15 = np.asarray(sc['r_15'], np.float64)
    pools = {'PRE': C['dated'] & (C['year'] < 1900), 'POST': C['dated'] & (C['year'] >= 1900), 'UNDATED': ~C['dated']}
    post_rows = np.flatnonzero(pools['POST'])
    say(f'manifold line (bars) D {BAR[0]:.6f} / S {BAR[1]:.6f}; r_15 D {r15[0]:.6f} / S {r15[1]:.6f}; r_1 {np.round(sc["r_1"], 6)}; '
        f'pools PRE {c_(pools["PRE"])}, POST {c_(pools["POST"])}, UNDATED {c_(pools["UNDATED"])}')

    def post_nearest(Qs):
        """t3.nearest's arithmetic (unit float32, X @ Cn.T in chunks of 512, top = max), plus the row of the max."""
        out = {}
        for nme in t3.OBS:
            X, Cn = t3.unit(Qs[nme]), C['E'][nme][post_rows]
            rid, ch = np.empty(len(X), int), np.empty(len(X))
            for i in range(0, len(X), 512):
                S = X[i:i + 512] @ Cn.T
                a = S.argmax(1)
                rid[i:i + 512] = post_rows[a]
                ch[i:i + 512] = t3.chord(S[np.arange(len(a)), a])
            out[nme] = (rid, ch)
        return out

    cells = {}
    for arm, tg, _ in ARMS:
        for n in STARTS:
            tag = f'qgc_{tg}_s{n}'
            d = f'{R}/{tag}_committee_euclidean_min_seedrow'
            r = ev.load(f'{tag}={d},{E}/{tag}_emb.npz,{L}/{tag}.out', P, T, ids, q, conf=True)
            z = r['z']
            m = z['iteration'] >= 0
            FF = np.asarray(r['F'], bool) & np.asarray(r['ok_exact'], bool)
            lc = t3.load_cell(tag, ns.run_tmpl, ns.exact_tmpl, 1921, C, layout=(60, 32), arm=arm, vtag=f'qgc_ra_s{n}')
            assert lc['valid'], lc['why']
            paths = np.array([str(p) for p in np.asarray(z['paths'])[m]])
            assert np.array_equal(np.array([str(p) for p in lc['paths'][1:]]), paths), f'{tag}: run and exact rows differ'
            nn = len(lc['src'])
            pc = {}
            for k in POOLS:
                dd = t3.nearest(lc['E'], C['E'], [np.zeros(0, int)] * nn, pools[k])
                pc[k] = np.stack([dd[o][:, 0] for o in t3.OBS], 1)
            anyb = lambda x: np.any(x <= BAR, axis=1)  # noqa: E731
            pre = anyb(pc['PRE'])
            since = ~pre & anyb(pc['POST'])
            und = ~pre & ~since & anyb(pc['UNDATED'])
            cls = np.select([pre, since, und, ~pre & ~since & ~und], [0, 1, 2, 3], -1)
            pn = post_nearest(lc['E'])
            pch = np.stack([pn[o][1] for o in t3.OBS], 1)
            same = (np.array_equal(paths, Cz[f'{tag}__paths']) and np.array_equal(FF, Fz[f'{tag}__FF']) and np.array_equal(cls, Dz[f'{tag}__class_one']))
            want(f'{tag} picture by picture: paths, FF (ev.load), class (t3.nearest, 3 pools) = saved arrays', same, True)
            gp = float(np.max(np.abs(pch - pc['POST'])))
            gs = float(np.max(np.abs(pc['POST'] - Dz[f'{tag}__pool_POST'])))
            want(f'{tag} nearest-POST chord here vs t3.nearest (max abs diff <= 1e-6) and t3.nearest vs saved pool_POST (<= 1e-6)', gp <= 1e-6 and gs <= 1e-6, True)
            say(f'      (diffs {gp:.2e}, {gs:.2e})')
            cells[tag] = dict(arm=arm, FF=FF, cls=cls, X={o: lc['E'][o] for o in t3.OBS}, rnd=lc['rnd'], row=lc['row'],
                              pid=np.stack([pn[o][0] for o in t3.OBS], 1), pch=pch)
    if not all(CHK):
        say('STOP: the guard did not reproduce; no new number is computed.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x', encoding='utf-8').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say(f'   GUARD PASSED ({len(CHK)} checks). {time.time() - t0:.0f} s')

    G = {}
    for arm, tg, _ in ARMS:
        cs = [cells[f'qgc_{tg}_s{n}'] for n in STARTS]
        cat = lambda k: np.concatenate([c[k] for c in cs])  # noqa: E731
        G[arm] = dict(FF=cat('FF'), cls=cat('cls'), st=np.concatenate([np.full(len(c['FF']), i) for i, c in enumerate(cs)]),
                      X={o: np.concatenate([c['X'][o] for c in cs]) for o in t3.OBS}, rnd=cat('rnd'), row=cat('row'), pid=cat('pid'), pch=cat('pch'))
    meet_equal(prefix, G, BAR, t0)


NEQ, NDRAW, RSEED = 113, 1000, 20261119


def either_pairs(pid, pch, bar):
    """pid, pch: (n, 2) nearest-POST work row and chord per observer (D, S). Returns unique (picture, work) pairs within each line, union."""
    ks = []
    for oi in range(2):
        ok = pch[:, oi] <= bar[oi]
        ks.append(np.flatnonzero(ok) * 200000 + pid[ok, oi])
    pe = np.unique(np.concatenate(ks))
    return pe // 200000, pe % 200000


def thin(XD, st, rnd, gap):
    """Within each start, in painted order (rnd, then stored order), keep a picture when its chord is > gap to every kept one."""
    keep = np.zeros(len(st), bool)
    for s in np.unique(st):
        js = np.flatnonzero(st == s)
        js = js[np.lexsort((js, rnd[js]))]
        kept = []
        for j in js:
            if not kept or chordm(XD[j:j + 1], XD[kept]).min() > gap:
                kept.append(j)
        keep[kept] = True
    return keep


def draws(st, pp, pw, pool, m, ndraw, seed, nstart=5):
    """Equal samples: m pictures per start from pool (bool per picture), without replacement. Share at works from >= 3 starts, such works."""
    rng = np.random.default_rng(seed)
    by = [np.flatnonzero((st == s) & pool) for s in range(nstart)]
    assert all(len(b) >= m for b in by), [len(b) for b in by]
    share, n3 = np.empty(ndraw), np.empty(ndraw, int)
    for i in range(ndraw):
        sel = np.zeros(len(st), bool)
        for b in by:
            sel[rng.choice(b, m, replace=False)] = True
        k = sel[pp]
        a = a_stats(st, pp[k], pw[k])
        share[i], n3[i] = a['pics3'] / (nstart * m), a['n3']
    return share, n3


def ci(x):
    return float(np.median(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def meet_equal(prefix, G, BAR, t0):
    sv, Q = {}, {}
    for arm, _, w in ARMS:
        g = G[arm]
        k = np.flatnonzero(g['FF'] & (g['cls'] == 1))
        st = g['st'][k]
        pp, pw = either_pairs(g['pid'][k], g['pch'][k], BAR)
        keep = thin(unit64(g['X']['dinov3-l'][k]), st, g['rnd'][k], GAP)
        Q[arm] = dict(st=st, pp=pp, pw=pw, keep=keep, n=len(k))
        say(f'   {arm} {w}: SINCE per start {[c_(st == s) for s in range(5)]}; thinned (> {GAP} DINOv3 to every earlier kept, painted order) '
            f'{[c_((st == s) & keep) for s in range(5)]}')
    mt = min(c_((Q[a]['st'] == s) & Q[a]['keep']) for a in ('V', 'C1') for s in range(5))
    R_ = {}
    for cond, m in (('thinned', mt),):
        say()
        say(f'{cond.upper()}: {m} pictures per start ({5 * m} per loop), {NDRAW} draws, rng {RSEED}; reach = either observer, works counted within the sample')
        for arm, _, w in [x for x in ARMS if x[0] in ('V', 'C1')]:
            q = Q[arm]
            pool = np.ones(q['n'], bool) if cond == 'all' else q['keep']
            sh, n3 = draws(q['st'], q['pp'], q['pw'], pool, m, NDRAW, RSEED)
            R_[(cond, arm)] = (sh, n3)
            sv[f'{cond}__{arm}__share'], sv[f'{cond}__{arm}__n3'] = sh, n3
            a, b = ci(sh), ci(n3)
            say(f'   {arm} {w:18s} share at works from >= 3 starts: median {100 * a[0]:.1f}%, 95% {100 * a[1]:.1f} to {100 * a[2]:.1f}% | '
                f'works from >= 3 starts: median {b[0]:.0f}, 95% {b[1]:.0f} to {b[2]:.0f}')
        for arm in ('C1',):
            sv_, sc_ = R_[(cond, 'V')][0], R_[(cond, arm)][0]
            nv_, nc_ = R_[(cond, 'V')][1], R_[(cond, arm)][1]
            say(f'   V vs {arm}: draw i against draw i (PAIRED: one random stream for both loops): share V > {arm} in {np.mean(sv_ > sc_):.3f} of draws; works V > {arm} in {np.mean(nv_ > nc_):.3f}')
    sv['m_thinned'] = np.array(mt)
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'saved {prefix}.npz; done {time.time() - t0:.0f} s')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
