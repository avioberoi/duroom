"""Do the pictures the search found from different starts land in the same places? A shared real works, B near pictures across starts, C places.
Usage: python -B ov_overlap_v1.py OUTPREFIX (cwd = the search folder)"""
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
c_ = lambda x: int(np.sum(x))
FH, OUT, CHK = [None], [], []


def say(s=''):
    OUT.append(s)
    print(s, flush=True)
    if FH[0] is not None:
        FH[0].write(s + '\n')
        FH[0].flush()


def want(label, got, exp):
    CHK.append(got == exp)
    say(f'   CHECK {label}: got {got} | expected {exp} | {"same" if got == exp else "DIFFERENT"}')


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
    say(f'ov_overlap_v1.py, numpy {np.__version__}')
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
            anyb = lambda x: np.any(x <= BAR, axis=1)
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
        say('STOP: a check against the saved arrays failed; no new number is computed.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x', encoding='utf-8').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say(f'   CHECKS PASSED ({len(CHK)} checks). {time.time() - t0:.0f} s')

    G = {}
    for arm, tg, _ in ARMS:
        cs = [cells[f'qgc_{tg}_s{n}'] for n in STARTS]
        cat = lambda k: np.concatenate([c[k] for c in cs])
        G[arm] = dict(FF=cat('FF'), cls=cat('cls'), st=np.concatenate([np.full(len(c['FF']), i) for i, c in enumerate(cs)]),
                      X={o: np.concatenate([c['X'][o] for c in cs]) for o in t3.OBS}, rnd=cat('rnd'), row=cat('row'), pid=cat('pid'), pch=cat('pch'))
    sv = {}
    res = {}
    for arm, _, _ in ARMS:
        g = G[arm]
        k = np.flatnonzero(g['FF'] & (g['cls'] == 1))
        st = g['st'][k]
        pr = {}
        for oi, o in enumerate(('D', 'S')):
            ok = g['pch'][k, oi] <= BAR[oi]
            pr[o] = (np.flatnonzero(ok), g['pid'][k, oi][ok])
        pe = np.unique(np.concatenate([pr['D'][0] * 200000 + pr['D'][1], pr['S'][0] * 200000 + pr['S'][1]]))
        pr['either'] = (pe // 200000, pe % 200000)
        res[arm] = {'A': {o: a_stats(st, *pr[o]) for o in pr}, 'k': k, 'st': st, 'pr': pr}
        res[arm]['A_perm'] = a_perm(st, *pr['either'], NPERM, SEED)
        sv[f'{arm}__since_idx'], sv[f'{arm}__since_start'] = k, st
        sv[f'{arm}__since_post_row'], sv[f'{arm}__since_post_chord'] = g['pid'][k], g['pch'][k]
        sv[f'{arm}__A_perm_n3'], sv[f'{arm}__A_perm_pics3'] = res[arm]['A_perm']
    for arm, _, _ in ARMS:
        g = G[arm]
        for setn, mask in (('FF', g['FF']), ('OFF', g['FF'] & (g['cls'] == 3))):
            k = np.flatnonzero(mask)
            b = b_stats(unit64(g['X']['dinov3-l'][k]), unit64(g['X']['siglip-so400m'][k]), g['st'][k], BAR, GAP)
            res[arm][f'B_{setn}'] = b
            for kk in ('dDo', 'dSo', 'dDs', 'dSs', 'lab'):
                sv[f'{arm}__{setn}__{kk}'] = b[kk]
            sv[f'{arm}__{setn}__idx'] = k
    Pe = {o: unit64(RP['P']['E'][o]) for o in t3.OBS}
    pa = RP['P']['artist']
    pscale = {}
    for o in t3.OBS:
        D = chordm(Pe[o], Pe[o])
        np.fill_diagonal(D, np.inf)
        plain = D.min(1)
        D[pa[:, None] == pa[None, :]] = np.inf
        pscale[o] = (len(plain), med(plain), med(D.min(1)))
    for arm, _, _ in ARMS:
        g = G[arm]
        for setn, mask in (('OFF', g['FF'] & (g['cls'] == 3)), ('SINCE', g['FF'] & (g['cls'] == 1))):
            try:
                A_ = dict(X=g['X'], unit=g['st'], rnd=g['rnd'], row=g['row'], elig=mask)
                el, Tm = t3.arm_pairs(A_, r15)
                pl = t3.places_of(A_, el, Tm)
                res[arm][f'C_{setn}'] = pl
                sv[f'{arm}__C_{setn}__support'] = np.array([p['support'] for p in pl], int)
            except Exception as e:
                res[arm][f'C_{setn}'] = f'C not done: {type(e).__name__}: {e}'
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'saved {prefix}.npz; {time.time() - t0:.0f} s')

    say()
    say('A. SHARED REAL WORKS. Pictures: FF and SINCE. A picture reaches the work dated 1900 on that is nearest to it in an observer, when that chord is within')
    say('   the line of that observer; "either" = the union over DINOv3 and SigLIP (a picture reaches 1 or 2 works). Starts per work = distinct starts of its pictures.')
    for arm, _, w in ARMS:
        for o in ('either', 'D', 'S'):
            a = res[arm]['A'][o]
            say(f'   {arm} {w:18s} {o:6s}: pictures {len(res[arm]["k"])}, reaching {a["pics_reached"]}; distinct works {a["n_works"]}; works by starts 1/2/3/4/5 '
                f'{a["by"]}; works from >= 3 starts {a["n3"]}; pictures at a work from >= 3 starts {a["pics3"]}')
        n3, p3 = res[arm]['A_perm']
        say(f'   {arm} chance (start labels shuffled {NPERM}x, rng {SEED}, either): works from >= 3 starts mean {n3.mean():.1f}, 95% range {np.percentile(n3, 2.5):.0f} to '
            f'{np.percentile(n3, 97.5):.0f} (observed {res[arm]["A"]["either"]["n3"]}); pictures at such works mean {p3.mean():.1f}, 95% range {np.percentile(p3, 2.5):.0f} to '
            f'{np.percentile(p3, 97.5):.0f} (observed {res[arm]["A"]["either"]["pics3"]}); share of shuffles with works >= observed {np.mean(n3 >= res[arm]["A"]["either"]["n3"]):.3f}')
    for arm, _, w in ARMS:
        a, st = res[arm]['A']['either'], res[arm]['st']
        pp, pw = res[arm]['pr']['either']
        per = np.zeros((len(a['uw']), 5), int)
        np.add.at(per, (a['inv'], st[pp]), 1)
        order = np.lexsort((a['uw'], -per.sum(1), -a['nst']))[:15]
        say(f'   {arm} {w}: the 15 works reached from the most starts (either observer); pictures per start 6/7/8/9/10:')
        for j in order:
            rw = int(a['uw'][j])
            say(f'      row {rw} id {C["ids"][rw]} | {C["artist"][rw]} | {C["title"][rw]} | {int(C["year"][rw])} | starts {a["nst"][j]} | {per[j].tolist()} = {per[j].sum()}')
    say()
    say(f'B. PICTURES NEAR EACH OTHER ACROSS STARTS (exact readings; line D {BAR[0]:.6f} / S {BAR[1]:.6f}; gap {GAP} in DINOv3)')
    for o in t3.OBS:
        say(f'   scale: P ({pscale[o][0]} reference repaints) nearest-neighbour chord {o}: median plain {pscale[o][1]:.6f}; to another artist {pscale[o][2]:.6f}')
    for arm, _, w in ARMS:
        for setn in ('FF', 'OFF'):
            b = res[arm][f'B_{setn}']
            n = len(b['dDo'])
            say(f'   {arm} {w:18s} {setn:3s} n {n}: nearest OTHER-start picture median D {med(b["dDo"]):.4f} S {med(b["dSo"]):.4f}; SAME-start median D {med(b["dDs"]):.4f} '
                f'S {med(b["dSs"]):.4f}')
            say(f'      with an other-start picture: D chord <= {GAP} {c_(b["gap_o"])}; within the line in at least one observer {c_(b["one_o"])}; same picture within the line '
                f'in both {c_(b["both_o"])} | same-start: {c_(b["gap_s"])} / {c_(b["one_s"])} / {c_(b["both_s"])}')
            gs = b['groups']
            say(f'      groups (links: other start, within the line in both; {b["n_links"]} links): by starts 2/3/4/5: groups '
                f'{[gs[k][0] for k in range(2, 6)]}, pictures {[gs[k][1] for k in range(2, 6)]}; largest group {int(b["size_g"].max()) if len(b["size_g"]) else 0} pictures')
    say()
    say(f'C. PLACES (t3_measure_v5 greedy: ball r_15 D {r15[0]:.6f} / S {r15[1]:.6f} in both; a start supports with >= 5 pictures from >= 2 rounds; '
        f'listed down to support 2; a place needs >= 3)')
    for arm, _, w in ARMS:
        for setn in ('OFF', 'SINCE'):
            pl = res[arm][f'C_{setn}']
            nel = c_(G[arm]['FF'] & (G[arm]['cls'] == (3 if setn == 'OFF' else 1)))
            if isinstance(pl, str):
                say(f'   {arm} {w:18s} {setn:5s} (n {nel}): {pl}')
                continue
            by = [sum(1 for p in pl if p['support'] == k) for k in range(2, 6)]
            say(f'   {arm} {w:18s} {setn:5s} (n {nel}): places by support 2/3/4/5 {by}; places with >= 3 starts {sum(by[1:])}')
            for p in pl:
                if p['support'] >= 3:
                    say(f'      support {p["support"]}: starts {[STARTS[u] for u in p["units"]]}, members {len(p["members"])}')
    say(f'done {time.time() - t0:.0f} s')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
