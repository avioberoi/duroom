"""Pairwise-separated counts of the lead-row and off-row pictures per cell (search, walk, random-score loop, no-step loop): greedy thinning at DINOv3 chord > 0.575 in generation order and in reverse, with rule words.
Usage: python -B lead_spaced_v1.py OUTPREFIX --run (reads onoff_walk_v2_rtx.npz and the run files; refuses to overwrite) | --selftest (synthetic points only)"""
import argparse
import importlib.util
import math
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
OO = f'{B}/tests12/onoff/out'
FLAGS_NPZ = f'{OO}/onoff_walk_v2_rtx.npz'
TR_FILE = os.path.dirname(os.path.abspath(__file__)) + '/tr_readout_v2.py'
RUN_TMPL = f'{B}/sphere/runs/{{tag}}_committee_euclidean_min_seedrow/all_candidates.npz'
EMB = 'emb_dinov3-l'
DELTA = 0.575
NROWS, NCAND = 1921, 1920
ARMS = ('V', 'W', 'C1', 'C2')
ARM_NAME = {'V': 'search', 'W': 'single walk', 'C1': 'random-score loop', 'C2': 'no-step loop'}
TAGF = {'V': 'qgc_ra_s{}', 'W': 'qgw_walk_s{}', 'C1': 'qgc_pin_s{}', 'C2': 'qgc_c2_s{}'}
STARTS = (6, 7, 8, 9, 10)
POPS = ('LEAD', 'OFF')
POP_LAB = {'LEAD': 'LEAD: FF and SINCE (one reader, class_one == 1) and LL',
           'OFF': 'OFF companion: FF and NONE (one reader, class_one == 3); LL not imposed, reported beside'}
ORDERS = ('forward', 'reverse')
PAIRS = (('V', 'W'), ('V', 'C1'), ('V', 'C2'), ('W', 'C1'), ('W', 'C2'))
HEADER = ("A count of pairwise-separated qualifying pictures at one DINOv3 scale under a greedy rule; not a maximum packing, not a count of distinct regions; "
          "the five starts are not to be summed into a number of regions (the same place may recur across starts). "
          "Delta 0.575 is the loop's own spacing bar.")
FH, OUT, PREFIX = [None], [], [None]


def say(s=''):
    OUT.append(s)
    print(s, flush=True)
    if FH[0] is not None:
        FH[0].write(s + '\n')
        FH[0].flush()


def stop(msg):
    """--run: write OUTPREFIX_STOP.txt (everything printed so far + the reason) and exit 3. --selftest: print and exit 3, nothing written."""
    say(f'STOP: {msg}.')
    if PREFIX[0] is not None:
        FH[0].close()
        FH[0] = None
        with open(PREFIX[0] + '_STOP.txt', 'x') as fh:
            fh.write('\n'.join(OUT) + '\n')
    sys.exit(3)


def need(cond, msg):
    if not cond:
        stop(msg)


def has_qgt(s):
    return 'qgt_' in str(s)


def nq(*ss):
    for s in ss:
        need(not has_qgt(s), f'refused: a path or tag containing qgt_ ({s})')


def tag_of(arm, n):
    t = TAGF[arm].format(n)
    nq(t)
    return t


def chord(c):
    """chord distance between unit vectors from their cosine: sqrt(max(0, 2 - 2 cos)), float64"""
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.asarray(c, np.float64)))


def thin(X, order, delta=DELTA):
    """Greedy thinning of unit rows X visited in `order`: a row is kept only if its chord to every kept row is strictly > delta; returns kept indices."""
    kept = []
    for j in order:
        j = int(j)
        if not kept or chord(X[kept] @ X[j]).min() > delta:
            kept.append(j)
    return np.asarray(kept, dtype=np.int64)


def eligible(FF, cls, LL, pop):
    if pop == 'LEAD':
        return FF & (cls == 1) & LL
    if pop == 'OFF':
        return FF & (cls == 3)
    raise ValueError(pop)


def unit_rows(E):
    E = np.asarray(E, np.float64)
    return E / np.linalg.norm(E, axis=1)[:, None]


def frac(a, b):
    return '-' if b == 0 else f'{a / b:.3f}'


def load_mod(name, path):
    nq(path)
    need(os.path.exists(path), f'{path} is missing')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_thin(X, order, delta):
    """independent pure-Python greedy thinning (no numpy dot), for comparison with thin()"""
    kept = []
    for j in order:
        j = int(j)
        ok = True
        for k in kept:
            cs = 0.0
            for a, b in zip(X[k].tolist(), X[j].tolist()):
                cs += a * b
            if math.sqrt(max(0.0, 2.0 - 2.0 * cs)) <= delta:
                ok = False
                break
        if ok:
            kept.append(j)
    return kept


def circle(deg, basis):
    r = np.radians(np.asarray(deg, np.float64))
    return unit_rows(np.stack([np.cos(r), np.sin(r)], 1) @ basis)


def selftest():
    rng = np.random.default_rng(20261115)
    basis = np.linalg.qr(rng.standard_normal((16, 16)))[0][:, :2].T     # two orthonormal directions in 16 dimensions
    th = math.degrees(2.0 * math.asin(DELTA / 2.0))
    say(f'   threshold angle for chord {DELTA}: {th:.4f} degrees (kept only if the angle to every kept one is larger)')
    need(30.0 < th < 40.0, 'selftest: threshold angle not between 30 and 40 degrees; the planted margins would not hold')

    # (1) circle at known angles; by hand: forward keeps 0, 40, 80, 120 (rows 0, 2, 4, 5); reverse keeps 120, 80, 40, 0 (rows 5, 4, 2, 0)
    deg = [0, 10, 40, 50, 80, 120]
    X = circle(deg, basis)
    need(abs(float(chord(X[0] @ X[2])) - 2.0 * math.sin(math.radians(20.0))) < 1e-12, 'selftest (1): chord of a 40-degree pair is not 2 sin(20 deg)')
    fw, rv = thin(X, np.arange(6)), thin(X, np.arange(6)[::-1])
    need(fw.tolist() == [0, 2, 4, 5], f'selftest (1): forward kept {fw.tolist()}, want [0, 2, 4, 5]')
    need(rv.tolist() == [5, 4, 2, 0], f'selftest (1): reverse kept {rv.tolist()}, want [5, 4, 2, 0]')
    say(f'   (1) circle {deg} degrees: forward kept rows {fw.tolist()} (4), reverse {rv.tolist()} (4): as by hand')

    # (2) strictness: a pair whose chord equals the threshold float is NOT kept; one ulp below the pair's chord it is kept
    cs = 1.0 - DELTA * DELTA / 2.0
    Y = circle([0.0, th], basis)                                          # a rotated pair: the cosine is computed with rounding
    for order in ([0, 1], [1, 0]):
        c = float(chord(Y[[order[0]]] @ Y[order[1]])[0])                  # the same expression and shapes thin() uses at the second visit
        a1, a2 = thin(Y, order, delta=c), thin(Y, order, delta=float(np.nextafter(c, 0.0)))
        need(a1.tolist() == [order[0]], f'selftest (2): order {order}: chord equal to delta ({c!r}) was kept: {a1.tolist()}')
        need(a2.tolist() == order, f'selftest (2): order {order}: chord one ulp above delta was not kept: {a2.tolist()}')
        say(f'   (2) pair at the threshold angle, order {order}: own chord {c!r}; delta = own chord keeps {a1.tolist()}, delta one ulp lower keeps {a2.tolist()}')
    found = None
    lo = hi = cs
    cands = [cs]
    for _ in range(64):
        lo, hi = float(np.nextafter(lo, -np.inf)), float(np.nextafter(hi, np.inf))
        cands += [lo, hi]
    for cc in cands:
        Z = np.array([[1.0, 0.0, 0.0], [cc, math.sqrt(1.0 - cc * cc), 0.0]])   # the dot of these two rows is exactly cc
        if float(chord(Z[[0]] @ Z[1])[0]) == DELTA:
            found = Z
            break
    if found is None:
        say(f'   (2) no cosine within 64 ulps of {cs!r} gives a chord of exactly {DELTA} in float64; strictness rests on the pair\'s own chord above')
    else:
        a1, a2 = thin(found, [0, 1]), thin(found, [1, 0])
        need(a1.tolist() == [0] and a2.tolist() == [1], f'selftest (2): a pair with chord exactly {DELTA} was kept ({a1.tolist()}, {a2.tolist()})')
        say(f'   (2) a pair with chord exactly {DELTA!r} (float64) at the run\'s own delta: forward keeps {a1.tolist()}, reverse {a2.tolist()} (not kept)')

    # (3) empty eligible set gives 0, both orders, through eligible() as well
    e0 = np.flatnonzero(eligible(np.zeros(6, bool), np.ones(6, int), np.ones(6, bool), 'LEAD')).astype(np.int64)
    k3f, k3r = thin(X, e0), thin(X, e0[::-1])
    need(len(e0) == 0 and len(k3f) == 0 and len(k3r) == 0 and k3f.dtype == np.int64, 'selftest (3): empty eligible set did not give 0')
    need(X[k3f].shape == (0, 16), 'selftest (3): an empty kept list does not index as an empty array')
    say('   (3) empty eligible set: 0 forward, 0 reverse')

    # (4) all-identical points give 1 (the first visited)
    v = unit_rows(rng.standard_normal((1, 16)))
    Xi = np.repeat(v, 5, axis=0)
    k4f, k4r = thin(Xi, np.arange(5)), thin(Xi, np.arange(5)[::-1])
    need(k4f.tolist() == [0] and k4r.tolist() == [4], f'selftest (4): identical points kept {k4f.tolist()} / {k4r.tolist()}')
    say(f'   (4) five identical points: forward kept {k4f.tolist()}, reverse {k4r.tolist()} (1 each)')

    # (5) order matters: angles 30, 0, 60 degrees in that generation order; forward keeps 30 only (1); reverse keeps 60 then 0 (2)
    X5 = circle([30, 0, 60], basis)
    k5f, k5r = thin(X5, np.arange(3)), thin(X5, np.arange(3)[::-1])
    need(k5f.tolist() == [0] and k5r.tolist() == [2, 1], f'selftest (5): kept {k5f.tolist()} / {k5r.tolist()}, want [0] / [2, 1]')
    say(f'   (5) circle [30, 0, 60] degrees: forward kept {k5f.tolist()} (1), reverse {k5r.tolist()} (2): orders differ as by hand')

    # (6) populations on planted flags
    FF = np.array([1, 1, 1, 1, 0, 1], bool)
    cl = np.array([1, 1, 3, 0, 1, 3])
    LL = np.array([1, 0, 1, 1, 1, 0], bool)
    need(eligible(FF, cl, LL, 'LEAD').tolist() == [True, False, False, False, False, False], 'selftest (6): LEAD mask')
    need(eligible(FF, cl, LL, 'OFF').tolist() == [False, False, True, False, False, True], 'selftest (6): OFF mask (LL must not be imposed)')
    say('   (6) planted flags: LEAD = FF & class_one == 1 & LL, OFF = FF & class_one == 3 (LL not imposed): as by hand')

    # (7) random clustered points at the run's delta: thin() equals an independent pure-Python loop; kept pairwise > delta; each rejected has an earlier kept one within
    cent = unit_rows(rng.standard_normal((6, 16)))
    P = unit_rows(cent[rng.integers(0, 6, 300)] + 0.12 * rng.standard_normal((300, 16)))
    el = np.flatnonzero(rng.random(300) < 0.6).astype(np.int64)
    got = {}
    for name, seq in (('forward', el), ('reverse', el[::-1])):
        k = thin(P, seq)
        need(k.tolist() == ref_thin(P, seq, DELTA), f'selftest (7): {name}: thin() differs from the pure-Python loop')
        need(1 < len(k) < len(seq), f'selftest (7): {name}: trivial case ({len(k)} of {len(seq)})')
        D = chord(P[k] @ P[k].T)
        np.fill_diagonal(D, np.inf)
        need(float(D.min()) > DELTA - 1e-12, f'selftest (7): {name}: two kept points within delta')
        pos = {int(j): i for i, j in enumerate(seq)}
        ks = set(k.tolist())
        for j in seq:
            if int(j) not in ks:
                earlier = [kk for kk in k.tolist() if pos[kk] < pos[int(j)]]
                need(len(earlier) > 0 and float(chord(P[earlier] @ P[j]).min()) <= DELTA + 1e-12, f'selftest (7): {name}: row {int(j)} rejected without a near kept one')
        got[name] = len(k)
    say(f'   (7) 300 random clustered points, {len(el)} eligible: kept forward {got["forward"]}, reverse {got["reverse"]}; '
        'equal to an independent pure-Python loop; kept pairwise > delta; every rejected one within delta of an earlier kept one')

    # (8) the qgt_ guard and the tags
    need(has_qgt('/x/qgt_ra_s6') and not any(has_qgt(RUN_TMPL.format(tag=TAGF[a].format(n))) for a in ARMS for n in STARTS), 'selftest (8): qgt_ guard')
    need([TAGF[a].format(6) for a in ARMS] == ['qgc_ra_s6', 'qgw_walk_s6', 'qgc_pin_s6', 'qgc_c2_s6'], 'selftest (8): tags')
    say('   (8) qgt_ guard refuses a qgt_ path and passes the 20 run paths; tags V qgc_ra_sN, W qgw_walk_sN, C1 qgc_pin_sN, C2 qgc_c2_sN')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    if a.selftest == a.run:
        print('give exactly one of --selftest or --run')
        sys.exit(2)
    if a.selftest:
        say('lead_spaced_v1.py; mode --selftest: synthetic points only; no data file is opened; nothing is written'
            + (f' (OUTPREFIX {a.out} ignored)' if a.out else ''))
        selftest()
        say('SELFTEST PASSED.')
        return
    if not a.out:
        print('--run needs OUTPREFIX')
        sys.exit(2)
    prefix = a.out
    if has_qgt(prefix):
        print(f'refused: an output prefix containing qgt_ ({prefix})')
        sys.exit(3)
    for ext in ('.txt', '.npz', '_STOP.txt'):
        if os.path.exists(prefix + ext):
            print(f'{prefix + ext} exists: refusing to overwrite')
            sys.exit(2)
    FH[0] = open(prefix + '.txt', 'x')
    PREFIX[0] = prefix
    say(HEADER)
    say()
    say('lead_spaced_v1.py; mode --run. Counts only; no picture is opened. Delta 0.575 (strictly greater keeps); chord = sqrt(max(0, 2 - 2 cos)) on unit DINOv3 readings (float64); empty start list.')
    say()
    say('SELFTEST (synthetic points)')
    selftest()
    say('   SELFTEST PASSED.')
    say()

    nq(FLAGS_NPZ)
    need(os.path.exists(FLAGS_NPZ), f'{FLAGS_NPZ} is missing')
    say(f'saved flags: {FLAGS_NPZ}')
    Fz = np.load(FLAGS_NPZ, allow_pickle=True)
    FL = {}
    for arm in ARMS:
        for n in STARTS:
            t = tag_of(arm, n)
            for k in ('FF', 'LL', 'class_one', 'paths'):
                need(f'{t}__{k}' in Fz.files, f'{t}__{k} not in the saved flags')
            FF, LL, cl = np.asarray(Fz[f'{t}__FF']), np.asarray(Fz[f'{t}__LL']), np.asarray(Fz[f'{t}__class_one'])
            pa = np.array([str(p) for p in Fz[f'{t}__paths']])
            need(FF.dtype == bool and LL.dtype == bool, f'{t}: FF / LL are not boolean ({FF.dtype}, {LL.dtype})')
            need(FF.shape == LL.shape == cl.shape == pa.shape == (NCAND,), f'{t}: saved flag shapes {FF.shape} {LL.shape} {cl.shape} {pa.shape}, want ({NCAND},)')
            need(set(np.unique(cl).tolist()) <= {0, 1, 2, 3}, f'{t}: class_one values {np.unique(cl).tolist()} outside 0..3')
            need(len(set(pa.tolist())) == NCAND, f'{t}: saved paths are not unique')
            nq(*pa[:1])
            FL[t] = dict(FF=FF, LL=LL, cls=cl, paths=pa)
    say()

    tr = load_mod('tr_readout_v2', TR_FILE)
    need(tr.rule([6] * 5, [5] * 5) == 'cannot tell' and tr.rule([20] * 5, [1] * 5) == 'ahead' and tr.rule([1] * 5, [20] * 5) == 'behind'
         and tr.rule([2] * 5, [0] * 5) == 'ahead' and tr.rule([2, 2, 2, 2, 1], [0] * 5) == 'cannot tell', 'rule self-test')
    say(f'rule: {TR_FILE}; rule self-test passed')
    say()

    say(f'RUN FILES ({EMB}, rows with iteration >= 0; start row first and excluded; paths equal to the saved ones; finite, non-zero; made unit length)')
    X = {}
    for arm in ARMS:
        for n in STARTS:
            t = tag_of(arm, n)
            f = RUN_TMPL.format(tag=t)
            nq(f)
            need(os.path.exists(f), f'{f} is missing')
            z = np.load(f, allow_pickle=True)
            for k in ('paths', 'iteration', EMB):
                need(k in z.files, f'{t}: {k} not in the run file')
            it = np.asarray(z['iteration'])
            need(it.shape == (NROWS,), f'{t}: {it.shape[0] if it.ndim else it} rows, want {NROWS}')
            m = it >= 0
            need(int((~m).sum()) == 1 and int(m.sum()) == NCAND and bool(it[0] < 0), f'{t}: not one start row first and {NCAND} candidates')
            need(bool(np.all(np.diff(it[m]) >= 0)), f'{t}: iteration is not non-decreasing over the candidates (not in generation order)')
            rp = np.array([str(p) for p in np.asarray(z['paths'])[m]])
            need(np.array_equal(rp, FL[t]['paths']), f'{t}: run-file paths (iteration >= 0) differ from the saved paths')
            E = np.asarray(z[EMB], np.float64)
            need(E.ndim == 2 and E.shape[0] == NROWS, f'{t}: {EMB} shape {E.shape}')
            E = E[m]
            nrm = np.linalg.norm(E, axis=1)
            need(bool(np.isfinite(E).all()) and bool(np.isfinite(nrm).all()) and bool((nrm > 0).all()), f'{t}: a non-finite or zero DINOv3 reading')
            X[t] = E / nrm[:, None]
            say(f'   {t:13s} | {NROWS} rows, {NCAND} candidates, dim {E.shape[1]} | paths = saved | norms {nrm.min():.4f} to {nrm.max():.4f}')
    say('   all 20 run files passed.')
    say()

    K, EL, LLk = {}, {}, {}
    for arm in ARMS:
        for n in STARTS:
            t = tag_of(arm, n)
            for pop in POPS:
                el = np.flatnonzero(eligible(FL[t]['FF'], FL[t]['cls'], FL[t]['LL'], pop)).astype(np.int64)
                EL[(t, pop)] = el
                for order, seq in (('forward', el), ('reverse', el[::-1])):
                    k = thin(X[t], seq)
                    need(set(k.tolist()) <= set(el.tolist()), f'{t} {pop} {order}: a kept row is not eligible')
                    if len(k) > 1:
                        D = chord(X[t][k] @ X[t][k].T)
                        np.fill_diagonal(D, np.inf)
                        need(float(D.min()) > DELTA - 1e-12, f'{t} {pop} {order}: two kept rows within delta')
                    K[(t, pop, order)] = k
                    LLk[(t, pop, order)] = int(FL[t]['LL'][k].sum())

    def per(pop, arm, what):
        if what == 'eligible':
            return [len(EL[(tag_of(arm, n), pop)]) for n in STARTS]
        if what.startswith('LL '):
            return [LLk[(tag_of(arm, n), pop, what[3:])] for n in STARTS]
        return [len(K[(tag_of(arm, n), pop, what)]) for n in STARTS]

    say('COUNTS per start (s6..s10) and summed: raw eligible beside well-separated (forward = generation order, reverse = reverse generation order); '
        'fraction separated / eligible for information')
    for pop in POPS:
        say(f' {POP_LAB[pop]}')
        for arm in ARMS:
            e, fw, rv = per(pop, arm, 'eligible'), per(pop, arm, 'forward'), per(pop, arm, 'reverse')
            say(f'   {arm:2s} ({ARM_NAME[arm]}): eligible {e} = {sum(e)} | separated forward {fw} = {sum(fw)} | reverse {rv} = {sum(rv)}')
            say(f'      fraction separated / eligible: forward {[frac(x, y) for x, y in zip(fw, e)]} summed {frac(sum(fw), sum(e))} | '
                f'reverse {[frac(x, y) for x, y in zip(rv, e)]} summed {frac(sum(rv), sum(e))}')
            if pop == 'OFF':
                lf, lr = per(pop, arm, 'LL forward'), per(pop, arm, 'LL reverse')
                say(f'      kept representatives with LL (beside, not imposed): forward {lf} = {sum(lf)} | reverse {lr} = {sum(lr)}')
    say()

    say('RULE WORDS (tr_readout_v2.rule): "ahead" = more at all five starts and 4 x sum_first >= 5 x sum_second (and sum_first >= 10 when sum_second is 0); '
        '"behind" the same the other way round; otherwise "cannot tell". On the well-separated counts. V against W is the primary word; each other word separately.')
    W = {}
    for pop in POPS:
        for order in ORDERS:
            for x, y in PAIRS:
                vx, vy = per(pop, x, order), per(pop, y, order)
                w = tr.rule(vx, vy)
                W[(pop, order, x, y)] = w
                lab = 'PRIMARY' if (x, y) == ('V', 'W') else ('beside' if x == 'W' else 'separately')
                say(f'   {pop} {order}: {x} against {y} ({lab}): {x} {vx} = {sum(vx)} | {y} {vy} = {sum(vy)} -> {x} is "{w.upper()}"')
    for pop in POPS:
        wf, wr = W[(pop, 'forward', 'V', 'W')], W[(pop, 'reverse', 'V', 'W')]
        say(f'   {pop}: order-sensitive: {"yes" if wf != wr else "no"} (V against W: forward "{wf}", reverse "{wr}")')
    say()

    sv = {'delta': np.float64(DELTA)}
    for arm in ARMS:
        for n in STARTS:
            t = tag_of(arm, n)
            runrow = np.arange(NROWS, dtype=np.int64)[1:]   # candidate index i is run-file row i + 1 (start row first, asserted)
            for pop in POPS:
                sv[f'{t}__{pop}__eligible_idx'] = EL[(t, pop)]
                for order in ORDERS:
                    k = K[(t, pop, order)]
                    sv[f'{t}__{pop}__{order}__idx'] = k
                    sv[f'{t}__{pop}__{order}__runrow'] = runrow[k]
                    sv[f'{t}__{pop}__{order}__paths'] = FL[t]['paths'][k]
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'wrote {prefix}.txt and {prefix}.npz (per cell, population LEAD / OFF and order forward / reverse: __idx = index among the 1,920 candidates in run-file order, '
        f'__runrow = run-file row, __paths; __eligible_idx per cell and population)')
    FH[0].close()
    FH[0] = None


if __name__ == '__main__':
    main()
