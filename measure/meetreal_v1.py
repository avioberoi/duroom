"""Can the search's own meeting rule find real art meeting where earlier art was not? Places among the far works of the originals dated 1970 on (index: originals before 1970), 20 artist-disjoint groupings, two radii, with accident check.
Usage (CPU): python -B meetreal_v1.py OUTPREFIX -> OUTPREFIX.txt and OUTPREFIX.npz, each refused if it exists."""
import importlib.util
import os
import sys
import time
import types

import numpy as np

B = os.environ['DUROOM_OUT']
T3_FILE = os.path.dirname(os.path.abspath(__file__)) + '/t3_measure_v5.py'
SCALES = f'{B}/tests12/t3/out/scales_v5.npz'
L, M, NGROUP, NDRAWS, SEED, NCAL, CUT = 99, 3, 5, 20, 20261119, 10000, 1970
FH, OUT = [None], []
FH, OUT, CHK = [None], [], []


def say(s=''):
    OUT.append(str(s))
    print(s, flush=True)
    if FH[0] is not None:
        FH[0].write(str(s) + '\n')
        FH[0].flush()


def load_t3(path=T3_FILE):
    spec = importlib.util.spec_from_file_location('t3_measure_v5', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def artist_groups(artists, seed, n=NGROUP):
    """the distinct artists, shuffled (seeded), dealt into n groups of sizes differing by at most 1: {artist: group}"""
    u = np.unique(np.asarray(artists).astype(str))
    perm = np.random.default_rng(seed).permutation(len(u))
    return {u[p]: i % n for i, p in enumerate(perm)}


def real_places(t3, X, group, artist, row, r, q, min_rounds, m):
    """v5's place rule on real works: units = groups, rounds = artists (compact ids). X: {obs: eligible vectors}."""
    rnd = np.unique(np.asarray(artist).astype(str), return_inverse=True)[1]
    return t3.greedy(t3.together(X, X, r), np.asarray(group, int), rnd, np.asarray(row, int), q, min_rounds, m)


def accident(t3, Xc, accE, r):
    """True when the place centre (one row per observer) is within r of an ACC picture in both observers"""
    return bool(t3.together({n: Xc[n][None] for n in t3.OBS}, accE, r).sum() > 0)


def main(prefix):
    t0 = time.time()
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    t3 = load_t3()
    say(f'meetreal_v1.py; numpy {np.__version__}')
    a = types.SimpleNamespace(corpus='real', root=f'{B}/sphere', acc_prior=f'{B}/tests12/t2/renders_v1/prior256_v1',
                              acc_noise=f'{B}/tests12/t2/renders_v1/noise160_v1', acc_aim_prior=None, acc_aim_noise=None)
    C, S, idx = t3.load_common(a)
    acc = t3.load_acc(a, need_aim=False)
    sc = np.load(SCALES, allow_pickle=True)
    r_rec = np.asarray(sc['r_15'], np.float64)
    OBS = t3.OBS

    T = S['T']
    pre = np.flatnonzero(C['dated'] & (C['year'] < CUT))
    test = np.flatnonzero(C['dated'] & (C['year'] >= CUT))
    rng = np.random.default_rng(SEED)
    cal_rows = np.sort(rng.choice(pre, NCAL, replace=False))
    ctx = t3.Ctx(C, idx, t3.dup_rows(C, np.concatenate([cal_rows, test])), cols=C['dated'] & (C['year'] < CUT))
    orig = lambda rows: dict(E={n: C['E'][n][rows] for n in OBS}, src=rows)  # noqa: E731
    rc = t3.readings(orig(cal_rows), ctx)
    cal = t3.calibrate(rc['e'], np.zeros(len(cal_rows)), L)
    rt = t3.readings(orig(test), ctx)
    Ecal = np.all(rc['e'] > cal['theta'] * cal['med'], axis=1)
    Et = np.all(rt['e'] > cal['theta'] * cal['med'], axis=1)
    el = test[Et]
    say(f'originals: index {len(pre)} dated before {CUT}; calibration {NCAL} of them (rng {SEED}): med {np.round(cal["med"], 6)}, theta {cal["theta"]:.6f} '
        f'(L {L}); E in the calibration set {int(Ecal.sum())} ({100 * Ecal.mean():.2f}%)')
    say(f'test: {len(test)} originals dated {CUT} on, {len(np.unique(C["artist"][test]))} artists; E (far in both observers) {len(el)} '
        f'({100 * len(el) / len(test):.2f}%), {len(np.unique(C["artist"][el]))} artists')
    r_orig = t3.radius({n: C['E'][n][T['src']] for n in OBS}, C['artist'][T['src']])['r_15']
    say(f'radii r_15: on record (reference repaints) {r_rec[0]:.6f} / {r_rec[1]:.6f}; on the originals of the reference\'s {len(T["src"])} works '
        f'{r_orig[0]:.6f} / {r_orig[1]:.6f}')

    X = {n: C['E'][n][el] for n in OBS}
    sv = dict(test_rows=test, el_rows=el, cal_rows=cal_rows, cal_e=rc['e'], test_e=rt['e'], theta=np.array(cal['theta']), med=cal['med'],
              r_rec=r_rec, r_orig=np.asarray(r_orig))
    for rname, r in (('record', r_rec), ('originals', np.asarray(r_orig))):
        say()
        say(f'RADIUS {rname} ({r[0]:.4f} / {r[1]:.4f}): {NDRAWS} groupings; a place = >= {M} of {NGROUP} groups, each >= {t3.Q} eligible works from '
            f'>= {t3.MIN_ROUNDS} artists in the ball')
        rows = []
        for d in range(NDRAWS):
            g = artist_groups(C['artist'][test], SEED + d)
            grp = np.array([g[a_] for a_ in C['artist'][el]], int)
            pl = real_places(t3, X, grp, C['artist'][el], el, r, t3.Q, t3.MIN_ROUNDS, M)
            pl = [p for p in pl if p['support'] >= M]
            acc_ = [accident(t3, {n: X[n][p['c']] for n in OBS}, acc['E'], r) for p in pl]
            nonacc = sum(1 for x in acc_ if not x)
            rows.append((len(pl), nonacc, [int((grp == k).sum()) for k in range(NGROUP)]))
            sv[f'{rname}__d{d}__centres'] = np.array([el[p['c']] for p in pl], int)
            sv[f'{rname}__d{d}__support'] = np.array([p['support'] for p in pl], int)
            sv[f'{rname}__d{d}__accident'] = np.array(acc_, bool)
            say(f'   grouping {d:2d}: eligible per group {rows[-1][2]}; places {len(pl)}, non-accident {nonacc}: {"PASS" if nonacc >= 1 else "FAIL"}')
        npl, nna = np.array([x[0] for x in rows]), np.array([x[1] for x in rows])
        say(f'   SUMMARY {rname}: places median {np.median(npl):.0f} (range {npl.min()}-{npl.max()}); non-accident median {np.median(nna):.0f} '
            f'(range {nna.min()}-{nna.max()}); PASS in {int((nna >= 1).sum())} of {NDRAWS} groupings')
        sv[f'{rname}__places'], sv[f'{rname}__nonacc'] = npl, nna
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'saved {prefix}.npz; done {time.time() - t0:.0f} s')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
