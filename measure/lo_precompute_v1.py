"""Leftover-share precompute: per observer the basis (P's mean and K = 32 leading directions) and the artist-out cross-fit null, then leftover shares of noise renders, random-request renders, the 15 cells' candidates and later repaints. CPU.
Usage: python -B lo_precompute_v1.py OUTPREFIX   (writes OUTPREFIX_basis.npz, _scores.npz, .txt; refuses if any exists)"""
import importlib.util
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leftover_fn as lf  # noqa: E402

B = os.environ['DUROOM_OUT']
C, O = HERE, f'{B}/sphere'
R2 = f'{B}/tests12/t2/renders_v1'
EV_FILE = f'{C}/controls_evaluate.py'
DINO, SIG, CLIP = 'dinov3-l', 'siglip-so400m', 'clip-vit-l-14'
NOV_OBS, ALL_OBS = (DINO, SIG), (DINO, SIG, CLIP)
K, FOLDS, SEED = 32, 5, 0
ARMS = (('V', 'ra'), ('C1', 'pin'), ('C2', 'c2'))
KINDS = {'a': 'a uniform pixel noise', 'b': 'b gaussian pixel noise', 'c': 'c solid colour', 'd': 'd smooth low-freq', 'e': 'e 8x8 colour blocks'}
OUT = []


def say(s=''):
    OUT.append(s)
    print(s, flush=True)


def over_both(L, line):
    """L, line: {observer: shares} and {observer: line}; over in every observer of NOV_OBS (strictly above)."""
    return np.logical_and.reduce([np.asarray(L[o]) > line[o] for o in NOV_OBS])


def auc(a, b):
    """probability that a random member of a scores above a random member of b (ties count half)."""
    bs = np.sort(b)
    lt, le = np.searchsorted(bs, a, 'left'), np.searchsorted(bs, a, 'right')
    return float((lt + 0.5 * (le - lt)).sum() / (len(a) * len(b)))


def halves_line(E, artist, k):
    """Null from five held-out-half splits (artists permuted, one half fits, the other is scored; median over seeds of the 99th percentile)."""
    bars = []
    for s in range(5):
        arts = np.random.default_rng(s).permutation(np.unique(artist))
        inA = np.isin(artist, arts[:len(arts) // 2])
        L = np.empty(len(E))
        for fit, test in ((inA, ~inA), (~inA, inA)):
            L[test] = lf.share(E[test], *lf.fit(E[fit], k))
        bars.append(np.percentile(L, 99))
    return float(np.median(bars))


def main(prefix):
    for ext in ('_basis.npz', '_scores.npz', '.txt'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    spec = importlib.util.spec_from_file_location('controls_evaluate', EV_FILE)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    P, T, Pz = ev.references(O)
    say(f'K {K} directions fitted on P (2,048 repaint readings, renormalised); null = artist-out cross-fit, {FOLDS} folds, seed {SEED}; line = null 99th percentile; '
        f'"over" = strictly above the line; told observers (nov) {list(NOV_OBS)}, CLIP logged only')
    Part = np.array([str(a) for a in Pz['artist']])
    assert len(Part) == 2048
    Z2 = np.load(f'{O}/push2_emb.npz', allow_pickle=True)
    art2 = np.array([str(a) for a in Z2['artist']])
    out2 = ~np.isin(art2, Part)
    basis = dict(K=np.array(K), folds=np.array(FOLDS), seed=np.array(SEED),
                 observers=np.array(ALL_OBS), nov_observers=np.array(NOV_OBS))
    line, fitted = {}, {}
    say()
    say('BASIS AND NULL')
    for o in ALL_OBS:
        E = np.asarray(Pz[f'emb_{o}'], dtype=np.float64)
        mean, V = lf.fit(E, K)
        assert V.shape == (K, E.shape[1]) and np.allclose(V @ V.T, np.eye(K), atol=1e-10)
        Lcf, fold = lf.crossfit(E, Part, K, FOLDS, SEED)
        assert np.isfinite(Lcf).all() and set(fold.tolist()) == set(range(FOLDS))
        for f in range(FOLDS):                        # no artist is in both a fit fold and its scored fold
            assert not set(Part[fold == f]) & set(Part[fold != f]), f
        med, mad, q99 = lf.null(Lcf)
        Lin = lf.share(E, mean, V)
        L2o = lf.share(np.asarray(Z2[f'emb_{o}'])[out2], mean, V)
        mnorm = float(np.linalg.norm(mean))
        mshare = float(lf.share(mean / mnorm, mean, V)[0])
        line[o], fitted[o] = q99, (mean, V)
        basis.update({f'mean_{o}': mean, f'V_{o}': V, f'null_median_{o}': np.array(med), f'null_mad_{o}': np.array(mad), f'null_q99_{o}': np.array(q99),
                      f'insample_q99_{o}': np.array(float(np.percentile(Lin, 99))), f'crossfit_share_{o}': Lcf, f'fold_{o}': fold})
        say(f'{o}: dim {E.shape[1]}; max | |e| - 1 | of P\'s stored readings {np.abs(np.linalg.norm(E, axis=1) - 1).max():.2e}; fold sizes {np.bincount(fold).tolist()}')
        say(f'   cross-fit null (used): median {med:.6f}, MAD {mad:.6f}, 99th {q99:.6f}; P in-sample 99th {np.percentile(Lin, 99):.6f} (printed, never used); '
            f'P2 rows whose artist is not in P: {int(out2.sum())} rows, {len(np.unique(art2[out2]))} artists, their 99th {np.percentile(L2o, 99):.6f}')
        say(f'   |mean_P| {mnorm:.6f}; share of the unit mean direction itself {mshare:.6f}; P rows over the line (cross-fit) {int((Lcf > q99).sum())} of 2048')
        if o in NOV_OBS:
            h = halves_line(E, Part, K)
            say(f'   held-out halves line: {h:.6f}')
    cf_both = (basis[f'crossfit_share_{DINO}'] > line[DINO]) & (basis[f'crossfit_share_{SIG}'] > line[SIG])
    say(f'P rows over the line in both DINOv3 and SigLIP (cross-fit shares): {int(cf_both.sum())} of 2048')
    with open(prefix + '_basis.npz', 'xb') as fh:
        np.savez(fh, **basis)
    say(f'basis written: {prefix}_basis.npz')
    scores = {}

    def left(emb):
        return {o: lf.share(emb[o], *fitted[o]) for o in ALL_OBS}

    def count_line(name, L):
        b = over_both(L, line)
        c3 = b & (L[CLIP] > line[CLIP])
        return (f'{name:34s} n {len(L[DINO]):5d} | over DINOv3 {int((L[DINO] > line[DINO]).sum()):5d} | over SigLIP {int((L[SIG] > line[SIG]).sum()):5d} | '
                f'over BOTH {int(b.sum()):5d} | over CLIP (logged only) {int((L[CLIP] > line[CLIP]).sum()):5d} | all three {int(c3.sum()):5d}')

    say()
    say('NOISE REQUESTS PAINTED BY THE DECODER (renders_v1/noise160_v1)')
    em4 = np.load(f'{R2}/noise160_v1/emb.npz', allow_pickle=True)
    kat4 = [ln.strip() for ln in open(f'{R2}/noise160_v1/inputs/expected_kat.txt') if ln.strip()]
    assert [str(p) for p in em4['paths'][:len(kat4)]] == kat4
    man4 = np.load(f'{R2}/noise160_v1/noise/manifest.npz', allow_pickle=True)
    k4 = len(kat4)
    assert [str(p) for p in em4['paths'][k4:]] == [str(p) for p in man4['paths']]
    nk = np.array([str(t) for t in man4['tag']])
    LN = left({o: em4[f'emb_{o}'][k4:] for o in ALL_OBS})
    say(f'file {R2}/noise160_v1/emb.npz; {k4} known-answer rows dropped')
    for kd in 'abcde':
        s = nk == kd
        assert s.sum() == 32
        say(count_line(KINDS[kd], {o: LN[o][s] for o in ALL_OBS}))
    say(count_line('all 160', LN))
    scores.update({f'noise__{o}': LN[o] for o in ALL_OBS}, noise__kind=nk)

    say()
    say('RANDOM REQUESTS PAINTED BY THE DECODER (renders_v1/prior256_v1)')
    em = np.load(f'{R2}/prior256_v1/emb.npz', allow_pickle=True)
    kat = [ln.strip() for ln in open(f'{R2}/prior256_v1/inputs/expected_kat.txt') if ln.strip()]
    assert [str(p) for p in em['paths'][:len(kat)]] == kat
    man = np.load(f'{R2}/prior256_v1/prior/manifest.npz', allow_pickle=True)
    k = len(kat)
    assert [str(p) for p in em['paths'][k:]] == [str(p) for p in man['paths']]
    kind = np.array([str(x) for x in man['kind']])
    LR = left({o: em[f'emb_{o}'][k:] for o in ALL_OBS})
    say(f'file {R2}/prior256_v1/emb.npz; {k} known-answer rows dropped')
    for kd, n in (('cone', 192), ('sphere', 64)):
        s = kind == kd
        assert s.sum() == n
        say(count_line(f'random request, {kd}', {o: LR[o][s] for o in ALL_OBS}))
    say(count_line('all 256', LR))
    scores.update({f'prior__{o}': LR[o] for o in ALL_OBS}, prior__kind=kind)

    say()
    say('STORED CANDIDATES (15 cells, 1,920 each; loop readings = run file emb_dinov3-l, emb_siglip-so400m, render_clip; exact = eval/<cell>_emb.npz)')
    for arm, tg in ARMS:
        Ls, Lx, feas, dev = {o: [] for o in ALL_OBS}, {o: [] for o in ALL_OBS}, [], {o: 0.0 for o in ALL_OBS}
        for n in (6, 7, 8, 9, 10):
            cell = f'qgc_{tg}_s{n}'
            rf, xf = f'{O}/runs/{cell}_committee_euclidean_min_seedrow/all_candidates.npz', f'{O}/eval/{cell}_emb.npz'
            z, ex = np.load(rf, allow_pickle=True), np.load(xf, allow_pickle=True)
            m = z['iteration'] >= 0
            assert m.sum() == 1920 and all(str(a) == str(b) for a, b in zip(z['paths'], ex['paths']))
            run = {DINO: z[f'emb_{DINO}'][m], SIG: z[f'emb_{SIG}'][m], CLIP: z['render_clip'][m]}
            a, b = left(run), left({o: ex[f'emb_{o}'][m] for o in ALL_OBS})
            for o in ALL_OBS:
                Ls[o].append(a[o])
                Lx[o].append(b[o])
                dev[o] = max(dev[o], float(np.abs(np.linalg.norm(np.asarray(run[o], np.float64), axis=1) - 1).max()))
                scores[f'cell__{cell}__loop__{o}'], scores[f'cell__{cell}__exact__{o}'] = a[o], b[o]
            feas.append(np.asarray(z['feasible'][m], bool))
        Ls, Lx, feas = {o: np.concatenate(v) for o, v in Ls.items()}, {o: np.concatenate(v) for o, v in Lx.items()}, np.concatenate(feas)
        say(count_line(f'{arm}: all candidates, loop', Ls))
        say(count_line(f'{arm}: feasible (fid <= 0.7788), loop', {o: Ls[o][feas] for o in ALL_OBS}))
        say(count_line(f'{arm}: all candidates, exact', Lx))
        say(f'   {arm}: max | |e| - 1 | of the loop readings ' + ', '.join(f'{o} {dev[o]:.2e}' for o in ALL_OBS) +
            '; max |loop share - exact share| ' + ', '.join(f'{o} {np.abs(Ls[o] - Lx[o]).max():.2e}' for o in ALL_OBS) +
            '; share p50 / p99 / max (loop) ' + ', '.join(f'{o} {np.percentile(Ls[o], 50):.4f} / {np.percentile(Ls[o], 99):.4f} / {Ls[o].max():.4f}' for o in ALL_OBS))

    say()
    say('LATER REPAINTS AGAINST HELD-OUT PRE-1900 REPAINTS (P2 = push2_emb.npz, 1,000; pos = pos_emb.npz, 1900-1969, 512; today 1970 on = push_today_emb.npz rows with year >= 1970)')
    nullz = {o: (float(np.median(P[o]['own'])), float(np.median(np.abs(P[o]['own'] - np.median(P[o]['own']))))) for o in NOV_OBS}

    def combined(emb):
        L = left(emb)
        zl = np.min([lf.z(L[o], float(basis[f'null_median_{o}']), float(basis[f'null_mad_{o}'])) for o in NOV_OBS], axis=0)
        zd = []
        for o in NOV_OBS:                               # mean chord to the 15 nearest P rows, all P rows (as the loop scores a candidate)
            d = np.sort(ev.chord(ev.unit(emb[o]) @ P[o]['E'].T), axis=1)[:, :ev.K].mean(1)
            zd.append((d - nullz[o][0]) / (nullz[o][1] + 1e-8))
        return L, zl, np.min(zd, axis=0)

    Zp = np.load(f'{O}/pos_emb.npz', allow_pickle=True)
    G = {'P2 (earlier, held out)': {o: Z2[f'emb_{o}'] for o in ALL_OBS}, 'pos (1900-1969)': {o: Zp[f'emb_{o}'] for o in ALL_OBS}}
    Tz = np.load(f'{O}/push_today_emb.npz', allow_pickle=True)
    t70 = T['year'] >= 1970
    G['today, 1970 on'] = {o: np.asarray(Tz[f'emb_{o}'])[t70] for o in ALL_OBS}
    R = {}
    for name, emb in G.items():
        L, zl, zd = combined(emb)
        R[name] = (zl, zd)
        say(count_line(name, L))
        for o in ALL_OBS:
            scores[f'repaint__{name.split()[0]}__{o}'] = L[o]
    say(f'   P2 rows whose artist is not in P: {int(out2.sum())}')
    e_l, e_d = R['P2 (earlier, held out)']
    for name in ('pos (1900-1969)', 'today, 1970 on'):
        zl, zd = R[name]
        say(f'   AUC {name} against P2: leftover {auc(zl, e_l):.3f}, distance {auc(zd, e_d):.3f}; against the {int(out2.sum())} P2 rows whose artist is not in P: '
            f'leftover {auc(zl, e_l[out2]):.3f}, distance {auc(zd, e_d[out2]):.3f}')
    with open(prefix + '_scores.npz', 'xb') as fh:
        np.savez(fh, **scores)
    say()
    say(f'scores written: {prefix}_scores.npz')
    open(prefix + '.txt', 'x').write('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
