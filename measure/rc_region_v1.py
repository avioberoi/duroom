"""A far bar per region of P for the 15 main runs: far and faithful when a picture is over its region's 99th-percentile bar in both observers (k-means regions of P, K 4 and 8, seeds 0 to 4), beside the one-bar count.
Usage (CPU, cwd = the search folder): python -B rc_region_v1.py OUTPREFIX -> OUTPREFIX.txt and OUTPREFIX.npz, each refused if it exists."""
import importlib.util
import os
import sys

import numpy as np
import sklearn
from sklearn.cluster import KMeans

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
R, L, E = f'{O}/runs', f'{B}/logs', f'{O}/eval'
EV_FILE = f'{C_}/controls_evaluate.py'
OBS = [("DINOv3", "emb_dinov3-l"), ("SigLIP", "emb_siglip-so400m")]
K_NEIGH = 15
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
STARTS = (6, 7, 8, 9, 10)
c_ = lambda x: int(np.sum(x))  # noqa: E731
FH, OUT = [None], []


def say(s=''):
    OUT.append(s)
    print(s, flush=True)
    FH[0].write(s + '\n')
    FH[0].flush()


# as in perregion_v2.py
def unit(a):
    a = a.astype(np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def chords(a, b):
    return np.sqrt(np.clip(2.0 - 2.0 * a @ b.T, 0.0, None))


def score_ref(e, artist):
    d = chords(e, e)
    same = artist[:, None] == artist[None, :]
    d[same] = np.inf  # drops the row itself and its artist's rows
    return np.sort(d, axis=1)[:, :K_NEIGH].mean(axis=1)


def score_other(x, e):
    return np.sort(chords(x, e), axis=1)[:, :K_NEIGH].mean(axis=1)


def main(prefix):
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x')
    spec = importlib.util.spec_from_file_location('controls_evaluate', EV_FILE)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    say(f'rc_region_v1.py; sklearn {sklearn.__version__}; numpy {np.__version__}')
    Pf = np.load(f'{O}/push_emb.npz', allow_pickle=True)
    P2f, POSf = np.load(f'{O}/push2_emb.npz', allow_pickle=True), np.load(f'{O}/pos_emb.npz', allow_pickle=True)
    S = {}
    for name, key in OBS:
        e = unit(Pf[key])
        S[name] = dict(e=e, sP=score_ref(e, Pf['artist']), x2=unit(P2f[key]), xp=unit(POSf[key]))
        S[name]['s2'], S[name]['sp'] = score_other(S[name]['x2'], e), score_other(S[name]['xp'], e)
        S[name]['bar'] = np.percentile(S[name]['sP'], 99)
        say(f'{name}: global bar {S[name]["bar"]:.6f}; P over it {c_(S[name]["sP"] > S[name]["bar"])} of 2048; P2 {c_(S[name]["s2"] > S[name]["bar"])} of 1000; POS {c_(S[name]["sp"] > S[name]["bar"])} of 512')
    Pe, Te, Pz = ev.references(O)
    ids, q = ev.startids(O), ev.qfun(O)
    cells = {}
    ok_guard = True
    for arm, tg, _ in ARMS:
        for n in STARTS:
            tag = f'qgc_{tg}_s{n}'
            d = f'{R}/{tag}_committee_euclidean_min_seedrow'
            r = ev.load(f'{tag}={d},{E}/{tag}_emb.npz,{L}/{tag}.out', Pe, Te, ids, q, conf=True)
            m = r['z']['iteration'] >= 0
            X = {name: unit(np.asarray(r['z'][key])[m]) for name, key in OBS}
            sc = {name: score_other(X[name], S[name]['e']) for name, _ in OBS}
            far_g = np.all([sc[name] > S[name]['bar'] for name, _ in OBS], axis=0)
            same = bool(np.array_equal(far_g, np.asarray(r['F'], bool)))
            ok_guard &= same
            cells[tag] = dict(arm=arm, X=X, sc=sc, ok=np.asarray(r['ok_exact'], bool), F=np.asarray(r['F'], bool), far_g=far_g)
            say(f'   {tag}: global-bar far here = evaluator F picture by picture: {same}; far {c_(far_g)}, far and faithful {c_(far_g & cells[tag]["ok"])}')
    if not ok_guard:
        say('STOP: the guard did not reproduce; no region number follows.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say('GUARD PASSED.')
    say()
    say('Far and faithful under REGION bars (over the region bar in both observers), per loop: per start (6..10) = sum; one-bar count beside.')
    sv = {}
    for K in (4, 8):
        for seed in range(5):
            fits = {}
            for name, _ in OBS:
                km = KMeans(n_clusters=K, n_init=3, random_state=seed).fit(S[name]['e'])
                rP = km.labels_
                rbar = np.array([np.percentile(S[name]['sP'][rP == k], 99) for k in range(K)])
                fits[name] = (km, rP, rbar)
            f2 = np.all([S[n]['s2'] > fits[n][2][fits[n][0].predict(S[n]['x2'])] for n, _ in OBS], axis=0)
            fp = np.all([S[n]['sp'] > fits[n][2][fits[n][0].predict(S[n]['xp'])] for n, _ in OBS], axis=0)
            line = []
            for arm, tg, w in ARMS:
                v, g = [], []
                for n in STARTS:
                    c = cells[f'qgc_{tg}_s{n}']
                    reg = {name: fits[name][0].predict(c['X'][name]) for name, _ in OBS}
                    far_r = np.all([c['sc'][name] > fits[name][2][reg[name]] for name, _ in OBS], axis=0)
                    c[f'reg_{K}_{seed}'], c[f'farr_{K}_{seed}'] = reg, far_r
                    v.append(c_(far_r & c['ok']))
                    g.append(c_(c['far_g'] & c['ok']))
                    sv[f'qgc_{tg}_s{n}__farr_K{K}_seed{seed}'] = far_r
                    for name, _ in OBS:
                        sv[f'qgc_{tg}_s{n}__region_{name}_K{K}_seed{seed}'] = reg[name]
                line.append(f'{arm} {v} = {sum(v)} (one bar {sum(g)})')
            say(f' K={K} seed {seed}: ' + ' | '.join(line) + f' || P2 far {c_(f2)} of 1000, POS far {c_(fp)} of 512')
            if seed == 0:
                fits0 = fits
        say(f' K={K} seed 0, by region (regions ordered by region bar), per observer: P pictures, region bar; far-and-faithful under the ONE bar in each loop '
            f'(V / C1 / C2) whose reading of that observer falls in the region; same under REGION bars')
        for name, _ in OBS:
            km, rP, rbar = fits0[name]
            for k in np.argsort(rbar):
                one = [sum(c_((cells[f'qgc_{tg}_s{n}']['reg_{0}_0'.format(K)][name] == k) & cells[f'qgc_{tg}_s{n}']['far_g'] & cells[f'qgc_{tg}_s{n}']['ok'])
                           for n in STARTS) for _, tg, _ in ARMS]
                reg = [sum(c_((cells[f'qgc_{tg}_s{n}']['reg_{0}_0'.format(K)][name] == k) & cells[f'qgc_{tg}_s{n}'][f'farr_{K}_0'] & cells[f'qgc_{tg}_s{n}']['ok'])
                           for n in STARTS) for _, tg, _ in ARMS]
                allp = [sum(c_(cells[f'qgc_{tg}_s{n}']['reg_{0}_0'.format(K)][name] == k) for n in STARTS) for _, tg, _ in ARMS]
                say(f'    {name} region: {c_(rP == k):4d} P pictures, region bar {rbar[k]:.3f} (global {S[name]["bar"]:.3f}) | loop pictures in it V/C1/C2 {allp} '
                    f'| far&faithful one bar {one} | far&faithful region bars {reg}')
        say()
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'wrote {prefix}.txt and {prefix}.npz')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
