"""Step angle per round and start: median angle between each round's 32 requests and the round's centre, from the run files qgc_ra_s6..10.
Usage: python walk_theta_v1.py OUT   (loop env, pycma 4.4.4; writes OUT.npz and OUT.txt, refuses existing files)"""
import ast
import importlib.util
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
O = f'{B}/sphere'
QV3 = os.path.join(os.environ['DUROOM_REPO'], 'measure', 'arms_barfree_qgt_v3.py')
STARTS = (6, 7, 8, 9, 10)


def unit64(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ang(a, b):
    a, b = unit64(a), unit64(b)
    return np.degrees(2 * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1)))


def main(out):
    for ext in ('.npz', '.txt'):
        assert not os.path.exists(out + ext), out + ext
    fh = open(out + '.txt', 'x')

    def say(s=''):
        print(s, flush=True)
        fh.write(s + '\n')
        fh.flush()

    spec = importlib.util.spec_from_file_location('abq3', QV3)
    abq3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abq3)
    import cma
    import yaml
    assert cma.__version__ == '4.4.4', cma.__version__
    say(f'pycma {cma.__version__}')
    say('theta_r = median angle (deg) between the round\'s 32 requests and the round\'s centre (round 0: start vector; later: unit mean of render_clip of the '
        'previous round\'s best <= 4 feasible by novelty).')
    res = {k: np.zeros((5, 60)) for k in ('theta_deg', 'theta_min', 'theta_q25', 'theta_q75', 'theta_max', 'sigma_asked', 'sigma_angle_deg',
                                          'centre_vs_replay_deg', 'n_feas_prev')}
    res['centre_from_replay'] = np.zeros((5, 60), bool)
    row0 = {}
    files = []
    for j, n in enumerate(STARTS):
        tag = f'qgc_ra_s{n}'
        f = f'{O}/runs/{tag}_committee_euclidean_min_seedrow/all_candidates.npz'
        files.append(f)
        z = np.load(f, allow_pickle=True)
        fl = ast.literal_eval(str(z['flags']))
        want = dict(tag=tag, seed=n, maxiter=60, popsize=32, reanchor=4, random_fitness=False, mode='cma', space='sphere', fidelity_max=0.7788, sigma_path=None)
        for k, v in want.items():
            assert fl.get(k) == v, (tag, k, fl.get(k), v)
        it = z['iteration'].astype(int)
        assert len(it) == 1921 and it[0] < 0 and (np.bincount(it[1:], minlength=60) == 32).all()
        X, RC = np.asarray(z['style_embs'], np.float64), np.asarray(z['render_clip'], np.float64)
        feas, nov = z['feasible'].astype(bool), np.asarray(z['novelties'], np.float64)
        R = abq3.replay_cell(tag, cma, yaml)
        assert R['used_replay'].all() and float(R['askdiff'].max()) < 1e-6, (tag, float(R['askdiff'].max()))
        x0 = unit64(X[0])
        row0[n] = (float(z['fidelity_clip'][0]), float(z['coherence'][0]), bool(z['feasible'][0]))
        say()
        say(f'{tag}: replay asks equal in {int(R["used_replay"].sum())}/60 rounds, max diff {float(R["askdiff"].max()):.1e}; row 0 (start repaint) fidelity_clip '
            f'{row0[n][0]:.4f}, coherence {row0[n][1]:.4f}, stored feasible {row0[n][2]} (the stored start flag has no gap test)')
        say('   round | theta median | min | q25 | q75 | max | centre: feasible pictures of previous round (<=4 used), centre vs replay (deg) | logged sigma_asked, '
            'atan(sigma sqrt 768) deg')
        for r in range(60):
            rows = np.flatnonzero(it == r)
            # centre as the loop defines it: round 0 = start vector; later = unit mean of render_clip of the previous round's best four feasible
            # (by -novelty, stable); with none feasible, the replay's centre (pycma's, after tell)
            if r == 0:
                c, nf, rep = x0, 32, False
            else:
                pr = np.flatnonzero(it == r - 1)
                fe = [i for i in pr if feas[i]]
                nf = len(fe)
                best = sorted(fe, key=lambda i: -nov[i])[:4]   # stable: slot order on ties, as sorted() over the told fitnesses
                if best:
                    c, rep = unit64(np.mean(RC[best], axis=0)), False
                else:
                    c, rep = unit64(R['prev'][r]), True
            a = ang(X[rows], np.repeat(c[None], len(rows), 0))
            cv = float(ang(c[None], unit64(R['prev'][r])[None])[0])
            sg = float(z['log_sigma_asked'][r])
            vals = dict(theta_deg=np.median(a), theta_min=a.min(), theta_q25=np.percentile(a, 25), theta_q75=np.percentile(a, 75), theta_max=a.max(),
                        sigma_asked=sg, sigma_angle_deg=np.degrees(np.arctan(sg * np.sqrt(768))), centre_vs_replay_deg=cv, n_feas_prev=nf)
            for k, v in vals.items():
                res[k][j, r] = v
            res['centre_from_replay'][j, r] = rep
            say(f'   {r:3d} | {vals["theta_deg"]:8.4f} | {a.min():8.4f} | {vals["theta_q25"]:8.4f} | {vals["theta_q75"]:8.4f} | {a.max():8.4f} | {nf:2d}'
                f'{" REPLAY CENTRE" if rep else ""}, {cv:.2e} | {sg:.6g}, {vals["sigma_angle_deg"]:.4f}')
        t = res['theta_deg'][j]
        say(f'   {tag} summary: theta median over rounds {np.median(t):.4f} deg [min {t.min():.4f}, max {t.max():.4f}]; implied-by-sigma median '
            f'{np.median(res["sigma_angle_deg"][j]):.4f} deg; median (theta - implied) {np.median(t - res["sigma_angle_deg"][j]):+.4f} deg; largest centre vs '
            f'replay {res["centre_vs_replay_deg"][j].max():.2e} deg; rounds with a replay centre: {np.flatnonzero(res["centre_from_replay"][j]).tolist()}')
    assert np.isfinite(res['theta_deg']).all() and (res['theta_deg'] > 0).all()
    with open(out + '.npz', 'xb') as g:
        np.savez(g, seeds=np.array(STARTS), source_files=np.array(files), **res)
    say()
    say('THETA TABLE (deg; rows = starts 6..10, the 60 rounds in order)')
    for j, n in enumerate(STARTS):
        say(f'   s{n}: ' + ' '.join(f'{v:.3f}' for v in res['theta_deg'][j]))
    say(f'wrote {out}.npz')
    fh.close()


if __name__ == '__main__':
    main(sys.argv[1])
