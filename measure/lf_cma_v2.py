"""pycma replay of the loop on the 15 main runs: per round the sampling centre, pycma's centre after tell, the re-anchored centre, the step size asked and after tell, and the replayed asks against the saved requests.
Usage (search environment, pycma): python -B lf_cma_v2.py OUT.npz"""
import ast
import os
import re
import sys

import cma
import numpy as np
import yaml

sys.dont_write_bytecode = True
O = os.environ['DUROOM_OUT'] + '/sphere'
CELLS = [f'qgc_{a}_s{n}' for a in ('ra', 'pin', 'c2') for n in (6, 7, 8, 9, 10)]


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main(out):
    assert not os.path.exists(out), out
    res = {'cma_version': np.array(cma.__version__), 'numpy_version': np.array(np.__version__)}
    for cell in CELLS:
        z = np.load(f'{O}/runs/{cell}_committee_euclidean_min_seedrow/all_candidates.npz', allow_pickle=True)
        fl = ast.literal_eval(str(z['flags']))
        cfg = yaml.safe_load(open(fl['config']))['cma']
        sigma0, it_all = float(cfg['sigma0']), z['iteration'].astype(int)
        sig_path = np.asarray(np.load(fl['sigma_path'], allow_pickle=True)['log_sigma'], np.float64) if fl.get('sigma_path') else None
        x0 = np.asarray(z['style_embs'][0], np.float32)
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {'popsize': fl['popsize'], 'CMA_diagonal': cfg['diagonal'], 'maxiter': fl['maxiter'], 'seed': fl['seed'],
                                                          'tolfun': cfg['tolfun'], 'tolx': cfg['tolx'], 'verbose': -9})
        rng_fit = np.random.default_rng(fl['seed'] + 7919)
        n = fl['maxiter']
        prev, cmac, rea = np.zeros((n, 768)), np.zeros((n, 768)), np.zeros((n, 768))
        s_asked, s_after, askdiff, fits_all, used_replay = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, 32)), np.zeros(n, bool)
        stops = []
        odd = {'raw': np.zeros((n, 3)), 'told': np.zeros((n, 3))}   # per round: median over the 32 of |v - centre| / sigma_asked in dim 644, dim 413, the other 766 pooled
        for it in range(n):
            st = es.stop()
            if st:
                stops.append(f'{it}:{st}')
            if sig_path is not None and it > 0:
                es.sigma = float(sig_path[it - 1])
            if fl['sigma_cap'] is not None and es.sigma > fl['sigma_cap'] * sigma0:
                es.sigma = fl['sigma_cap'] * sigma0
            s_asked[it] = float(es.sigma)
            prev[it] = np.asarray(es.mean, np.float64)
            X = es.ask()
            rows = np.flatnonzero(it_all == it)
            assert len(rows) == 32 and (z['index'][rows] == np.arange(32)).all()
            Xu = unit(np.array(X))
            saved = np.asarray(z['style_embs'][rows], np.float64)
            askdiff[it] = float(np.abs(Xu - saved).max())
            used_replay[it] = askdiff[it] < 1e-6
            told = [Xu[i] for i in range(32)] if used_replay[it] else [saved[i] for i in range(32)]
            feas = z['feasible'][rows].astype(bool)
            assert (np.asarray(z['coherence'][rows], np.float64) >= 0).all(), (cell, it)   # the loop adds max(0 - coherence, 0)^2 to a told score: zero only when every coherence is >= 0
            Xa = np.array(X, np.float64)
            for key, V_ in (('raw', Xa), ('told', Xu)):
                dv = np.abs(V_ - prev[it]) / s_asked[it]
                odd[key][it] = [np.median(dv[:, 644]), np.median(dv[:, 413]), np.median(np.delete(dv, [644, 413], axis=1))]
            if fl['random_fitness']:
                fits = [float(v) for v in rng_fit.standard_normal(32)]
            else:
                nov = np.asarray(z['novelties'][rows], np.float64)
                fid = np.asarray(z['fidelity_clip'][rows], np.float64)
                ff = [-nov[i] for i in range(32) if feas[i]]
                off = np.median(ff) if ff else 0.0
                fits = [(-nov[i]) if feas[i] else off + max(fid[i] - fl['fidelity_max'], 0.0) ** 2 for i in range(32)]
            fits_all[it] = fits
            es.tell(told, fits)
            s_after[it] = float(es.sigma)
            es.mean = es.mean / np.linalg.norm(es.mean)
            cmac[it] = np.asarray(es.mean, np.float64)
            order = sorted(range(32), key=lambda i: fits[i])
            best = [np.asarray(z['render_clip'][rows[i]], np.float64) for i in order if feas[i]][:fl['reanchor']]
            if best:
                m = np.mean(np.asarray(best, np.float64), axis=0)
                es.mean = m / np.linalg.norm(m)
            rea[it] = np.asarray(es.mean, np.float64)
        la, ls = np.asarray(z['log_sigma_asked'], np.float64), np.asarray(z['log_sigma'], np.float64)
        print(f'{cell}: sigma0 {sigma0}, sigma path {fl.get("sigma_path")}; replayed asks equal the saved requests (1e-6) in {int(used_replay.sum())} of {n} rounds '
              f'(max abs diff median {np.median(askdiff):.2e}, max {askdiff.max():.2e}); sigma asked vs log_sigma_asked max |diff| {np.abs(s_asked - la).max():.2e}; '
              f'sigma after tell vs log_sigma max |diff| {np.abs(s_after - ls).max():.2e}; es.stop() lines {stops}', flush=True)
        for k, v in dict(prev=prev, cma=cmac, rea=rea, s_asked=s_asked, s_after=s_after, askdiff=askdiff, fits=fits_all, used_replay=used_replay,
                         odd_raw=odd['raw'], odd_told=odd['told']).items():
            res[f'{cell}__{k}'] = v
    with open(out, 'xb') as fh:
        np.savez_compressed(fh, **res)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1])
