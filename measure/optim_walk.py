"""Optimizers on the novelty score in embedding space, no decoder: 200 starts from a reference of works before a cutoff; SGD, momentum, Adam, RMSprop, Adagrad, CMA-ES and three controls; far rate, angle, nearest real work, later share, and displacement in the start's local directions.
Usage: python optim_walk.py EDA_DIR (EDA_DIR holds data.py; CPU; torch, numpy, pycma)"""
import sys

import cma
import numpy as np
import torch

from band import one_per_artist

K, N_START, MAX_DEG, STEP_DEG, MAX_STEPS, M_LOCAL, K_LOCAL, W = 15, 200, 60.0, 1.0, 400, 39, 100, 40
torch.set_default_dtype(torch.float64)


def chord(A, B):
    return torch.sqrt(torch.clamp(2 - 2 * A @ B.T, min=1e-12))


def novelty(X, R, mask):
    """mean chord from each row of X (unit) to its K nearest rows of R, the rows flagged in mask left out."""
    return chord(X, R).masked_fill(mask, float('inf')).topk(K, dim=1, largest=False).values.mean(1)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def angle(X, S):
    return torch.rad2deg(torch.acos(torch.clamp((X * S).sum(1), -1, 1)))


def walk_torch(make_opt, S, R, mask, thr, lr):
    """one optimizer on -N from the starts S; returns stopping points, whether each stopped FAR, and the median first-step angle."""
    x = S.clone().requires_grad_(True)
    opt = make_opt([x], lr)
    stop, far = S.clone(), torch.zeros(len(S), dtype=torch.bool)
    done = torch.zeros(len(S), dtype=torch.bool)
    first = None
    for _ in range(MAX_STEPS):
        opt.zero_grad()
        (-novelty(unit(x), R, mask).sum()).backward()
        opt.step()
        with torch.no_grad():
            x /= x.norm(dim=1, keepdim=True)
            if first is None:
                first = float(angle(x, S).median())
            n, a = novelty(x, R, mask), angle(x, S)
            hit = ~done & ((n >= thr) | (a >= MAX_DEG))
            stop[hit], far[hit] = x[hit], (n >= thr)[hit]
            done |= hit
            if done.all():
                break
    with torch.no_grad():
        stop[~done] = x[~done]                               # steps ran out: the last point, not FAR
    return stop.detach(), far, first


def walk_line(U, S, R, mask, thr):
    """controls: x(theta) = cos(theta) s + sin(theta) u, theta in steps of STEP_DEG; stop at the first FAR point or at MAX_DEG."""
    stop, far, done = S.clone(), torch.zeros(len(S), dtype=torch.bool), torch.zeros(len(S), dtype=torch.bool)
    for k in range(1, int(MAX_DEG / STEP_DEG) + 1):
        t = np.radians(k * STEP_DEG)
        x = np.cos(t) * S + np.sin(t) * U
        n = novelty(x, R, mask)
        hit = ~done & ((n >= thr) | (k * STEP_DEG >= MAX_DEG))
        stop[hit], far[hit] = x[hit], (n >= thr)[hit]
        done |= hit
    return stop, far


def walk_cma(S, R, mask, thr, seed):
    """CMA-ES as in the search (diagonal, 32 samples, sigma 0.0208, centre put back on the sphere after each tell); the centre is read."""
    stop, far = S.clone(), torch.zeros(len(S), dtype=torch.bool)
    for i in range(len(S)):
        es = cma.CMAEvolutionStrategy(S[i].numpy(), 0.0208, {'popsize': 32, 'CMA_diagonal': True, 'maxiter': 60, 'verbose': -9, 'seed': seed * 1000 + i + 1})
        m = S[i:i + 1]
        for _ in range(60):
            Z = es.ask()
            f = -novelty(unit(torch.tensor(np.asarray(Z))), R, mask[i:i + 1].expand(len(Z), -1))
            es.tell(Z, f.tolist())
            es.mean = es.mean / np.linalg.norm(es.mean)
            m = torch.tensor(es.mean)[None]
            n = novelty(m, R, mask[i:i + 1])
            if n >= thr or angle(m, S[i:i + 1]) >= MAX_DEG:
                far[i] = bool(n >= thr)
                break
        stop[i] = m[0]
    return stop, far


def run(X, year, artist, c, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dated = ~np.isnan(year) & (year >= 1000) & (year <= 2026) & (artist != '')
    arts = rng.permutation(np.unique(artist[dated]))
    inA = np.isin(artist, arts[:len(arts) // 2])
    ref = one_per_artist(np.flatnonzero(dated & (year < c) & inA), artist, rng, 20)
    ei = rng.permutation(one_per_artist(np.flatnonzero(dated & (year < c) & ~inA), artist, rng, 3))        # shuffled again: a cut after the per-artist pass would favour prolific artists
    li = rng.permutation(one_per_artist(np.flatnonzero(dated & (year >= c) & (year < c + W) & ~inA), artist, rng, 3))
    n_pool = min(len(ei), len(li))
    ei, li = ei[:n_pool], li[:n_pool]
    R = torch.tensor(X[ref].astype(np.float64))
    ra = artist[ref]
    own = torch.cat([novelty(R[s:s + 1000], R, torch.tensor(ra[s:s + 1000, None] == ra[None, :])) for s in range(0, len(R), 1000)])
    thr = float(torch.quantile(own, 0.99))
    Pool = torch.tensor(X[np.concatenate([ei, li])].astype(np.float64))
    is_later = torch.cat([torch.zeros(n_pool, dtype=torch.bool), torch.ones(n_pool, dtype=torch.bool)])
    pa = artist[np.concatenate([ei, li])]
    no_mask = torch.zeros(1, len(R), dtype=torch.bool)

    def read(P_, self_mask=None):
        """NEAR and LATER SHARE of the rows of P_ against the pooled held-out real works (rows flagged in self_mask left out)."""
        D = chord(P_, Pool)
        if self_mask is not None:
            D = D.masked_fill(self_mask, float('inf'))
        d, idx = D.topk(10, dim=1, largest=False)
        return d[:, 0], is_later[idx].double().mean(1)

    pn = torch.cat([novelty(Pool[s:s + 1000], R, no_mask.expand(len(Pool[s:s + 1000]), -1)) for s in range(0, len(Pool), 1000)])
    near_p, share_p = read(Pool, torch.tensor(pa[:, None] == pa[None, :]))
    pf = pn >= thr
    print(f'\ncutoff {c} seed {seed}: REFERENCE {len(ref)} works; pools {n_pool} earlier and {n_pool} later (held-out artists, cap 3); FAR threshold {thr:.4f} (reference median N {float(own.median()):.3f})')
    print(f'   real held-out works: earlier NEAR {float(near_p[~is_later].mean()):.3f} later share {float(share_p[~is_later].mean()):.3f} FAR {int((pf & ~is_later).sum())} of {n_pool} | '
          f'later NEAR {float(near_p[is_later].mean()):.3f} later share {float(share_p[is_later].mean()):.3f} FAR {int((pf & is_later).sum())} of {n_pool} | '
          f'among the {int(pf.sum())} real works that are FAR: {int((pf & is_later).sum())} are later; their NEAR {float(near_p[pf].mean()):.3f}, later share {float(share_p[pf].mean()):.3f}')

    si = rng.choice(len(R), N_START, replace=False)
    S = R[si].clone()
    mask = torch.tensor(ra[si][:, None] == ra[None, :])
    D0 = chord(S, R).masked_fill(mask, float('inf'))
    nn = D0.topk(K_LOCAL, dim=1, largest=False).indices
    V = []                                                    # each start's 39 local directions (rows), from its 100 nearest reference works
    for i in range(N_START):
        Nb = R[nn[i]]
        V.append(torch.linalg.svd(Nb - Nb.mean(0), full_matrices=False)[2][:M_LOCAL])
    V = torch.stack(V)
    g = torch.Generator().manual_seed(seed)

    def tangent(U):
        U = U - (U * S).sum(1, keepdim=True) * S
        return unit(U)

    U_even = tangent(torch.randn(N_START, X.shape[1], generator=g))
    U_loc = tangent(torch.einsum('nk,nkd->nd', torch.randn(N_START, M_LOCAL, generator=g), V))
    opts = {'SGD': lambda p, lr: torch.optim.SGD(p, lr=lr), 'SGD momentum 0.9': lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
            'Adam': lambda p, lr: torch.optim.Adam(p, lr=lr), 'RMSprop': lambda p, lr: torch.optim.RMSprop(p, lr=lr), 'Adagrad': lambda p, lr: torch.optim.Adagrad(p, lr=lr)}
    out = {'START': (S.clone(), novelty(S, R, mask) >= thr, None)}
    out['EVEN'] = walk_line(U_even, S, R, mask, thr) + (None,)
    out['LOCAL-EVEN'] = walk_line(U_loc, S, R, mask, thr) + (None,)
    for name, mk in opts.items():
        x = S[:40].clone().requires_grad_(True)             # calibrate: one step at a trial rate, then scale so the first step turns STEP_DEG
        o = mk([x], 1e-3)
        (-novelty(unit(x), R, mask[:40]).sum()).backward()
        o.step()
        a1 = float(angle(unit(x.detach()), S[:40]).median())
        lr = 1e-3 * STEP_DEG / a1
        stop, far, first = walk_torch(mk, S, R, mask, thr, lr)
        out[name] = (stop, far, f'lr {lr:.2e}, first step {first:.2f} deg')
    nc = 80                                                   # CMA-ES on the first 80 starts only (it is slow); its rows are padded as not FAR
    cs, cf = walk_cma(S[:nc], R, mask[:nc], thr, seed)
    out['CMA-ES'] = (torch.cat([cs, S[nc:]]), torch.cat([cf, torch.zeros(N_START - nc, dtype=torch.bool)]), f'run on the first {nc} starts only')

    ev = {}
    for name, (stop, far, note) in out.items():
        near, share = read(stop)
        disp = stop - S
        inspan = (torch.einsum('nkd,nd->nk', V, disp) ** 2).sum(1) / torch.clamp((disp ** 2).sum(1), min=1e-18)
        ev[name] = dict(far=far, ang=angle(stop, S), n=novelty(stop, R, mask), near=near, share=share, inspan=inspan)
    eb = ev['EVEN']
    for name, e in ev.items():
        f = e['far']
        both = f & eb['far']
        s = (f'   {name:17s} FAR {int(f.sum()):3d} of {N_START} | at the FAR point: median angle {float(e["ang"][f].median()) if f.any() else float("nan"):5.1f} | N {float(e["n"][f].mean()) if f.any() else float("nan"):.3f} | '
             f'NEAR {float(e["near"][f].mean()) if f.any() else float("nan"):.3f} | later share {float(e["share"][f].mean()) if f.any() else float("nan"):.3f} | in the 39 local directions {float(e["inspan"][f].mean()) if f.any() else float("nan"):.3f}')
        if name not in ('EVEN', 'START') and both.sum() >= 20:
            for key in ('share', 'near'):
                d = (e[key] - eb[key])[both].numpy()
                b = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
                s += f' | {"later share" if key == "share" else "NEAR"} minus EVEN {d.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}] (n {len(d)})'
        if out[name][2]:
            s += f' | {out[name][2]}'
        print(s, flush=True)
    names = [k for k in ev if k != 'START']
    diff = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = ((ev[a]['share'] - ev[b]['share'])[ev[a]['far'] & ev[b]['far']]).numpy()
            if len(d) >= 20:
                bt = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
                lo, hi = np.percentile(bt, [2.5, 97.5])
                if abs(d.mean()) >= 0.05 and (lo > 0 or hi < 0):
                    diff.append(f'{a} minus {b} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]')
    print('   pairs that differ in later share by 0.05 or more with the interval off 0 (both FAR): ' + ('; '.join(diff) or 'none'), flush=True)

if __name__ == '__main__':
    sys.path.insert(0, sys.argv[1])
    import data
    X = data.lens(data.CLIP)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    for c, seeds in ((1900, (0, 1, 2)), (1850, (0,)), (1950, (0,))):
        for seed in seeds:
            run(X, data.year, data.labels['artist'], c, seed)
