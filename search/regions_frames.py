"""Writes 868 unfiltered request frames between k-means regions (K = 12) of P's pre-1900 request vectors: BASE, PP12, PP35, RD35, SH35, GD35, CL, IW, P2RE.
Usage: python regions_frames.py PUSH_EMB PUSH2_EMB STARTS_NPZ OUT_NPZ"""
import json
import sys

import numpy as np

K, RANK, NSLOT, SEED = 12, 8, 10, 4243
# x = cos(phi) b + sin(phi) d_perp; d = b - a (PP), centre(B) - centre(A) (RD), b's offset from centre(B) in B's local directions (SH), p - q (GD)
PHI = {'BASE': 0.0, 'PP12': 12.0, 'PP35': 35.0, 'RD35': 35.0, 'SH35': 35.0, 'GD35': 35.0}
CL_T = (0.5, 1.25, 1.5)
RQ, CONE = 0.877, 0.506          # the reach gate's window (probe_frames.py), recorded only
OBS = ('dinov3-l', 'siglip-so400m')
unit = lambda v: np.asarray(v, np.float64) / np.linalg.norm(np.asarray(v, np.float64), axis=-1, keepdims=True)  # noqa: E731
chord = lambda c: np.sqrt(np.clip(2 - 2 * c, 0, None))  # noqa: E731


def skmeans(X, k, rng, iters=500):
    """Spherical k-means: centres are unit means; stops when the labels stop changing."""
    C, lab = X[rng.choice(len(X), k, replace=False)], None
    for _ in range(iters):
        new = (X @ C.T).argmax(1)
        if lab is not None and (new == lab).all():
            return lab, C
        lab = new
        assert np.bincount(lab, minlength=k).min() > 0, 'empty region'
        C = np.stack([unit(X[lab == j].sum(0)) for j in range(k)])
    raise RuntimeError('k-means did not converge')


def loo(E, art, k=15):
    """Leave-one-out mean chord to the k nearest other P renders, same artist excluded (the search's null)."""
    D = chord(E @ E.T)
    D[art[:, None] == art[None, :]] = np.inf
    return np.sort(D, axis=1)[:, :k].mean(1)


def step(b, d, phi):
    d = d - (d @ b) * b
    n = np.linalg.norm(d)
    assert n > 1e-6, 'direction parallel to the base'
    r = np.radians(phi)
    return np.cos(r) * b + np.sin(r) * d / n


def main(push, push2, starts, dest):
    P, P2, st = np.load(push, allow_pickle=True), np.load(push2, allow_pickle=True), np.load(starts, allow_pickle=True)
    Pv, Pa = unit(P['vec']), np.array([str(a) for a in P['artist']])
    P2v, P2a = unit(P2['vec']), np.array([str(a) for a in P2['artist']])
    assert Pv.shape == (2048, 768) and P2v.shape == (1000, 768)
    axis = unit(st['mean_pre'])
    own = {o: loo(unit(P[f'emb_{o}']), Pa) for o in OBS}
    q90 = {o: float(np.percentile(own[o], 90)) for o in OBS}
    rng = np.random.default_rng(SEED)
    lab, C = skmeans(Pv, K, rng)
    Vt = [np.linalg.svd(Pv[lab == j] - C[j], full_matrices=False)[2][:RANK] for j in range(K)]
    perm = rng.permutation(K)
    pairs = [(int(perm[i]), int(perm[(i + 1) % K])) for i in range(K)]

    def req_dist(x, arts):
        c = Pv @ x
        c = np.where(np.isin(Pa, arts), -np.inf, c)
        return float(chord(np.partition(c, -15)[-15:]).mean())

    recs = []

    def add(fam, x, pair, A, B, a=-1, b=-1, p=-1, q=-1, slot=-1, t=np.nan, p2=-1, arts=()):
        x = unit(x)
        rd, cone = req_dist(x, list(arts)), float(x @ axis)
        recs.append(dict(family=fam, v=x, pair=pair, region_a=A, region_b=B, a_row=a, b_row=b, p_row=p, q_row=q, slot=slot, t=t, p2_row=p2,
                         phi=PHI.get(fam, np.nan), angle_from_b=float(np.degrees(np.arccos(np.clip(x @ Pv[b], -1, 1)))) if b >= 0 else np.nan,
                         angle_from_a=float(np.degrees(np.arccos(np.clip(x @ Pv[a], -1, 1)))) if a >= 0 else np.nan,
                         cone_cos=cone, req_dist=rd, in_window=int(rd > RQ and cone >= CONE),
                         **{f'b_own_{o}': float(own[o][b]) if b >= 0 else np.nan for o in OBS},
                         b_ordinary=int(all(own[o][b] < q90[o] for o in OBS)) if b >= 0 else -1))

    slot = 0
    for pi, (A, B) in enumerate(pairs):
        a_rows = rng.choice(np.flatnonzero(lab == A), NSLOT, replace=False)
        used = set()
        for j, a in enumerate(a_rows):
            cand = np.array([r for r in np.flatnonzero(lab == B) if Pa[r] != Pa[a] and r not in used])
            b = int(rng.choice(cand))
            used.add(b)
            others = np.array([r for r in range(len(Pv)) if r not in (a, b)])
            p, q = (int(r) for r in rng.choice(others, 2, replace=False))
            va, vb = Pv[a], Pv[b]
            dirs = {'PP12': vb - va, 'PP35': vb - va, 'RD35': C[B] - C[A], 'SH35': (Vt[B] @ (vb - C[B])) @ Vt[B], 'GD35': Pv[p] - Pv[q]}
            kw = dict(pair=pi, A=A, B=B, a=int(a), b=b, slot=slot, arts=(Pa[b],))
            add('BASE', vb, **kw)
            for fam, d in dirs.items():
                add(fam, step(vb, d, PHI[fam]), p=p if fam == 'GD35' else -1, q=q if fam == 'GD35' else -1, **kw)
            if j < 4:
                add('IW', va + vb, pair=pi, A=A, B=B, a=int(a), b=b, slot=slot, arts=(Pa[a], Pa[b]))
            slot += 1
        for t in CL_T:
            add('CL', C[A] + t * (C[B] - C[A]), pair=pi, A=A, B=B, t=t)
    for r in rng.choice(len(P2v), 64, replace=False):
        add('P2RE', P2v[r], pair=-1, A=-1, B=-1, p2=int(r), arts=(P2a[r],))

    out = {k: np.array([r[k] for r in recs]) for k in recs[0]}
    n = len(recs)
    assert n == 120 * 6 + 48 + 36 + 64 == 868, n
    out['render_seed'] = 97000 + np.arange(n)
    for fam in ('PP12', 'PP35', 'RD35', 'SH35', 'GD35'):   # the requested angle from b is exact
        m = out['family'] == fam
        assert np.allclose(out['angle_from_b'][m], PHI[fam], atol=1e-4), fam
    assert np.allclose(np.linalg.norm(out['v'], axis=1), 1, atol=1e-9)
    summ = {'K': K, 'region_sizes': np.bincount(lab, minlength=K).tolist(), 'pairs': pairs,
            'centre_angle_deg': [float(np.degrees(np.arccos(np.clip(C[A] @ C[B], -1, 1)))) for A, B in pairs],
            'own_q90': q90, 'own_q99': {o: float(np.percentile(own[o], 99)) for o in OBS}, 'families': {}}
    L = out['family'] == 'PP35'
    for other in ('RD35', 'SH35', 'GD35'):   # how far apart the families' requests are within a slot
        m = out['family'] == other
        summ[f'angle_PP35_vs_{other}_median'] = float(np.median(np.degrees(np.arccos(np.clip((out['v'][L] * out['v'][m]).sum(1), -1, 1)))))
    for fam in dict.fromkeys(out['family']):
        m = out['family'] == fam
        summ['families'][str(fam)] = dict(n=int(m.sum()), median_cone_cos=float(np.median(out['cone_cos'][m])), median_req_dist=float(np.median(out['req_dist'][m])),
                                          in_window=int(out['in_window'][m].sum()), b_ordinary=int((out['b_ordinary'][m] == 1).sum()),
                                          median_angle_from_a=float(np.nanmedian(out['angle_from_a'][m])) if (out['a_row'][m] >= 0).any() else None)
    np.savez(dest, region_label=lab, region_centre=C, **out)
    json.dump(summ, open(dest.replace('.npz', '_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1), flush=True)
    print('wrote', dest, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
