"""Writes probe_requests.npz: 1,056 depth-1 requests (families R1, R05, R0, LR, L2, LT, CAL per start, then M3), drawn with rng 4242 and filtered by request distance and cone cosine.
Usage: python probe_frames.py [OUT_NPZ]   (CPU; data/ is put on the path for data.py)"""
import json
import os
import sys

import numpy as np

O = os.environ['DUROOM_OUT'] + '/sphere'
DATA_DIR = os.path.join(os.environ['DUROOM_REPO'], 'data')
FAMS = ('R1', 'R05', 'R0', 'LR', 'L2', 'LT')
ALPHA = {'R1': 1.0, 'R05': 0.5, 'R0': 0.0, 'CAL': 1.0}
NCAL = (13, 13, 13, 13, 12)
TS = (-0.5, 1.25, 1.5)   # line steps: back away from p, and beyond p
unit = lambda v: np.asarray(v, np.float64) / np.linalg.norm(np.asarray(v, np.float64), axis=-1, keepdims=True)  # noqa: E731
chord = lambda c: np.sqrt(np.clip(2 - 2 * c, 0, None))  # noqa: E731


def make_ctx(Xpre, Pv, Pa, rq=0.877, cone=0.506):
    """Xpre: unit pre-1900 CLIP rows (n, 768); Pv/Pa: P's unit request vectors and artists."""
    _, s, Vt = np.linalg.svd(Xpre - Xpre.mean(0), full_matrices=False)
    C = Pv @ Pv.T
    C[Pa[:, None] == Pa[None, :]] = -np.inf   # leave-one-out with the artist out (this also drops the row itself)
    d = chord(np.partition(C, -15, axis=1)[:, -15:]).mean(1)
    tail = np.argsort(-d)[:int(np.ceil(0.1 * len(Pv)))]
    return dict(Pv=Pv, Pa=Pa, Vt=Vt, sd=s / np.sqrt(len(Xpre) - 1), axis=unit(Xpre.mean(0)), tail=tail, rq=rq, cone=cone)


def sd_eff(ctx, alpha):
    """sd**alpha rescaled so the total variance equals the record's, whatever alpha (alpha 0 = isotropic at the same total spread)."""
    sd = ctx['sd']
    return sd ** alpha * np.sqrt((sd ** 2).sum() / (sd ** (2 * alpha)).sum())


def req_dist(ctx, x, art):
    c = ctx['Pv'] @ x
    if art is not None:
        c = np.where(ctx['Pa'] == art, -np.inf, c)
    return float(chord(np.partition(c, -15)[-15:]).mean())


def _line(x, ctx, rng, pool, art):
    cand = pool[ctx['Pa'][pool] != art] if art is not None else pool
    j, t = int(rng.choice(cand)), float(rng.choice(TS))
    return unit(x + t * (ctx['Pv'][j] - x)), j, t


def _draw(fam, x0, rng, ctx, art):
    nan, P3 = float('nan'), -np.ones(3, int)
    if fam in ALPHA:
        T = rng.uniform(0.3, 0.6) if fam == 'CAL' else rng.uniform(1.0, 1.5)
        g = rng.standard_normal(len(ctx['sd']))
        return unit(x0 + T * (g * sd_eff(ctx, ALPHA[fam])) @ ctx['Vt']), dict(alpha=ALPHA[fam], T=T, t=nan, t2=nan, partner_rows=P3)
    allP = np.arange(len(ctx['Pv']))
    if fam in ('LR', 'LT'):
        x, j, t = _line(x0, ctx, rng, allP if fam == 'LR' else ctx['tail'], art)
        return x, dict(alpha=nan, T=nan, t=t, t2=nan, partner_rows=np.array([j, -1, -1]))
    if fam == 'L2':
        x1, j, t = _line(x0, ctx, rng, allP, art)
        x, j2, t2 = _line(x1, ctx, rng, allP, art)
        return x, dict(alpha=nan, T=nan, t=t, t2=t2, partner_rows=np.array([j, j2, -1]))
    assert fam == 'M3'
    for _ in range(1000):   # pairwise angle >= 60 deg is part of the draw, not of F
        r = rng.choice(ctx['tail'], 3, replace=False)
        V = ctx['Pv'][r]
        if (V @ V.T)[np.triu_indices(3, 1)].max() <= 0.5:
            return unit(V.mean(0)), dict(alpha=nan, T=nan, t=nan, t2=nan, partner_rows=r)
    raise RuntimeError('no tail triple at 60 deg')


def make_frame(fam, x_from, rng, ctx, art=None):
    """One request of family fam from x_from; art's P works do not count as neighbours or partners (None for M3); CAL is unfiltered."""
    for tries in range(1, 201):
        x, rec = _draw(fam, x_from, rng, ctx, art)
        cone, rd = float(x @ ctx['axis']), req_dist(ctx, x, art)
        ok = fam == 'CAL' or (rd > ctx['rq'] and cone >= ctx['cone'])
        if ok:
            break
    return dict(rec, v=x, tries=tries, filter_fail=int(not ok), cone_cos=cone, req_dist=rd)


def build(ctx, starts):
    """starts: [(seed, row_full, x0, artist)]. The rng order is: per seed R1, R05, R0, LR, L2, LT, CAL; then M3."""
    rng, recs = np.random.default_rng(4242), []
    for seed, row, x0, art in starts:
        for fam, n in [(f, 32) for f in FAMS] + [('CAL', NCAL[seed - 1])]:
            for _ in range(n):
                r = make_frame(fam, x0, rng, ctx, art)
                recs.append(dict(r, family=fam, start_seed=seed, start_row=row, angle_from_start=float(np.degrees(np.arccos(np.clip(r['v'] @ x0, -1, 1))))))
    for _ in range(32):
        recs.append(dict(make_frame('M3', None, rng, ctx, None), family='M3', start_seed=0, start_row=-1, angle_from_start=float('nan')))
    out = {k: np.array([r[k] for r in recs]) for k in recs[0]}
    out['render_seed'] = 93000 + np.arange(len(recs))
    return out


def summary(out):
    s = {}
    for fam in dict.fromkeys(out['family']):
        m = out['family'] == fam
        s[str(fam)] = dict(n=int(m.sum()), median_cone_cos=float(np.median(out['cone_cos'][m])), median_req_dist=float(np.median(out['req_dist'][m])),
                           median_angle_from_start=float(np.nanmedian(out['angle_from_start'][m])) if (fam != 'M3') else None,
                           filter_fail=int(out['filter_fail'][m].sum()))
    return s


def main(dest):
    sys.path.insert(0, DATA_DIR)
    from data import CLIP, lens  # noqa: E402
    push, st = np.load(f'{O}/push_emb.npz', allow_pickle=True), np.load(f'{O}/pools/starts_pre1900.npz', allow_pickle=True)
    assert len(push['vec']) == 2048 and (st['year'] < 1900).all()
    Xpre = unit(lens(CLIP).astype(np.float64))[st['rows']]
    assert np.allclose(Xpre.mean(0), st['mean_pre'], atol=1e-6), 'corpus rows do not reproduce mean_pre'
    starts = []
    for s in (1, 2, 3, 4, 5):
        j = int(np.random.default_rng(s + 31337).integers(len(st['rows_full'])))   # search_v2's seedrow rule
        starts.append((s, int(st['rows_full'][j]), Xpre[j].copy(), str(st['artist'][j])))
    ctx = make_ctx(Xpre, unit(push['vec']), np.array([str(a) for a in push['artist']]))
    out = build(ctx, starts)
    assert len(out['v']) == 1056
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.savez(dest, **out)
    summ = summary(out)
    json.dump(summ, open(dest.replace('.npz', '_summary.json'), 'w'), indent=1)
    for k, v in summ.items():
        print(k, v, flush=True)
    print('wrote', dest, flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else f'{O}/eval/probe/probe_requests.npz')
