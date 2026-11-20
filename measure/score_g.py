"""Per run: loadings of the requested vectors on the record's directions (shift, axes 1 and 2), fidelity to the request, displacement from the start, years of the nearest works (judge lens), admitted pictures split by exact fidelity against P's own q99, paired gaps between arms.
Usage: score_g.py OUT_JSON --pools starts_pre1900.npz --axes newaxis.json --p-emb push_emb.npz --runs ARM:SEED=file ... --emb arm_sSEED=emb.npz ... [--data-dir DIR with data.py]   (SCORE_JUDGE=dinov3-l reads the years in DINOv3)"""
import argparse
import json
import os
import sys

import numpy as np

GAP_FOUND, GAP_WEAK = 0.05, 0.02
OBS = os.environ['DUROOM_EMB']
DINO, JUDGE, COORD = 'dinov3-l', os.environ.get('SCORE_JUDGE', 'aimv2-huge'), 'clip-vit-l-14'   # JUDGE: the lens whose corpus file gives the nearest work's year; aimv2-huge needs an AIMv2 corpus file
BINS = [0, 0.5, 0.7, 0.9, 1.2, 9]
chord = lambda cos: np.sqrt(np.clip(2 - 2 * cos, 0, None))    # noqa: E731
unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)    # noqa: E731
med = lambda v: float(np.median(v)) if len(v) else None    # noqa: E731


def load_run(path):
    z = np.load(path, allow_pickle=True)
    V = unit(z['style_embs'].astype(np.float64))    # requested unit vectors; row 0 is the start
    n = len(V)
    out = {'paths': [str(p) for p in z['paths']], 'V': V, 'x0': V[0],
           'admitted': np.asarray(z['admitted'], bool) if 'admitted' in z.files else np.zeros(n, bool),
           'angle': np.asarray(z['angle_deg'], np.float64) if 'angle_deg' in z.files else np.full(n, np.nan),
           'raw': {k[4:]: np.asarray(z[k], np.float64) for k in z.files if k.startswith('raw_')},
           'q99': {k[9:]: float(z[k]) for k in z.files if k.startswith('norm_q99_')}}
    out['admitted'][0] = False
    E = unit(z[f'emb_{DINO}'].astype(np.float64)) if f'emb_{DINO}' in z.files else None
    out['disp'] = chord(E @ E[0]) if E is not None else np.full(n, np.nan)
    return out


def table(sel, L, r, fid, years):
    """Medians over the selected rows of each direction's loading and the item fields."""
    t = {'n': int(sel.sum())}
    for k, v in L.items():
        t[f'median_{k}'] = med(v[sel])
    t['median_fidelity'] = med(fid[sel][np.isfinite(fid[sel])])
    t['median_disp'] = med(r['disp'][sel][np.isfinite(r['disp'][sel])])
    t['median_angle_req'] = med(r['angle'][sel][np.isfinite(r['angle'][sel])])
    for n_, raw in r['raw'].items():
        t[f'median_raw_{n_}'] = med(raw[sel])
    if years is not None:
        y = years[sel]
        y = y[np.isfinite(y)]
        t['median_nearest_year'] = med(y)
        t['share_nearest_1900_1969'] = float(np.mean((y >= 1900) & (y < 1970))) if len(y) else None
        t['share_nearest_after_1900'] = float(np.mean(y >= 1900)) if len(y) else None
    return t


def main(a):
    pools = np.load(a.pools, allow_pickle=True)
    mean_pre = pools['mean_pre'].astype(np.float64)
    ax = json.load(open(a.axes))['cutoffs']['1900']['directions']
    dirs = {k: np.asarray(ax[k], np.float64) for k in ('shift', 'axis1', 'axis2')}
    P = np.load(a.p_emb, allow_pickle=True)
    if 'fidelity_clip' not in P.files:
        raise SystemExit('P embedding file has no fidelity_clip (run attach_artist.py first): the faithful/unfaithful split needs P\'s own q99')
    fid_bar = float(np.percentile(P['fidelity_clip'], 99))
    emb = {}
    for spec in a.emb or []:
        tag, f = spec.split('=', 1)
        emb[tag] = np.load(f, allow_pickle=True)
    sys.path.insert(0, a.data_dir)
    from data import lens, year    # noqa: E402
    judge = (unit(lens(f'{OBS}/{JUDGE}.npz').astype(np.float32)), year)
    res = {'gap_found': GAP_FOUND, 'gap_weak': GAP_WEAK, 'fidelity_bar_P_q99': fid_bar, 'bins': BINS, 'runs': {}}
    for spec in a.runs:
        tag, f = spec.split('=', 1)
        arm, seed = tag.split(':')
        key = f'{arm}_s{seed}'
        r = load_run(f)
        C = r['V'] - mean_pre
        L = {k: C @ d for k, d in dirs.items()}
        n = len(r['V'])
        fid, years = np.full(n, np.nan), None
        if key in emb:
            z = emb[key]
            pos = {str(p): i for i, p in enumerate(z['paths'])}
            idx = np.array([pos.get(p, -1) for p in r['paths']])
            ok = idx >= 0
            if f'emb_{COORD}' in z.files:
                R = unit(np.asarray(z[f'emb_{COORD}'], np.float64))
                fid[ok] = chord((R[idx[ok]] * r['V'][ok]).sum(1))
            if judge is not None and f'emb_{JUDGE}' in z.files:
                Rj = unit(np.asarray(z[f'emb_{JUDGE}'], np.float32))[idx[ok]]
                near = np.argmax(Rj @ judge[0].T, axis=1)
                years = np.full(n, np.nan)
                years[ok] = judge[1][near]
            cover = float(ok.mean())
        else:
            cover = 0.0
        cand = np.ones(n, bool)
        cand[0] = False
        adm = r['admitted']
        # the start's own artist has renders in P (cap 3); P's null excluded same-artist rows but candidates' scores did not: recompute every
        # candidate's raw 15-NN to P with the start's artist's rows masked, and count admissions that would change under the same q99 bar
        start_rows = pools['rows_full'] if 'rows_full' in pools.files else None
        masked = {}
        if start_rows is not None and 'artist' in P.files:
            z0 = np.load(f, allow_pickle=True)
            sidx = int(z0['start_idx'])
            hit = np.flatnonzero(start_rows == sidx)
            start_artist = str(pools['artist'][hit[0]]) if len(hit) else None
            keep_P = np.asarray(P['artist']) != start_artist if start_artist else np.ones(len(P['paths']), bool)
            for name in ('dinov3-l', 'siglip-so400m'):
                if f'emb_{name}' in z0.files and f'emb_{name}' in P.files:
                    Er = unit(np.asarray(z0[f'emb_{name}'], np.float64))
                    Ep = unit(np.asarray(P[f'emb_{name}'], np.float64))[keep_P]
                    masked[name] = np.sort(chord(Er @ Ep.T), axis=1)[:, :15].mean(1)
            if masked and r['q99']:
                pass_masked = np.all([masked[nm] > r['q99'][nm] for nm in masked if nm in r['q99']], axis=0)
                pass_orig = np.all([r['raw'][nm] > r['q99'][nm] for nm in masked if nm in r['raw'] and nm in r['q99']], axis=0)
                masked_summary = {'start_artist': start_artist, 'P_rows_masked': int((~keep_P).sum()), 'candidates_changing_pass': int((pass_masked != pass_orig)[1:].sum()),
                                  'median_raw_dinov3_masked': med(masked['dinov3-l'][1:]) if 'dinov3-l' in masked else None,
                                  'median_raw_dinov3_orig': med(r['raw']['dinov3-l'][1:]) if 'dinov3-l' in r['raw'] else None,
                                  # pass rate against P alone (the run's raw scores were against P plus the growing archive)
                                  'pass_rate_vs_P_alone': float(pass_masked[1:].mean()), 'pass_rate_as_run': float(pass_orig[1:].mean())}
            else:
                masked_summary = None
        else:
            masked_summary = None
        # pass rate by requested angle from the start (5-degree bins), for the matched-angle contrast between arms
        ang = r['angle']
        by_angle = []
        for lo in range(0, 95, 5):
            m = cand & (ang >= lo) & (ang < lo + 5)
            if m.sum() >= 10:
                by_angle.append({'bin': [lo, lo + 5], 'n': int(m.sum()), 'admitted_rate': float(adm[m].mean()),
                                 'pass_rate_vs_P_alone': float(pass_masked[m].mean()) if masked_summary else None,
                                 'median_fidelity': med(fid[m][np.isfinite(fid[m])]), 'median_shift': med(L['shift'][m])})
        faithful = adm & np.isfinite(fid) & (fid <= fid_bar)
        unfaithful = adm & np.isfinite(fid) & (fid > fid_bar)
        if cover < 1:
            print(f'WARNING {key}: embedding covers {cover:.1%} of the run rows; rows without an embedding have no fidelity or year', flush=True)
        row = {'file': f, 'n_rows': n, 'embedding_coverage': cover, 'n_admitted': int(adm.sum()), 'n_admitted_no_fidelity': int((adm & ~np.isfinite(fid)).sum()),
               'admission_rate': float(adm[1:].mean()), 'P_q99': r['q99'], 'start_artist_masked': masked_summary, 'by_requested_angle': by_angle,
               'a_all_candidates': table(cand, L, r, fid, years),
               'b_admitted_faithful': table(faithful, L, r, fid, years),
               'c_admitted_unfaithful': table(unfaithful, L, r, fid, years),
               'a_by_displacement': []}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = cand & (r['disp'] >= lo) & (r['disp'] < hi)
            row['a_by_displacement'].append({'bin': [lo, hi], **table(m, L, r, fid, years)})
        res['runs'][key] = row
    # paired gaps per seed (main minus each control) on the median shift loading: a all candidates, b faithful admitted, a_matched per displacement
    # bin with at least MIN_BIN items in both arms (median over bins)
    MIN_BIN, MIN_SEEDS = 20, 3
    seeds = sorted({k.split('_s')[1] for k in res['runs']})
    arms = sorted({k.split('_s')[0] for k in res['runs']})
    res['gaps'] = {}

    def gap_of(m, c, group):
        if group == 'a_matched':
            g = [bm['median_shift'] - bc['median_shift'] for bm, bc in zip(m['a_by_displacement'], c['a_by_displacement'])
                 if bm['n'] >= MIN_BIN and bc['n'] >= MIN_BIN and bm['median_shift'] is not None and bc['median_shift'] is not None]
            return (float(np.median(g)), len(g)) if g else None
        if m[group]['median_shift'] is None or c[group]['median_shift'] is None:
            return None
        return (m[group]['median_shift'] - c[group]['median_shift'], min(m[group]['n'], c[group]['n']))

    for group in ('a_all_candidates', 'a_matched', 'b_admitted_faithful'):
        for ctrl in [x for x in arms if x != 'main']:
            gaps, ns = [], []
            for s in seeds:
                m, c = res['runs'].get(f'main_s{s}'), res['runs'].get(f'{ctrl}_s{s}')
                g = gap_of(m, c, group) if m and c else None
                if g is not None:
                    gaps.append(g[0])
                    ns.append(g[1])
            if gaps:
                wins, pooled = sum(g > 0 for g in gaps), float(np.median(gaps))
                verdict = ('found' if wins == len(gaps) and len(gaps) >= 5 and pooled >= GAP_FOUND
                           else 'weak' if (len(gaps) >= MIN_SEEDS and wins >= len(gaps) - 1 and pooled >= GAP_WEAK)
                           else 'none' if len(gaps) >= MIN_SEEDS else 'too few seeds')
                res['gaps'][f'{group}:main_minus_{ctrl}'] = {'seeds': len(gaps), 'seeds_total': len(seeds), 'wins': int(wins), 'pooled_median_gap': pooled,
                                                             'gaps': [float(g) for g in gaps], 'n_per_seed': ns, 'verdict': verdict}
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)
    f3 = lambda v: 'na' if v is None else f'{v:.3f}'    # noqa: E731
    for k, r in res['runs'].items():
        A, Bf, Cu = r['a_all_candidates'], r['b_admitted_faithful'], r['c_admitted_unfaithful']
        print(f"{k}: rows {r['n_rows']} coverage {r['embedding_coverage']:.0%} admitted {r['n_admitted']} ({r['admission_rate']:.1%}, {r['n_admitted_no_fidelity']} without fidelity); "
              f"all: shift {f3(A['median_shift'])} axis1 {f3(A['median_axis1'])} fid {f3(A['median_fidelity'])} disp {f3(A['median_disp'])} near1900-69 {A.get('share_nearest_1900_1969')}; "
              f"faithful admitted n {Bf['n']} shift {f3(Bf['median_shift'])}; unfaithful admitted n {Cu['n']} shift {f3(Cu['median_shift'])}")
    print('fidelity bar (P q99)', fid_bar, '| gaps', {k: (v['wins'], v['seeds'], round(v['pooled_median_gap'], 3), v['verdict']) for k, v in res['gaps'].items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--pools', required=True)
    p.add_argument('--axes', required=True)
    p.add_argument('--p-emb', required=True)
    p.add_argument('--runs', nargs='+', required=True, help='ARM:SEED=candidate file; arms main, rf, b0')
    p.add_argument('--emb', nargs='*', default=None, help='arm_sSEED=embed_images.npz covering that run\'s paths')
    p.add_argument('--data-dir', default=os.environ['DUROOM_REPO'] + '/data')
    main(p.parse_args())
