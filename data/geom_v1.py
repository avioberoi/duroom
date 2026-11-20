"""Data for three geometric maps per space (A regions by classical MDS, B1 two named straight directions, B2 azimuthal map centred on the past), from the Test 1 runs and the stored unit vectors.
Usage: python -B geom_v1.py -> $DUROOM_OUT/tests12/geom/geom_<space>_v1.json (written only after every check of that space passed; never overwrites)"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T1 = os.environ['DUROOM_OUT'] + '/tests12/t1'
OBS = os.environ['DUROOM_EMB']
DEL = os.environ['WIKIART_DIR'] + '/artwork_data/delete_ids.txt'
OUT = os.environ['DUROOM_OUT'] + '/tests12/geom'
SPACES = {  # space: (run dir, strict far threshold = B's 99th pct of S1_k15_excl, looser = B's 95th)
    'clip-vit-l-14': ('run_rob_clip', 0.8046082632076672, 0.7388103378444228),
    'dinov3-l': ('run_headline_dinov3', 0.8944054834757185, 0.7752205757788371),
    'siglip-so400m': ('run_headline_siglip', 0.8990353088223895, 0.8258338745487731),
}
NG = 70


def log(*a):
    print(*a, flush=True)


def unit(v):
    return v / np.linalg.norm(v)


def deg_between(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def sign_fix(v):
    """Sign rule: the component of largest absolute value is made positive."""
    return v if v[np.argmax(np.abs(v))] > 0 else -v


# ---------- classical MDS ----------
def cmds(D):
    D = np.array(D, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    Bm = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh((Bm + Bm.T) / 2)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    Y = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))
    return Y, w


def mds_fidelity(D, Y):
    iu = np.triu_indices(len(D), 1)
    d2 = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
    t = np.asarray(D)[iu]
    return {'n_pairs': int(len(t)), 'pearson': float(np.corrcoef(d2, t)[0, 1]), 'spearman': float(spearmanr(d2, t)[0]),
            'max_abs_error_deg': float(np.max(np.abs(d2 - t))), 'median_abs_error_deg': float(np.median(np.abs(d2 - t)))}


def procrustes_residual(P, Y):
    """Max distance after the best orthogonal map (rotation or reflection) of centred Y onto centred P."""
    Pc, Yc = P - P.mean(0), Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Yc.T @ Pc)
    R = U @ Vt
    return float(np.max(np.linalg.norm(Yc @ R - Pc, axis=1))), float(np.linalg.det(R))


def planted_mds(seed=0, n=32, dim=10):
    rng = np.random.default_rng(seed)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    P2 = 3.0 * np.c_[np.cos(ang), np.sin(ang)]
    Bq = np.linalg.qr(rng.standard_normal((dim, 2)))[0]
    Xs = P2 @ Bq.T + rng.standard_normal(dim)
    D = np.sqrt(((Xs[:, None] - Xs[None]) ** 2).sum(-1))
    Y, w = cmds(D)
    res, det = procrustes_residual(P2, Y)
    fid = mds_fidelity(D, Y)
    return {'n': n, 'max_procrustes_residual': res, 'det_of_fit': det, 'eig3_over_eig1': float(abs(w[2]) / w[0]), **fid,
            'passed': bool(res < 1e-8 and fid['max_abs_error_deg'] < 1e-8)}


# ---------- grids ----------
def grid(px, py, ex, ey, layers):
    """counts per layer on a 70 x 70 grid over the closed box; the last bin includes the upper edge; points outside the box are counted separately"""
    inr = (px >= ex[0]) & (px <= ex[-1]) & (py >= ey[0]) & (py <= ey[-1])
    ix = np.clip(np.searchsorted(ex, px, side='right') - 1, 0, NG - 1)
    iy = np.clip(np.searchsorted(ey, py, side='right') - 1, 0, NG - 1)
    out, chk = {}, {}
    for name, m in layers.items():
        H = np.zeros((NG, NG), dtype=np.int64)
        s = m & inr
        np.add.at(H, (ix[s], iy[s]), 1)
        o = int((m & ~inr).sum())
        assert int(H.sum()) + o == int(m.sum()), name
        out[name] = {'counts_ix_iy': H.tolist(), 'out_of_range': o, 'total': int(m.sum())}
        chk[name] = (int(H.sum()), o, int(m.sum()))
    return out, chk


def r6(a, d=6):
    return np.round(np.asarray(a, dtype=np.float64), d).tolist()


# ---------- styles ----------
SUFFIX = re.compile(r'\s*\([^()]*\)\s*$')


def parse_top5(s):
    out = []
    if not isinstance(s, str) or not s.strip():
        return out
    for part in s.split('; '):
        name, c = part.rsplit(':', 1)
        out.append((SUFFIX.sub('', name).strip(), int(c)))
    return out


def top_styles(row):
    tot = {}
    for col in ('top5_styles_A', 'top5_styles_later'):
        for name, c in parse_top5(row[col]):
            tot[name] = tot.get(name, 0) + c
    ranked = sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked


# ---------- one space ----------
def run_space(space, X, pw, allfit, csv, strict, loose, checks):
    gid = pw['id']
    g = pw['group_0A_1B_2later'].astype(int)
    year = pw['year'].astype(np.float64)
    S1 = pw['S1_k15_excl'].astype(np.float64)
    A, B, L = g == 0, g == 1, g == 2
    L19, L60 = L & (year < 1960), L & (year >= 1960)
    pre = A | B
    fit = A | L
    sizes = {'A': int(A.sum()), 'B': int(B.sum()), 'later': int(L.sum()), 'later_1900_1959': int(L19.sum()), 'later_1960_on': int(L60.sum())}
    assert np.isfinite(year[L]).all() and np.isfinite(S1).all()
    assert (allfit['Q_ids'] == gid).all()
    reg = allfit['Q_region'].astype(int)
    far_s, far_l = L & (S1 > strict), L & (S1 > loose)

    # mean direction of A and the per-work angle to it
    mA = X[A].mean(0)
    mu = mA / np.linalg.norm(mA)
    cmu = np.clip(X @ mu, -1, 1)
    theta_deg = np.degrees(np.arccos(cmu))
    dev = float(np.max(np.abs(theta_deg - pw['centre_deg'])))
    assert dev < 1e-9, dev
    checks['centre_deg_max_abs_dev_deg'] = dev

    # ---- Map A ----
    sub = csv[csv.fit == 'A_plus_later'].set_index('region').sort_index()
    assert list(sub.index) == list(range(32))
    C = np.zeros((32, X.shape[1]))
    regions = []
    cnt_dev = 0
    for k in range(32):
        mem = fit & (reg == k)
        mk = X[mem].mean(0)
        C[k] = mk / np.linalg.norm(mk)
        row = sub.loc[k]
        nA, nB = int((A & (reg == k)).sum()), int((B & (reg == k)).sum())
        n19, n60 = int((L19 & (reg == k)).sum()), int((L60 & (reg == k)).sum())
        share = nA / (nA + n19 + n60)
        mine = (int(mem.sum()), nA, nB, n19 + n60, n19, n60)
        theirs = (int(row.n_fit), int(row.n_A), int(row.n_B), int(row.n_later), int(row.n_later_1900_1959), int(row.n_later_1960_on))
        cnt_dev += int(mine != theirs) + int(int(allfit['n_fit'][k]) != mine[0]) + int(abs(share - row.pre1900_share_of_A_plus_later) > 1e-12)
        st = top_styles(row)
        regions.append({'id': k, 'top_styles': [{'style': s, 'count_in_top5_lists': c} for s, c in st[:2]],
                        'n_A': nA, 'n_B': nB, 'n_1900_1959': n19, 'n_1960_on': n60, 'n_fit_A_plus_later': mine[0],
                        'pre1900_share_of_A_plus_later': share, 'angle_deg_centre_to_A_mean_direction': deg_between(C[k], mu)})
    assert cnt_dev == 0, cnt_dev
    checks['region_counts_equal_csv'] = 'pass (n_fit, n_A, n_B, n_later, n_later_1900_1959, n_later_1960_on, pre1900 share; 32 regions)'
    rc_dev = float(np.max(np.abs(C - allfit['raw_centres'])))
    assert rc_dev < 1e-9, rc_dev
    checks['region_centres_equal_stored_raw_centres_max_abs_dev'] = rc_dev
    Dang = np.degrees(np.arccos(np.clip(C @ C.T, -1, 1)))
    np.fill_diagonal(Dang, 0.0)
    Ymds, w = cmds(Dang)
    fidA = mds_fidelity(Dang, Ymds)
    fidA['eig_top2_share_of_positive'] = float(w[:2].sum() / w[w > 0].sum())
    fidA['eig_negative_sum_over_positive_sum'] = float(abs(w[w < 0].sum()) / w[w > 0].sum())
    fidA['eigenvalues_top5'] = r6(w[:5], 4)
    for k in range(32):
        regions[k]['mds_xy_deg'] = r6(Ymds[k], 4)
    mapA = {'regions': regions, 'angle_matrix_deg': r6(Dang, 4), 'fidelity': fidA}

    # ---- Map B1 ----
    m60 = X[L60].mean(0)
    xax = unit(m60 - mA)
    Pa = X[A] - mA
    wA, VA = np.linalg.eigh(Pa.T @ Pa)
    pc1 = sign_fix(VA[:, -1])
    cos_pc1_x = float(pc1 @ xax)
    y0 = pc1 - (pc1 @ xax) * xax
    yax = unit(y0)
    orth = float(max(abs(xax @ xax - 1), abs(yax @ yax - 1), abs(xax @ yax)))
    assert orth < 1e-9
    checks['B1_axes_orthonormal_max_dev'] = orth
    P = X - mA
    bx, by = P @ xax, P @ yax
    var_share = {}
    for name, m in (('A', A), ('B', B), ('later', L), ('later_1900_1959', L19), ('later_1960_on', L60), ('all', np.ones(len(g), bool))):
        Pc = P[m] - P[m].mean(0)
        tot = float((Pc ** 2).sum())
        vx, vy = float(((Pc @ xax) ** 2).sum()), float(((Pc @ yax) ** 2).sum())
        var_share[name] = {'x': vx / tot, 'y': vy / tot, 'x_plus_y': (vx + vy) / tot}
    ex = np.linspace(*np.percentile(bx, [0.5, 99.5]), NG + 1)
    ey = np.linspace(*np.percentile(by, [0.5, 99.5]), NG + 1)
    layers = {'pre1900_A_plus_B': pre, 'later_1900_1959': L19, 'later_1960_on': L60,
              'later_far_strict': far_s, 'later_far_loose': far_l}
    gB1, chkB1 = grid(bx, by, ex, ey, layers)
    era_masks = {'A': A, 'B': B, 'pre1900_A_plus_B': pre, 'later_1900_1959': L19, 'later_1960_on': L60, 'later_all': L}
    mapB1 = {'axes': {'cos_pc1_with_x_before_orthogonalising': cos_pc1_x, 'pc1_eigenvalue_share_of_A_variance': float(wA[-1] / wA.sum()),
                      'norm_of_mean_difference_1960on_minus_A': float(np.linalg.norm(m60 - mA))},
             'variance_share_held_by_axes': var_share,
             'grid': {'x_edges': ex.tolist(), 'y_edges': ey.tolist(), 'layers': gB1},
             'region_centres': [{'id': k, 'unit_centre_xy': r6([(C[k] - mA) @ xax, (C[k] - mA) @ yax]),
                                 'members_mean_xy': r6([bx[fit & (reg == k)].mean(), by[fit & (reg == k)].mean()])} for k in range(32)],
             'era_means_xy': {e: r6([bx[m].mean(), by[m].mean()]) for e, m in era_masks.items()}}

    # ---- Map B2 ----
    th = np.arccos(cmu)
    t = X - cmu[:, None] * mu[None, :]
    u = t / np.linalg.norm(t, axis=1, keepdims=True)
    v = th[:, None] * u                                    # log map at mu (radians)
    M = v[fit].T @ v[fit]
    wT, VT = np.linalg.eigh(M)
    e1, e2 = sign_fix(VT[:, -1]), sign_fix(VT[:, -2])
    e1 = unit(e1 - (e1 @ mu) * mu)
    e2 = e2 - (e2 @ mu) * mu
    e2 = unit(e2 - (e2 @ e1) * e1)
    orth2 = float(max(abs(e1 @ e1 - 1), abs(e2 @ e2 - 1), abs(e1 @ e2), abs(e1 @ mu), abs(e2 @ mu)))
    assert orth2 < 1e-9
    checks['B2_directions_orthonormal_and_tangent_max_dev'] = orth2
    a1, a2 = v @ e1, v @ e2
    phi = np.arctan2(a2, a1)
    zx, zy = theta_deg * np.cos(phi), theta_deg * np.sin(phi)
    sq = (v ** 2).sum(1)
    inplane = (a1 ** 2 + a2 ** 2)
    share_fit = float((wT[-1] + wT[-2]) / np.trace(M))
    per_group_share = {}
    for name, m in (('A', A), ('B', B), ('later', L), ('later_1900_1959', L19), ('later_1960_on', L60), ('fit_A_plus_later', fit)):
        per_group_share[name] = {'sum_share': float(inplane[m].sum() / sq[m].sum()), 'median_per_work_share': float(np.median(inplane[m] / sq[m]))}
    Rx = float(np.percentile(theta_deg, 99.5))
    e = np.linspace(-Rx, Rx, NG + 1)
    gB2, chkB2 = grid(zx, zy, e, e, layers)

    def azi(c):
        cc = float(np.clip(c @ mu, -1, 1))
        tt = c - cc * mu
        vv = np.arccos(cc) * tt / np.linalg.norm(tt)
        p = np.arctan2(vv @ e2, vv @ e1)
        r = np.degrees(np.arccos(cc))
        return [r * np.cos(p), r * np.sin(p)]

    rad_pct = {nm: r6(np.percentile(theta_deg[m], [10, 50, 90]), 4) for nm, m in era_masks.items()}
    mapB2 = {'tangent_pca': {'share_of_fit_tangent_variance_in_2_directions': share_fit, 'eig1_share': float(wT[-1] / np.trace(M)),
                             'eig2_share': float(wT[-2] / np.trace(M)), 'share_by_group': per_group_share},
             'grid': {'x_edges': e.tolist(), 'y_edges': e.tolist(), 'extent_deg': Rx, 'layers': gB2},
             'region_centres': [{'id': k, 'radius_deg': deg_between(C[k], mu), 'unit_centre_xy_deg': r6(azi(C[k]), 4),
                                 'members_mean_xy_deg': r6([zx[fit & (reg == k)].mean(), zy[fit & (reg == k)].mean()], 4)} for k in range(32)],
             'radius_deg_p10_p50_p90': rad_pct,
             'era_means_xy_deg': {nm: r6([zx[m].mean(), zy[m].mean()], 4) for nm, m in era_masks.items()}}
    checks['grid_totals'] = {'B1': chkB1, 'B2': chkB2}
    counts = {**sizes, 'pre1900_A_plus_B': int(pre.sum()), 'later_far_strict': int(far_s.sum()), 'later_far_loose': int(far_l.sum()), 'all_works': int(len(g))}
    return {'sizes': counts, 'mapA': mapA, 'mapB1': mapB1, 'mapB2': mapB2}


NOTES = [
    'Works = the 122,499 rows of <space>_H1_perwork.npz (Test 1, split 1): A = fit half of the pre-1900 artists (28,404), B = held-out half (24,627), later = scored works dated 1900 on whose artist is not in A (69,468 = 35,874 dated 1900-1959 + 33,594 dated 1960 on). Vectors = stored vectors of the original images, float64, rows renormalised to unit length (as t1_v2.py).',
    'Far = S1_k15_excl (mean chord to the 15 nearest A works, own artist excluded) above the strict line (B 99th percentile) or the looser line (B 95th percentile); far layers count later works only; strict > comparison.',
    'A mean direction mu = unit(mean of A unit vectors). Mean of A mA = mean of A unit vectors (not renormalised).',
    'Map A: 32 regions = the all-works fit of Test 1 H2 (spherical k-means on A plus later works, seed 1, from <space>_H2_allfit_K32.npz Q_region). Region centre = unit mean of the unit vectors of its fit works (A plus later; B works are NOT members of the centre), equal to the stored raw_centres. angle_deg_centre_to_A_mean_direction = angle between that centre and mu. Position = classical MDS (double-centred squared angles, top 2 eigenvectors) of the 32 x 32 matrix of angles (degrees) between centres; mds_xy_deg in degrees. Fidelity over the 496 pairs: 2-D distance vs true angle.',
    'Map A top_styles: counts summed style by style over the CSV columns top5_styles_A and top5_styles_later only (truncated top-5 lists, so counts are partial); a trailing parenthetical is trimmed (e.g. "Art Nouveau (Modern)" -> "Art Nouveau", "Mannerism (Late Renaissance)" -> "Mannerism"); ties broken alphabetically.',
    'Map B1: x = unit(mean of 1960-on unit vectors - mA). y = first principal direction of A (eigenvector of the scatter of A unit vectors about mA; sign: largest-|component| positive), made orthogonal to x and renormalised. Coordinates of a work = (vector - mA) dot x, dot y (same units as the unit vectors: chord-scale). variance_share_held_by_axes: per group, sum of squared x and y coordinates about the group\'s own mean over the group\'s total squared offset from its own mean (all dimensions).',
    'Map B1 grid: 70 x 70 equal bins from the 0.5th to 99.5th percentile of all 122,499 works on each axis; counts_ix_iy[i][j] = works with x in bin i and y in bin j; bin i = [edge_i, edge_i+1), last bin closed; out_of_range = outside the box. region_centres: unit_centre_xy = projection of the unit centre of Map A; members_mean_xy = mean coordinates of the region\'s fit works (A plus later). era_means_xy = mean coordinates of the era\'s works (= projection of the era\'s mean vector).',
    'Map B2: azimuthal (equidistant) map centred on mu. Radius = exact angle in degrees between a work\'s unit vector and mu (equals perwork centre_deg). Offsets = log map at mu: angle (radians) times the unit direction of the work\'s component orthogonal to mu. The two directions e1, e2 = top two eigenvectors of the UNCENTRED second-moment matrix of the offsets of A plus later works (about mu, the map centre; sign rule as B1), re-orthogonalised to mu and each other. Direction angle = atan2(offset.e2, offset.e1); position = radius * (cos, sin). The radius is exact; the direction uses only the two plane components.',
    'Map B2 grid: 70 x 70 equal bins over [-R, R] on both axes, R = 99.5th percentile of the radius over all works (extent_deg). region_centres: unit_centre_xy_deg = the azimuthal image of the Map A unit centre (radius_deg = Map A angle); members_mean_xy_deg = mean map position of its fit works. radius_deg_p10_p50_p90 per era; era_means_xy_deg = mean map position of the era\'s works.',
    'share_of_fit_tangent_variance_in_2_directions = (top two eigenvalues) / trace over A plus later works (uncentred, about mu); share_by_group = in-plane squared offset over total squared offset (sum ratio, and median per-work ratio).',
]


def main():
    os.makedirs(OUT, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'nojob')
    log(f'job {job}')
    planted = planted_mds()
    log('planted MDS check (32 irregular points on a circle in R^10, Euclidean distances):', json.dumps(planted))
    assert planted['passed']
    dele = set(np.loadtxt(DEL, dtype=int).tolist())
    ids0 = None
    for space, (rd, strict, loose) in SPACES.items():
        t0 = time.time()
        d = f'{T1}/{rd}'
        files = {'perwork': f'{d}/{space}_H1_perwork.npz', 'allfit': f'{d}/{space}_H2_allfit_K32.npz', 'regions_csv': f'{d}/{space}_H2_regions_K32.csv',
                 'centres_radii': f'{d}/{space}_centres_radii.npz', 'vectors': f'{OBS}/{space}.npz'}
        log(f'\n===== {space}: inputs', json.dumps(files))
        pw = dict(np.load(files['perwork'], allow_pickle=False))
        allfit = dict(np.load(files['allfit'], allow_pickle=False))
        cr = np.load(files['centres_radii'], allow_pickle=False)
        csv = pd.read_csv(files['regions_csv'])
        gid = pw['id'].astype(np.int64)
        assert len(np.unique(gid)) == len(gid)
        if ids0 is None:
            ids0 = (gid.copy(), pw['group_0A_1B_2later'].copy())
        else:
            assert (gid == ids0[0]).all() and (pw['group_0A_1B_2later'] == ids0[1]).all(), 'ids/groups differ across spaces'
        assert (cr['Q_ids'] == gid).all() and (cr['Q_group_0A_1B_2later'] == pw['group_0A_1B_2later']).all()
        assert not (set(gid.tolist()) & dele)
        z = np.load(files['vectors'])
        oid = z['ids']
        row = {int(i): r for r, i in enumerate(oid)}
        assert len(row) == len(oid)
        rows = np.array([row[int(i)] for i in gid])
        X = z['embeddings'][rows].astype('float64')
        assert np.isfinite(X).all()
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        checks = {}
        res = run_space(space, X, pw, allfit, csv, strict, loose, checks)
        checks['planted_mds'] = planted
        checks['ids_and_groups_identical_across_spaces'] = 'pass'
        doc = {'space': space, 'role': 'search space' if space == 'clip-vit-l-14' else 'observer', 'version': 'v1', 'job': job,
               'script': os.path.abspath(__file__), 'inputs': files,
               'thresholds': {'strict_B_p99_S1_k15_excl': strict, 'loose_B_p95_S1_k15_excl': loose},
               'sizes': res['sizes'], 'notes': NOTES, 'checks': checks,
               'fidelity': {'mapA_mds': res['mapA']['fidelity'], 'mapB1_variance_share': res['mapB1']['variance_share_held_by_axes'],
                            'mapB2_tangent_share': res['mapB2']['tangent_pca']},
               'mapA': res['mapA'], 'mapB1': res['mapB1'], 'mapB2': res['mapB2']}
        log(f'{space} checks:', json.dumps({k: v for k, v in checks.items() if k != 'grid_totals'}))
        log(f'{space} grid totals (in box, out of range, total):', json.dumps(checks['grid_totals']))
        log(f'{space} fidelity:', json.dumps(doc['fidelity']))
        path = f'{OUT}/geom_{space}_v1.json'
        with open(path, 'x') as f:
            json.dump(doc, f, separators=(',', ':'))
        log(f'{space}: wrote {path} bytes {os.path.getsize(path)} ({time.time() - t0:.0f}s)')
        del X, z
    log('done')


if __name__ == '__main__':
    sys.exit(main())
