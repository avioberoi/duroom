"""Per-work numbers behind the corpus figures: sampled works with angle, coordinates and decade; the time and East Asian groups against their bases (views, pooled kept shares); per-work kept and missed squared lengths; leftover share per work. Seed 0.
Usage: figdata.py OUT_DIR"""
import json
import os
import shutil
import sys
import tempfile

OUT = sys.argv[1]
TMP = tempfile.mkdtemp(dir=os.environ.get('SLURM_TMPDIR') or None)    # common writes thumbnails at import; keep them out of OUT
import atexit    # noqa: E402
atexit.register(shutil.rmtree, TMP, True)    # removed on failure too
sys.argv = ['common.py', TMP, '1', 'artist']    # common reads its arguments at import
import numpy as np    # noqa: E402

import common    # noqa: E402
from basis import KS, capped, code    # noqa: E402

X, named, dated, year = common.X, common.named, common.dated, common.year
SEED = 0
rng = np.random.default_rng(SEED)
K, K2, K32 = 32, KS.index(2), KS.index(32)
corpus = np.flatnonzero(named & (year >= 1000) & (year <= 2026))

decade = lambda r: [None if np.isnan(y) else int(y // 10 * 10) for y in year[r]]    # noqa: E731  (the cluster groups of common.py include undated works)
rnd = lambda a: np.round(a, 4).tolist()    # noqa: E731
PCT = [0, 1, 5, 25, 50, 75, 95, 99, 100]
out = {'provenance': {'script': 'figdata.py', 'lens': os.path.basename(common.LENS), 'unit_vectors': True, 'seed': SEED,
                      'cap_per_artist': 20, 'k': K, 'privacy': 'no ids, titles or artists; group label, decade and coordinates only'}}


def var_share(Y, m, dirs):
    """Share of the total variance of Y around m along each direction (rows of dirs)."""
    c = Y - m
    return [float(((c @ d) ** 2).sum() / (c ** 2).sum()) for d in dirs]


def spread(xy):
    a = np.asarray(xy)
    return {'centre': rnd(a.mean(0)), 'sd': rnd(a.std(0))}


# sphere: corpus basis, shared with leftover
m0, V0 = code(X[corpus])
mdir = m0 / np.linalg.norm(m0)
ang_all = np.degrees(np.arccos(np.clip(X[corpus] @ mdir, -1, 1)))
s = rng.choice(corpus, 3000, replace=False)
out['sphere'] = {'n_population': int(len(corpus)), 'n_sample': 3000, 'mean_vector_norm': float(np.linalg.norm(m0)),
                 'angle_percentiles_population': dict(zip(map(str, PCT), rnd(np.percentile(ang_all, PCT)))),
                 'angle_percentiles_sample': dict(zip(map(str, PCT), rnd(np.percentile(ang_all[np.searchsorted(corpus, s)], PCT)))),
                 'sample_angle_deg': rnd(ang_all[np.searchsorted(corpus, s)]), 'sample_xy': rnd((X[s] - m0) @ V0[:2].T),
                 'sample_decade': decade(s), 'variance_share_dir1_dir2': var_share(X[corpus], m0, V0[:2]),
                 'axes': 'xy = position on the corpus basis directions 1 and 2, measured from the corpus mean (inside the sphere)'}


def move(name, basis_rows, test_rows):
    fit = capped(basis_rows, 20, rng)
    T = capped(test_rows, 20, rng)
    m, V = code(X[fit])
    B = capped(np.setdiff1d(np.flatnonzero(named), T), 20, rng)[:len(fit)]
    assert len(B) == len(fit)
    mB, VB = code(X[B])
    mT = X[T].mean(0)
    res = (X[T] - m) - ((X[T] - m) @ V[:K].T) @ V[:K]    # what the first 32 basis directions miss of each test work
    rc = res - res.mean(0)
    r1 = np.linalg.eigh(rc.T @ rc)[1][:, -1]    # leading direction of the residuals' spread, outside the 32 by construction
    assert np.abs(V[:K] @ r1).max() < 1e-8    # outside the first 32 basis directions, checked
    mean_dot = float(res.mean(0) @ r1)
    ratio = abs(mean_dot) / float((rc @ r1).std())
    if ratio >= 0.1:
        r1 *= np.sign(mean_dot)    # the test group's mean residual is positive along it
        sign_rule = 'mean residual positive'
    else:
        r1 *= np.sign(r1[np.argmax(np.abs(r1))])    # mean residual too close to zero to fix a sign; largest component positive
        sign_rule = 'largest component positive (mean residual near zero)'
    a = rng.choice(fit, min(2000, len(fit)), replace=False)
    t = rng.choice(T, min(2000, len(T)), replace=False)
    view = lambda rows, ref, axes: rnd((X[rows] - ref) @ np.asarray(axes).T)    # noqa: E731
    v12, v1r = V[:2], np.vstack([V[0], r1])
    pooled = {f'{who}_{ref}_k{k}': float(common.kept(X[T], mm if ref == 'basis_mean' else mT, VV)[i])
              for who, mm, VV in (('unseen', m, V), ('random', mB, VB)) for ref in ('basis_mean', 'own_mean') for k, i in ((2, K2), (32, K32))}
    views = {}
    for vname, axes in (('dir1_dir2', v12), ('dir1_residual1', v1r)):
        views[vname] = {'basis_group_from_basis_mean': view(a, m, axes), 'test_from_basis_mean': view(t, m, axes), 'test_from_own_mean': view(t, mT, axes),
                        'centres_and_spread_full': {'basis_group': spread((X[fit] - m) @ axes.T), 'test_from_basis_mean': spread((X[T] - m) @ axes.T),
                                                    'test_from_own_mean': spread((X[T] - mT) @ axes.T)}}
    return {'name': name, 'n_fit': int(len(fit)), 'n_test': int(len(T)), 'n_random_basis': int(len(B)), 'n_sample_basis_group': int(len(a)), 'n_sample_test': int(len(t)),
            'basis_group_decade': decade(a), 'test_decade': decade(t), 'views': views, 'pooled_kept': pooled,
            'variance_share_basis_dir1_dir2_in_basis_group': var_share(X[fit], m, V[:2]),
            'squared_length_share_of_test_group_from_basis_mean': {'dir1': var_share(X[T], m, V[:1])[0], 'dir2': var_share(X[T], m, V[1:2])[0], 'residual1': var_share(X[T], m, r1[None])[0],
                                                                   'note': 'share of squared length from the basis mean, so it includes the mean offset; not a variance'},
            'residual1_share_of_centred_test_residual_spread': float(((rc @ r1) ** 2).sum() / (rc ** 2).sum()),
            'residual1_sign': {'rule': sign_rule, 'mean_residual_dot': mean_dot, 'ratio_to_sd': ratio},
            'n_undated_test': int(np.isnan(year[T]).sum()), 'n_undated_sample_test': int(np.isnan(year[t]).sum()),
            'axes_note': 'dir1, dir2 = the basis group\'s first two of 32 directions; residual1 = leading direction of the test group\'s residual outside those 32; '
                         'signs of dir1/dir2 are arbitrary, do not compare signs across panels; random basis drawn from the corpus minus the test set, cap 20'}


rows = np.flatnonzero(named & dated)    # the population of common.py's time groups
later, earlier = rows[year[rows] >= 1900], rows[year[rows] < 1900]
card = common.cluster_cards[15]
top = [n for n, _ in card['styles'][:3]]
assert 'Ukiyo-e' in top, top
c15 = np.flatnonzero(named & (common.cluster == 15))
rest = np.flatnonzero(named & (common.cluster != 15))
out['moves'] = {'time': move('before 1900 basis, after 1900 tested', earlier, later),
                'east_asian': move('rest-of-corpus basis, East Asian prints tested', rest, c15),
                'east_asian_top_styles': top}

# kept and missed per work, time: same construction as the time move, with its own draws
T = capped(later, 20, rng)
fitA = capped(earlier, 20, rng)
mA, VA = code(X[fitA])
B = capped(np.setdiff1d(np.flatnonzero(named), T), 20, rng)[:len(fitA)]
assert len(B) == len(fitA)
mB, VB = code(X[B])
t_idx = rng.choice(len(T), 1000, replace=False)


def parts(m, V):
    c = X[T] - m
    total = (c ** 2).sum(1)
    kept = ((c @ V[:K].T) ** 2).sum(1)
    return kept, total - kept, total


k = {}
for who, (m, V) in (('withheld', (mA, VA)), ('random', (mB, VB))):
    kp, ms, tt = parts(m, V)
    k[who] = {'sample_kept_sq': rnd(kp[t_idx]), 'sample_missed_sq': rnd(ms[t_idx]), 'sample_total_sq': rnd(tt[t_idx]),
              'sum_kept_sq': float(kp.sum()), 'sum_missed_sq': float(ms.sum()), 'sum_total_sq': float(tt.sum()),
              'pooled_kept_share': float(kp.sum() / tt.sum())}
out['kept'] = {'n_test': int(len(T)), 'n_sample': 1000, 'n_basis': int(len(fitA)), 'k': K, 'sample_decade': decade(T[t_idx]),
               'withheld': k['withheld'], 'random': k['random'],
               'gap_points': 100 * (k['withheld']['pooled_kept_share'] - k['random']['pooled_kept_share']),
               'note': 'withheld = pre-1900 basis, never saw the test set; random = same size, from the corpus minus the test set, cap 20; '
                       'each measured from its own basis mean; gap = difference of ratios of sums over the whole test set'}

# leftover share for every corpus work under the corpus basis
c = X[corpus] - m0
share = 1 - ((c @ V0[:K].T) ** 2).sum(1) / (c ** 2).sum(1)
hist, edges = np.histogram(share, bins=100, range=(0, 1))
dec = year[corpus] // 10 * 10
qs = {int(d): np.percentile(share[dec == d], [5, 25, 50, 75, 95]).round(4).tolist() for d in np.unique(dec) if (dec == d).sum() >= 50}
out['leftover'] = {'n': int(len(corpus)), 'hist_counts': hist.tolist(), 'bin_edges': edges.round(4).tolist(),
                   'median': float(np.median(share)), 'top1pct_cut': float(np.percentile(share, 99)), 'decade_quantiles_5_25_50_75_95': qs}

for part, dl in (('time', out['moves']['time']['test_decade']), ('time basis', out['moves']['time']['basis_group_decade']),
                 ('kept', out['kept']['sample_decade']), ('sphere', out['sphere']['sample_decade'])):
    assert None not in dl, f'undated work in {part}'
os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/figdata.json', 'w') as f:
    json.dump(out, f)
shutil.rmtree(TMP, ignore_errors=True)
print('done; leftover median %.3f; kept gap %.2f points' % (out['leftover']['median'], out['kept']['gap_points']),
      '| time pooled', out['moves']['time']['pooled_kept'], '| east asian pooled', out['moves']['east_asian']['pooled_kept'], flush=True)
