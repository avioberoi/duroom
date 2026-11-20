"""Time gap at every cutoff in equal windows, against a no-time placebo and a second placebo for the false-positive rate, both directions; peak and step tests and a verdict.
Usage: cutoff_sweep.py OUT_DIR [W=40] [SEEDS=20] [FIRST=1800] [LAST=2000] [STEP=10] [S_MAX=4000] [MIN_S=2000]"""
import json
import os
import sys
import zlib

OUT = sys.argv[1]
W, SEEDS, FIRST, LAST, STEP, S_MAX, MIN_S = (int(sys.argv[i]) if len(sys.argv) > i else d
                                             for i, d in ((2, 40), (3, 20), (4, 1800), (5, 2000), (6, 10), (7, 4000), (8, 2000)))
sys.argv = ['common.py', f'{OUT}/import', '1', 'artist']    # common reads its arguments at import; only X, labels and helpers are used
import numpy as np    # noqa: E402
from multiprocessing import get_context    # noqa: E402

import common    # noqa: E402
from basis import KS, capped, code    # noqa: E402

X, artist, year, named, dated = common.X, common.artist, common.year, common.named, common.dated
K32 = KS.index(32)
KINDS = ('time', 'placebo', 'placebo2')
window = lambda lo, hi: np.flatnonzero(dated & named & (year >= lo) & (year < hi))    # noqa: E731
cap_size = lambda rows: int(np.minimum(np.unique(artist[rows], return_counts=True)[1], 20).sum())    # noqa: E731  (capped size, no RNG)


def split(c):
    """PAST and NEXT windows at c, artist-disjoint: an artist on both sides stays on the side with more of their works (ties to PAST)."""
    past, nxt = window(c - W, c), window(c, c + W)
    cp = dict(zip(*np.unique(artist[past], return_counts=True)))
    cn = dict(zip(*np.unique(artist[nxt], return_counts=True)))
    both = set(cp) & set(cn)
    to_past = [a for a in both if cp[a] >= cn[a]]
    to_next = [a for a in both if cp[a] < cn[a]]
    return past[~np.isin(artist[past], to_next)], nxt[~np.isin(artist[nxt], to_past)], len(both)


# sizes: the same S at every cutoff run; cutoffs whose thinner window holds fewer than MIN_S capped works are left out
all_sides = {c: split(c) for c in range(FIRST, LAST + 1, STEP)}
print('per cutoff (raw past, raw next, capped past, capped next, straddling artists):',
      {c: (len(p), len(n), cap_size(p), cap_size(n), k) for c, (p, n, k) in all_sides.items()}, flush=True)
ok = [c for c, (p, n, k) in all_sides.items() if min(cap_size(p), cap_size(n)) >= MIN_S]
runs, cur = [], []    # neighbours must be STEP apart, so keep the longest contiguous run of cutoffs that pass the floor
for c in ok:
    cur = cur + [c] if cur and c - cur[-1] == STEP else [c]
    runs.append(cur)
cuts = max(runs, key=len) if runs else []
assert len(cuts) >= 5 and 1900 in cuts, f'cutoffs run {cuts}: need 5 or more, contiguous, including 1900'
straddle = {c: {'straddling_artists': all_sides[c][2],
                'share_dropped': 1 - (len(all_sides[c][0]) + len(all_sides[c][1])) / (len(window(c - W, c)) + len(window(c, c + W)))} for c in cuts}
sides = {c: all_sides[c][:2] for c in cuts}
smallest = min(min(cap_size(p), cap_size(n)) for p, n in sides.values())
S = min(smallest, S_MAX)
M, N = S // 2, (3 * S) // 4    # T = M works; A, B, B2 = N works each; B and B2 come from 2S - M >= 2N works
print('cutoffs run:', cuts, f'S {S} (smallest capped side {smallest}), T {M}, bases {N}', flush=True)


def run(job):
    """One cell: cutoff, direction ('after', 'before'), kind ('time', 'placebo', 'placebo2'), seed."""
    c, direction, kind, seed = job
    rng = np.random.default_rng([seed, zlib.crc32(f'{c}{direction}{kind}{W}'.encode())])
    past, nxt = (capped(r, 20, rng)[:S] for r in sides[c])
    assert len(past) == len(nxt) == S
    if kind != 'time':    # same 2S works, sides by artist at random: no time
        both = np.r_[past, nxt]
        names = rng.permutation(np.unique(artist[both]))
        order = np.concatenate([rng.permutation(both[artist[both] == a]) for a in names])
        past, nxt = order[:S], order[S:]
    a_side, t_side = (past, nxt) if direction == 'after' else (nxt, past)
    t_side = rng.permutation(t_side)
    T = t_side[:M]
    A = rng.permutation(a_side)[:N]
    pool = rng.permutation(np.r_[a_side, t_side[M:]])
    B, B2 = pool[:N], pool[N:2 * N]
    (mA, VA), (mB, VB), (mB2, VB2) = code(X[A]), code(X[B]), code(X[B2])
    mT = X[T].mean(0)
    k = lambda m, V: float(common.kept(X[T], m, V)[K32])    # noqa: E731
    return {'cut': c, 'direction': direction, 'kind': kind, 'seed': seed,
            'unseen': k(mA, VA), 'random': k(mB, VB), 'random2': k(mB2, VB2),
            'unseen_own_mean': k(mT, VA), 'random_own_mean': k(mT, VB),
            'share_B_from_T_side': float(np.isin(B, t_side).mean()),
            'share_B_by_T_artists': float(np.isin(artist[B], artist[T]).mean()),
            'shared_artist_share_T': float(np.isin(artist[T], artist[a_side]).mean()),
            'artists_T_A_B': [len(np.unique(artist[Y])) for Y in (T, A, B)]}


if __name__ == '__main__':
    jobs = [(c, d, kind, s) for c in cuts for d in ('after', 'before') for kind in KINDS for s in range(SEEDS)]
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    print(len(jobs), 'cells on', workers, 'processes', flush=True)
    with get_context('fork').Pool(workers) as p:
        rows = list(p.imap(run, jobs, chunksize=4))
    os.makedirs(OUT, exist_ok=True)
    json.dump({'W': W, 'seeds': SEEDS, 'S': S, 'M': M, 'N': N, 'cuts': cuts, 'rows': rows}, open(f'{OUT}/rows.json', 'w'))    # kept if the summary fails

    sel = lambda c, d, kind: [r for r in rows if (r['cut'], r['direction'], r['kind']) == (c, d, kind)]    # noqa: E731
    gap = lambda c, d, kind, a='unseen', b='random': np.array([100 * (r[a] - r[b]) for r in sel(c, d, kind)])    # noqa: E731
    summary = {}
    for c in cuts:
        for d in ('after', 'before'):
            g = {kind: gap(c, d, kind) for kind in KINDS}
            own = {kind: gap(c, d, kind, 'unseen_own_mean', 'random_own_mean') for kind in ('time', 'placebo')}
            v = {'gap_time': float(g['time'].mean()), 'gap_placebo': float(g['placebo'].mean()), 'gap_placebo2': float(g['placebo2'].mean()),
                 'excess_depth': float(g['placebo'].mean() - g['time'].mean()),
                 'excess_se': float(np.sqrt((g['time'].var(ddof=1) + g['placebo'].var(ddof=1)) / SEEDS)),
                 'calib_excess': float(g['placebo'].mean() - g['placebo2'].mean()),
                 'calib_se': float(np.sqrt((g['placebo2'].var(ddof=1) + g['placebo'].var(ddof=1)) / SEEDS)),
                 'own_mean_gap_time': float(own['time'].mean()), 'own_mean_gap_placebo': float(own['placebo'].mean()),
                 'artists_T_A_B_time': np.mean([r['artists_T_A_B'] for r in sel(c, d, 'time')], axis=0).tolist()}
            for key in ('shared_artist_share_T', 'share_B_by_T_artists', 'share_B_from_T_side'):
                for kind in ('time', 'placebo'):
                    v[f'{key}_{kind}'] = float(np.mean([r[key] for r in sel(c, d, kind)]))
            summary[f'{d}_{c}'] = v

    def detect(value, se):
        out = {}
        for d in ('after', 'before'):
            e = [(summary[f'{d}_{c}'][value], summary[f'{d}_{c}'][se]) for c in cuts]
            beats = lambda i, j: e[i][0] - e[j][0] > 2 * np.hypot(e[i][1], e[j][1])    # noqa: E731
            # step at c: the two cutoffs from c on sit above the two before c (a level change that stays, not one rise)
            lvl = lambda a, b: (np.mean([e[t][0] for t in (a, b)]), np.hypot(e[a][1], e[b][1]) / 2)    # noqa: E731
            step = lambda i: lvl(i, i + 1)[0] - lvl(i - 2, i - 1)[0] > 2 * np.hypot(lvl(i, i + 1)[1], lvl(i - 2, i - 1)[1])    # noqa: E731
            out[d] = {'peaks': [cuts[i] for i in range(1, len(cuts) - 1) if beats(i, i - 1) and beats(i, i + 1)],
                      'steps': [cuts[i] for i in range(2, len(cuts) - 1) if step(i)]}
        return out

    found, false_pos = detect('excess_depth', 'excess_se'), detect('calib_excess', 'calib_se')
    n_tests = {'peaks': 2 * (len(cuts) - 2), 'steps': 2 * (len(cuts) - 3)}
    false_count = {kind: sum(len(false_pos[d][kind]) for d in false_pos) for kind in ('peaks', 'steps')}
    near = {d: any(abs(c - 1900) <= STEP for c in found[d]['peaks'] + found[d]['steps']) for d in found}
    verdict = ('1900 in both directions' if all(near.values())
               else 'no detectable break' if not any(found[d]['peaks'] or found[d]['steps'] for d in found)
               else 'breaks: ' + json.dumps(found))
    json.dump({'W': W, 'seeds': SEEDS, 'S': S, 'M': M, 'N': N, 'cuts': cuts,
               'sign': 'gap < 0: the basis that never saw T keeps less of T; excess = depth(time) - depth(placebo)',
               'summary': summary, 'straddle': straddle, 'found': found, 'near_1900': near,
               'false_positives_placebo2': false_pos, 'false_positive_count': false_count, 'n_tests': n_tests, 'verdict': verdict},
              open(f'{OUT}/cutoff_sweep.json', 'w'), indent=1)
    print('straddle', straddle, flush=True)
    print('false positives (placebo2):', false_count, 'of', n_tests, flush=True)
    for key, v in summary.items():
        print(key, 'gap time %.2f placebo %.2f excess %.2f se %.2f own-mean time %.2f shared-artist T %.2f' % (
            v['gap_time'], v['gap_placebo'], v['excess_depth'], v['excess_se'], v['own_mean_gap_time'], v['shared_artist_share_T_time']), flush=True)
    print('found', found, 'false positives (placebo2)', false_pos, 'verdict', verdict, flush=True)
