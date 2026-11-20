"""Does the direction the record moves in turn at a cutoff, or drift? Per cutoff, the displacement between equal past and next windows and the next window's leftover direction, and how they line up with the previous cutoff's, against a no-time placebo.
Usage: turn.py OUT_DIR [W=40] [STEP=10] [SEEDS=20]"""
import json
import os
import sys

import numpy as np

from basis import capped, code
from data import CLIP, labels, lens, year

OUT = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 40
STEP = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 10
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 20
assert SEEDS >= 2
KDIR, CAP, SMAX, SMIN, YEAR_MAX = 32, 20, 4000, 800, 2026
CUTS = list(range(1800, 2001, STEP))
os.makedirs(OUT, exist_ok=True)
X = lens(CLIP).astype('float64')
X /= np.linalg.norm(X, axis=1, keepdims=True)
artist = labels['artist']
usable = (artist != '') & (year >= 1000) & (year <= YEAR_MAX)
unit = lambda a: a / np.linalg.norm(a)    # noqa: E731
q = lambda v: [float(np.percentile(v, p)) for p in (25, 50, 75)]    # noqa: E731


def halves(rows, rng):
    names = rng.permutation(np.unique(artist[rows]))
    a = np.isin(artist[rows], names[:len(names) // 2])
    return rows[a], rows[~a]


def stats(basis_rows, next_rows, rng):
    """move_dir between the two sides; shift from next_rows' fit half under basis_rows' basis; share and move on the other half."""
    mean, V = code(X[basis_rows])
    V = V[:KDIR]
    move_dir = unit(X[next_rows].mean(0) - mean)
    left = lambda rows: (X[rows] - mean) - ((X[rows] - mean) @ V.T) @ V    # noqa: E731
    fit, test = halves(next_rows, rng)
    u = unit(left(fit).mean(0))
    Rt = left(test)
    share = float(((Rt @ u) ** 2).sum() / (Rt ** 2).sum())
    move = float(((X[test] - mean) @ u).mean() / ((X[basis_rows] - mean) @ u).std())
    return move_dir, u, share, move


res = {'W': W, 'step': STEP, 'seeds': SEEDS, 'kdir': KDIR, 'cap': CAP, 'smax': SMAX, 'smin': SMIN, 'cuts': {}}
MD, U, MDpl, Upl = {}, {}, {}, {}    # cut -> array (seeds, 768)
for c in CUTS:
    past_all = np.flatnonzero(usable & (year >= c - W) & (year < c))
    next_all = np.flatnonzero(usable & (year >= c) & (year < c + W))
    row = {'n_past_window': int(len(past_all)), 'n_next_window': int(len(next_all)), 'next_window_years': int(min(W, YEAR_MAX + 1 - c)),
           'S': None, 'skipped': False, 'share': [], 'move': [], 'share_pl': [], 'move_pl': [], 'artists_past': [], 'artists_next': [], 'straddlers_in_next_share': []}
    mds, us, mdpls, upls = [], [], [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(100000 * c + s)
        P, N = capped(past_all, CAP, rng), capped(next_all, CAP, rng)    # shuffled, so the truncation below is a random draw
        S = min(len(P), len(N), SMAX)
        if S < SMIN:
            row['skipped'] = True
            break
        P, N = P[:S], N[:S]
        row['S'] = int(S)
        md, u, share, move = stats(P, N, rng)
        mds.append(md)
        us.append(u)
        row['share'].append(share)
        row['move'].append(move)
        row['artists_past'].append(int(len(np.unique(artist[P]))))
        row['artists_next'].append(int(len(np.unique(artist[N]))))
        row['straddlers_in_next_share'].append(float(np.isin(artist[N], artist[P]).mean()))
        pool = rng.permutation(np.concatenate([P, N]))    # placebo: the same 2S works, shuffled, two sides by artist, no time
        A, B = halves(pool, rng)
        m = min(len(A), len(B))
        mdpl, upl, share_pl, move_pl = stats(A[:m], B[:m], rng)
        mdpls.append(mdpl)
        upls.append(upl)
        row['share_pl'].append(share_pl)
        row['move_pl'].append(move_pl)
    if not row['skipped']:
        MD[c], U[c], MDpl[c], Upl[c] = (np.stack(v) for v in (mds, us, mdpls, upls))
        row['share_q'], row['move_q'], row['share_pl_q'], row['move_pl_q'] = q(row['share']), q(row['move']), q(row['share_pl']), q(row['move_pl'])
        iu = np.triu_indices(SEEDS, 1)
        row['within_cos_move_q'], row['within_cos_shift_q'] = q(np.abs(MD[c] @ MD[c].T)[iu]), q(np.abs(U[c] @ U[c].T)[iu])
        row['within_cos_move_pl_q'], row['within_cos_shift_pl_q'] = q(np.abs(MDpl[c] @ MDpl[c].T)[iu]), q(np.abs(Upl[c] @ Upl[c].T)[iu])
        print(c, 'S', row['S'], f"share {row['share_q'][1]:.3f} (pl {row['share_pl_q'][1]:.3f}) move {row['move_q'][1]:.2f} (pl {row['move_pl_q'][1]:.2f}) "
              f"within|cos| move {row['within_cos_move_q'][1]:.2f} (pl {row['within_cos_move_pl_q'][1]:.2f}) shift {row['within_cos_shift_q'][1]:.2f} (pl {row['within_cos_shift_pl_q'][1]:.2f})", flush=True)
    else:
        print(c, 'skipped, S', int(min(len(capped(past_all, CAP, np.random.default_rng(0))), len(capped(next_all, CAP, np.random.default_rng(0))))), flush=True)
    res['cuts'][c] = row

cuts = [c for c in CUTS if c in U]
adj = lambda A, B: q(np.abs((A * B).sum(1)))    # noqa: E731  # row-wise cosines of unit vectors, index-paired seeds
for c in cuts:
    prev = c - STEP
    if prev in U:
        r = res['cuts'][c]
        r['adjacent_cos_move_q'], r['adjacent_cos_move_pl_q'] = adj(MD[c], MD[prev]), adj(MDpl[c], MDpl[prev])
        r['adjacent_cos_shift_q'], r['adjacent_cos_shift_pl_q'] = adj(U[c], U[prev]), adj(Upl[c], Upl[prev])
for name, D in (('move', MD), ('shift', U)):
    M = np.array([[float(np.median(np.abs((D[a] * D[b]).sum(1)))) for b in cuts] for a in cuts])
    res[f'matrix_abs_cos_{name}'] = M.round(4).tolist()
    if 1900 in U:
        res[f'cos_with_1900_{name}'] = {c: float(M[cuts.index(c), cuts.index(1900)]) for c in cuts}
res['matrix_cuts'] = cuts
with open(f'{OUT}/turn.json', 'w') as fh:
    json.dump(res, fh, indent=1)
np.savez_compressed(f'{OUT}/directions.npz', cuts=np.array(cuts), move=np.stack([MD[c] for c in cuts]), shift=np.stack([U[c] for c in cuts]),
                    move_pl=np.stack([MDpl[c] for c in cuts]), shift_pl=np.stack([Upl[c] for c in cuts]))
print('adjacent |cos| move_dir (time / placebo):', ' '.join(f"{c}:{res['cuts'][c]['adjacent_cos_move_q'][1]:.2f}/{res['cuts'][c]['adjacent_cos_move_pl_q'][1]:.2f}" for c in cuts if 'adjacent_cos_move_q' in res['cuts'][c]))
print('adjacent |cos| shift    (time / placebo):', ' '.join(f"{c}:{res['cuts'][c]['adjacent_cos_shift_q'][1]:.2f}/{res['cuts'][c]['adjacent_cos_shift_pl_q'][1]:.2f}" for c in cuts if 'adjacent_cos_shift_q' in res['cuts'][c]))
print('done', flush=True)
