"""Does a picture's reading move when its request moves? Inside each round of 32: gain, cosine, share carried and share of readings nearest to their own request's offset, per run and arm, with the 2,048 reference works in random groups of 32 as a scale.
Usage: python gain.py RUNS_ROOT PUSH_EMB.npz tag:arm [tag:arm ...]   (arm N = told the novelty score, R = told random scores)"""
import sys

import numpy as np

unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731


# gain = <dr, dq> / |dq|^2 (least squares dr = b dq); cosine = median cos(dr_i, dq_i); carried = share of the readings' spread the requests' spread explains; own = share of readings nearest to their own request's offset
def four(Q, Rd):
    dq, dr = Q - Q.mean(0), Rd - Rd.mean(0)
    b = float((dr * dq).sum() / (dq ** 2).sum())
    C = unit(dr) @ unit(dq).T                               # row i: cosine of reading i's offset with every request's offset
    ceil = float((Q * Rd).sum(1).mean())                     # the gain of a decoder that follows the request exactly, plus independent displacement
    return b, float(np.median(np.diag(C))), float(b * b * (dq ** 2).sum() / (dr ** 2).sum()), float((C.argmax(1) == np.arange(len(Q))).mean()), ceil


def summarise(rows):
    return tuple(float(np.median([r[j] for r in rows])) for j in range(5))


if __name__ == '__main__':
    root, push, res = sys.argv[1], sys.argv[2], {}
    fmt = lambda s: f'gain {s[0]:.2f} (ceiling for a decoder that follows the request {s[4]:.2f}; gain over ceiling {s[0] / s[4]:.2f}) | cosine {s[1]:.2f} | carried {100 * s[2]:.0f}% | own request nearest {100 * s[3]:.0f}% (chance 3%)'  # noqa: E731
    for a in sys.argv[3:]:
        tag, arm = a.split(':')
        z = np.load(f'{root}/{tag}_committee_euclidean_min_seedrow/all_candidates.npz', allow_pickle=True)
        it = z['iteration']
        q, r = z['style'].astype(np.float64), z['render_clip'].astype(np.float64)
        rows = []
        for t in range(int(it.max()) + 1):
            k = np.flatnonzero(it == t)
            assert len(k) == 32 and np.isfinite(r[k]).all() and np.abs(np.linalg.norm(r[k], axis=1) - 1).max() < 1e-4, (tag, t)
            rows.append(four(unit(q[k]), unit(r[k])))
        s = summarise(rows)
        res.setdefault(arm, []).append(s)
        print(f'{tag:12s} {"novelty" if arm == "N" else "random":7s} {fmt(s)}', flush=True)
    for arm, name in (('N', 'novelty'), ('R', 'random')):
        if arm in res:
            print(f'{name}, median over {len(res[arm])} runs: {fmt(summarise(res[arm]))}')
    p = np.load(push, allow_pickle=True)
    Q, Rd = unit(p['vec'].astype(np.float64)), unit(p['emb_clip-vit-l-14'].astype(np.float64))
    rng = np.random.default_rng(0)
    rows = [four(Q[g], Rd[g]) for g in (rng.choice(len(Q), 32, replace=False) for _ in range(200))]
    print(f'ceiling at wide spread, NOT comparable in kind (reference works in 200 random groups of 32, far wider apart than a round; exact encoder): {fmt(summarise(rows))}')
