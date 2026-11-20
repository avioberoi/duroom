"""Writes a render's inputs: kat_pool.npz (first 8 works of push/shard1.npz), expected_kat.txt, expected_main.txt, and noise_req.npz for MODE noise.
Usage: python -B t2_prep.py OUT MODE   (MODE prior or noise; default env)"""
import os
import re
import sys

import numpy as np

O = os.environ['DUROOM_OUT'] + '/sphere'
OUT, MODE = sys.argv[1], sys.argv[2]
pid = lambda paths: np.array([int(re.search(r'push_(\d+)\.png$', str(p)).group(1)) for p in paths])  # noqa: E731
pool = np.load(f'{O}/pools/pre1900_cap3.npz', allow_pickle=True)
pidx = {int(i): k for k, i in enumerate(pool['ids'].astype(int))}
P = np.load(f'{O}/push_emb.npz', allow_pickle=True)
Pid = pid(P['paths'])
prow = {int(i): k for k, i in enumerate(Pid)}


def new(f):
    assert not os.path.exists(f), f'{f} exists'
    return f


def subpool(ids, f):
    """Pool rows of these ids in pool order (the order pushforward_sphere.py renders in)."""
    k = np.sort(np.array([pidx[int(i)] for i in ids]))
    got = pool['ids'][k].astype(int)
    assert sorted(got.tolist()) == sorted(int(i) for i in ids)
    assert (pool['vec'][k].astype('float32') == P['vec'][[prow[int(i)] for i in got]].astype('float32')).all(), 'pool vec differs from push_emb vec'
    np.savez(new(f), rows=pool['rows'][k], ids=pool['ids'][k], artist=np.asarray([str(x) for x in pool['artist'][k]], dtype=str), vec=pool['vec'][k])
    return got


kid = subpool(np.load(f'{O}/push/shard1.npz', allow_pickle=True)['ids'].astype(int)[:8], f'{OUT}/inputs/kat_pool.npz')
open(new(f'{OUT}/inputs/expected_kat.txt'), 'w').write(''.join(f'{OUT}/kat/img/push_{i}.png\n' for i in kid))
print('known-answer works (first 8 of O/push/shard1.npz), pool order:', kid.tolist())
if MODE == 'prior':
    main = [f'{OUT}/prior/cone_cone_{i:05d}_{60000 + i}.png' for i in range(192)] + [f'{OUT}/prior/sphere_sphere_{i:05d}_{61000 + i}.png' for i in range(64)]
elif MODE == 'noise':
    ne = np.load(f'{O}/eval/bound/noise_check/noise_emb.npz', allow_pickle=True)
    fam = np.array([str(p).rsplit('/', 1)[1][0] for p in ne['paths']])
    assert len(fam) == 160 and fam.tolist() == sorted(fam.tolist()) and all((fam == c).sum() == 32 for c in 'abcde')
    v = np.asarray(ne['emb_clip-vit-l-14'], np.float32)
    seed = (70000 + np.arange(len(fam))).astype(np.int64)
    np.savez(new(f'{OUT}/inputs/noise_req.npz'), v=v, family=fam, render_seed=seed, src=np.array([str(p) for p in ne['paths']]))
    main = [f'{OUT}/noise/probe_{fam[i]}_{i:05d}_{seed[i]}.png' for i in range(len(fam))]
    print('noise requests: 160 rows (a-e, 32 each), seeds 70000-70159, |v| min/max', float(np.linalg.norm(v, axis=1).min()), float(np.linalg.norm(v, axis=1).max()))
else:
    raise SystemExit(f'unknown mode {MODE}')
open(new(f'{OUT}/inputs/expected_main.txt'), 'w').write(''.join(p + '\n' for p in main))
print(f'expected renders: {len(kid)} known-answer + {len(main)} main')
