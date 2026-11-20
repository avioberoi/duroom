"""Renders pool works' unit CLIP vectors as the search renders candidates (variations backend, torch.manual_seed(artwork id)) and writes shard npz with rows, ids, artist, vec, paths.
Usage: python pushforward_sphere.py --pool pre1900_cap3.npz --out DIR --n 2048 [--shard I --nshards N] [--rng 11] [--exclude P_DIR]"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

CODE = os.environ.get('SPHERE_CODE', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import utils  # noqa: E402
from generators import generate, set_default_backend  # noqa: E402

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pool', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=2048)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--rng', type=int, default=11)
    p.add_argument('--exclude', default=None)
    a = p.parse_args()
    os.makedirs(f'{a.out}/img', exist_ok=True)
    cfg = utils.load_config(os.path.join(CODE, 'config.yaml'))
    os.environ.setdefault('VARIATIONS_NORM', str(cfg['generator']['norm']))
    os.environ.setdefault('VARIATIONS_GUIDANCE', str(cfg['generator']['guidance']))
    set_default_backend(cfg['generator']['backend'])
    pool = np.load(a.pool, allow_pickle=True)
    rows, ids, artist, vec = pool['rows'], pool['ids'].astype(int), pool['artist'], pool['vec'].astype('float32')
    cand = np.arange(len(rows))
    if a.exclude:
        used = np.concatenate([np.load(f, allow_pickle=True)['rows'] for f in sorted(glob.glob(f'{a.exclude}/shard[0-9].npz'))])
        cand = cand[~np.isin(rows, used)]
    if a.n > len(cand):
        raise SystemExit(f'pool has {len(cand)} rows left, {a.n} asked')
    pick = np.sort(np.random.default_rng(a.rng).choice(cand, a.n, replace=False))    # the same list in every shard
    mine = pick[a.shard::a.nshards]
    content = utils.get_reference_image()
    t0, done = time.time(), []
    for j in mine:
        path = f'{a.out}/img/push_{ids[j]}.png'
        if not os.path.exists(path):
            torch.manual_seed(int(ids[j]))
            generate(content, vec[j], path)
        done.append(int(j))
    done = np.array(done, dtype=int)
    np.savez(f'{a.out}/shard{a.shard}.npz', rows=rows[done], ids=ids[done], artist=np.asarray([str(x) for x in artist[done]], dtype=str),
             vec=vec[done], paths=np.array([f'{a.out}/img/push_{ids[j]}.png' for j in done], dtype=str))    # plain string arrays: no pickle across numpy versions
    secs = time.time() - t0
    print(json.dumps({'shard': a.shard, 'n': int(len(done)), 'artists': int(len(np.unique(artist[done]))), 'secs': round(secs, 1),
                      'per_render': round(secs / max(1, len(done)), 2)}), flush=True)
