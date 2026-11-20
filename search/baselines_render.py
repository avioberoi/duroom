"""Renders baseline pictures (MODE prior, chains or vectors) and writes PNGs, manifest.npz and list.txt for embed_images.py.
Usage: python baselines_render.py MODE OUT_DIR [--rng R] [--vectors NPZ] [--rows TXT] [--seed-offset N] [--shard I --nshards N]   (search env)"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generators.variations import generate  # noqa: E402  (the search's decoder path)

O = os.environ['DUROOM_OUT'] + '/sphere'
ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['prior', 'chains', 'vectors'])
ap.add_argument('out')
ap.add_argument('--rng', type=int, default=17)
ap.add_argument('--vectors', default=None)
ap.add_argument('--rows', default=None)
ap.add_argument('--seed-offset', type=int, default=0)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
import torch  # noqa: E402

unit = lambda v: np.asarray(v, np.float64) / np.linalg.norm(np.asarray(v, np.float64))  # noqa: E731
rng = np.random.default_rng(a.rng)
items = []  # kind, tag, row, seed, step, v


def render(v, seed, path):
    if not os.path.exists(path):
        torch.manual_seed(seed)
        generate(None, v, path)


if a.mode == 'prior':
    P = np.load(f'{O}/push_emb.npz', allow_pickle=True)
    mu = unit(np.load(f'{O}/pools/starts_pre1900.npz', allow_pickle=True)['mean_pre'])
    V = np.asarray(P['vec'], np.float64); V /= np.linalg.norm(V, axis=1, keepdims=True)
    ang = float(np.median(np.degrees(np.arccos(np.clip(V @ mu, -1, 1)))))
    sigma = np.tan(np.radians(ang)) / np.sqrt(V.shape[1])   # unit(mu + sigma g) sits at about that angle from mu
    print(f'prior: median angle of P requests from the pre-1900 mean {ang:.1f} deg, sigma {sigma:.4f}', flush=True)
    for i in range(192):
        items.append(dict(kind='cone', tag='cone', row=i, seed=60000 + i, step=0, v=unit(mu + sigma * rng.standard_normal(V.shape[1]))))
    for i in range(64):
        items.append(dict(kind='sphere', tag='sphere', row=i, seed=61000 + i, step=0, v=unit(rng.standard_normal(V.shape[1]))))
elif a.mode == 'vectors':    # requests of an npz (probe_requests.npz); row = frame index in the npz, so shards and stages align
    z = np.load(a.vectors, allow_pickle=True)
    idx = np.loadtxt(a.rows, dtype=int, ndmin=1) if a.rows else np.arange(len(z['v']))
    for i in idx[a.shard::a.nshards]:
        items.append(dict(kind='probe', tag=str(z['family'][i]), row=int(i), seed=int(z['render_seed'][i]) + a.seed_offset, step=0, v=unit(z['v'][i])))
else:   # chains: the render of step t is read by the loop's own encoder to give the request of step t+1
    import utils  # noqa: E402  (in this env clip-vit-l-14 loads the loop's encoder, as search_v2's fidelity does)
    enc = utils.load_encoder('clip-vit-l-14')[0]
    P = np.load(f'{O}/push_emb.npz', allow_pickle=True)
    V = np.asarray(P['vec'], np.float64)
    for c, r in enumerate(np.sort(rng.choice(len(V), 48, replace=False))):
        v = unit(V[r])
        for step in range(7):
            seed = 90000 + 10 * c + step
            path = f'{a.out}/chain{c:02d}_step{step}.png'
            render(v, seed, path)
            items.append(dict(kind='chain', tag=f'chain{c:02d}', row=int(r), seed=seed, step=step, v=v, path=path))
            v = unit(enc(path))
        print(f'chain {c} done', flush=True)

if not items:   # an empty shard still writes a manifest and a list
    np.savez(f'{a.out}/manifest.npz', kind=np.array([], str), tag=np.array([], str), row=np.array([], int), seed=np.array([], int), step=np.array([], int),
             v=np.zeros((0, 768), np.float32), paths=np.array([], str))
    open(f'{a.out}/list.txt', 'w').write('')
    print('done 0 items (empty shard)', flush=True)
    sys.exit(0)
for i, it in enumerate(items):
    if 'path' not in it:
        it['path'] = f"{a.out}/{it['kind']}_{it['tag']}_{it['row']:05d}_{it['seed']}.png"
        render(it['v'], it['seed'], it['path'])
    if i % 50 == 0:
        print(f'{i + 1}/{len(items)} rendered', flush=True)
np.savez(f'{a.out}/manifest.npz', kind=np.array([i['kind'] for i in items]), tag=np.array([i['tag'] for i in items]),
         row=np.array([i['row'] for i in items]), seed=np.array([i['seed'] for i in items]), step=np.array([i['step'] for i in items]),
         v=np.stack([i['v'] for i in items]).astype(np.float32), paths=np.array([i['path'] for i in items]))
open(f'{a.out}/list.txt', 'w').write('\n'.join(i['path'] for i in items) + '\n')
print('done', len(items), 'items', flush=True)
