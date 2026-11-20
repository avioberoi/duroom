"""Walks from each search start toward its nearest targets, renders each step with the search's decoder, and scores fidelity and the DINOv3/SigLIP 15-NN chords against P's bar.
Usage: python oracle.py INPUTS_NPZ PUSH_EMB_NPZ OUT_DIR [--angles 10,20,...] [--seed 0]   (render env)"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import utils  # noqa: E402
from generators.variations import generate  # noqa: E402  (the search's decoder path: unit vector scaled to VARIATIONS_NORM, guidance from env)

ap = argparse.ArgumentParser()
ap.add_argument('inputs'); ap.add_argument('push_emb'); ap.add_argument('out')
ap.add_argument('--angles', default='10,20,30,40,50,60,70')
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
os.makedirs(f'{a.out}/frames', exist_ok=True)
unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
chord = lambda c: float(np.sqrt(max(2.0 - 2.0 * float(c), 0.0)))  # noqa: E731

inp = np.load(a.inputs, allow_pickle=True)
starts, start_tags = np.asarray(inp['starts'], np.float64), [str(t) for t in inp['start_tags']]
targets, target_names = np.asarray(inp['targets'], np.float64), [str(t) for t in inp['target_names']]
nearest = np.asarray(inp['nearest'])                       # (n_starts, 3) indices into targets
P = np.load(a.push_emb, allow_pickle=True)
obs = {n: np.asarray(P[f'emb_{n}'], np.float64) for n in ('dinov3-l', 'siglip-so400m')}
for n in obs:
    obs[n] /= np.linalg.norm(obs[n], axis=1, keepdims=True)
bar = {}
for n, E in obs.items():   # P's leave-one-out q99 with same-artist rows excluded, as search_v2
    Dm = np.sqrt(np.clip(2 - 2 * E @ E.T, 0, None)); np.fill_diagonal(Dm, np.inf)
    Dm[P['artist'][:, None] == P['artist'][None, :]] = np.inf
    bar[n] = float(np.percentile(np.sort(Dm, axis=1)[:, :15].mean(1), 99))
print('bars', bar, flush=True)

frames = []
angles = [float(t) for t in a.angles.split(',')]
for si, (x, tag) in enumerate(zip(starts, start_tags)):
    x = unit(x)
    frames.append(dict(start=tag, target=-1, target_name='', angle=0.0, v=x, kind='start'))
    for tj in nearest[si]:
        t = unit(targets[int(tj)]); th = float(np.degrees(np.arccos(np.clip(t @ x, -1, 1))))
        d = unit(t - (t @ x) * x)
        for ang in sorted(set(angles) | {round(th, 1)}):
            r = np.radians(ang)
            v = unit(np.cos(r) * x + np.sin(r) * d)
            frames.append(dict(start=tag, target=int(tj), target_name=target_names[int(tj)], angle=float(ang), v=v, kind='target' if abs(ang - th) < 1e-6 else 'walk', target_angle=th))
print(len(frames), 'frames', flush=True)

import torch  # noqa: E402
for i, f in enumerate(frames):
    f['path'] = f'{a.out}/frames/{i:04d}.png'
    if not os.path.exists(f['path']):
        torch.manual_seed(a.seed)
        generate(None, f['v'], f['path'])
print('rendered', flush=True)

enc = {n: utils.load_encoder(n)[0] for n in ('clip-vit-l-14', 'dinov3-l', 'siglip-so400m')}
rows = []
for f in frames:
    c = unit(np.asarray(enc['clip-vit-l-14'](f['path']), np.float64))
    rec = {k: f[k] for k in ('start', 'target', 'target_name', 'angle', 'kind', 'path')}
    rec['target_angle'] = f.get('target_angle', 0.0)
    rec['fidelity'] = chord(c @ f['v'])
    rec['angle_achieved'] = float(np.degrees(np.arccos(np.clip(c @ unit(starts[start_tags.index(f['start'])]), -1, 1))))
    ok = True
    for n, E in obs.items():
        e = unit(np.asarray(enc[n](f['path']), np.float64))
        s = float(np.sort(np.sqrt(np.clip(2 - 2 * E @ e, 0, None)))[:15].mean())
        rec[f'knn15_{n}'] = s; ok &= s > bar[n]
    rec['passes_both'] = bool(ok)
    rows.append(rec)
    print(f"{rec['start']} -> {rec['target_name'] or 'start'} at {rec['angle']:.0f} deg ({rec['kind']}): fidelity {rec['fidelity']:.3f}, dino {rec['knn15_dinov3-l']:.3f}, siglip {rec['knn15_siglip-so400m']:.3f}, passes {ok}", flush=True)
json.dump({'bars': bar, 'angles': angles, 'frames': rows}, open(f'{a.out}/oracle.json', 'w'), indent=1)

from PIL import Image, ImageDraw  # noqa: E402
S = 192
for tag in start_tags:   # one contact image per start: rows = targets, columns = angles (start first)
    sel = [r for r in rows if r['start'] == tag]
    tids = sorted({r['target'] for r in sel if r['target'] >= 0})
    cols = sorted({r['angle'] for r in sel})
    contact = Image.new('RGB', (len(cols) * S, len(tids) * (S + 24)), 'white'); dr = ImageDraw.Draw(contact)
    for ri, tj in enumerate(tids):
        for ci, ang in enumerate(cols):
            r = next((r for r in sel if r['target'] == tj and r['angle'] == ang), None) or (next((r for r in sel if r['kind'] == 'start'), None) if ang == 0 else None)
            if r is None: continue
            contact.paste(Image.open(r['path']).convert('RGB').resize((S, S)), (ci * S, ri * (S + 24)))
            dr.text((ci * S + 3, ri * (S + 24) + S + 4), f"{ang:.0f} f{r['fidelity']:.2f} d{r['knn15_dinov3-l']:.2f}{' PASS' if r['passes_both'] else ''}", fill='black')
    contact.save(f'{a.out}/contact_{tag}.png')
print('done', flush=True)
