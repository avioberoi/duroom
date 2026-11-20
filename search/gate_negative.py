"""Scores 8 noise, 8 flat-colour and 4 text PNGs with the coherence gate and, per observer, the mean chord to the 15 nearest record works.
Usage: python gate_negative.py OUT_DIR   (GPU)"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

CODE = os.environ.get('SPHERE_CODE', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import utils  # noqa: E402

OUT = sys.argv[1]
os.makedirs(f'{OUT}/img', exist_ok=True)
rng = np.random.default_rng(0)
frames = []
for i in range(8):
    Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)).save(f'{OUT}/img/noise_{i}.png')
    frames.append(('noise', f'{OUT}/img/noise_{i}.png'))
for i in range(8):
    Image.new('RGB', (512, 512), tuple(int(v) for v in rng.integers(0, 256, 3))).save(f'{OUT}/img/flat_{i}.png')
    frames.append(('flat', f'{OUT}/img/flat_{i}.png'))
for i, text in enumerate(('NOVELTY', 'untitled 1962', 'ART', 'the record')):
    im = Image.new('RGB', (512, 512), (240, 236, 226))
    ImageDraw.Draw(im).text((40, 230), text, fill=(30, 30, 30))
    im.save(f'{OUT}/img/text_{i}.png')
    frames.append(('text', f'{OUT}/img/text_{i}.png'))
cfg = utils.load_config(os.path.join(CODE, 'config.yaml'))
neg, gate_enc, thr = utils.load_coherence_gate(cfg)
res = {'threshold_cfg': float(thr), 'items': []}
obs = {}
for name in ('dinov3-l', 'siglip-so400m'):
    E = utils.load_observer_corpus(cfg, name, normalize=True)
    obs[name] = (utils.load_encoder(name)[0], E)
for kind, path in frames:
    row = {'kind': kind, 'path': path, 'gate': float(utils.coherence_score(gate_enc(path), neg))}
    for name, (enc, E) in obs.items():
        r = enc(path).astype(np.float32)
        r /= np.linalg.norm(r)
        sims = np.sort(E @ r)[-15:]
        row[f'knn15_{name}'] = float(np.sqrt(np.clip(2 - 2 * sims, 0, None)).mean())
    res['items'].append(row)
with open(f'{OUT}/gate_negative.json', 'w') as fh:
    json.dump(res, fh, indent=1)
for kind in ('noise', 'flat', 'text'):
    sel = [r for r in res['items'] if r['kind'] == kind]
    print(kind, 'gate median %.3f min %.3f' % (np.median([r['gate'] for r in sel]), min(r['gate'] for r in sel)),
          '| knn15 to the record: dinov3 %.3f siglip %.3f' % (np.median([r['knn15_dinov3-l'] for r in sel]), np.median([r['knn15_siglip-so400m'] for r in sel])))
print('threshold', thr, '| any below threshold:', sum(r['gate'] < thr for r in res['items']), 'of', len(res['items']))
