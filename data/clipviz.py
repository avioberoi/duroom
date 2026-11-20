"""CLIP ViT-L/14 (OpenAI clip package, CPU, float32) on one image: the 224 x 224 crop, patch-token norms after blocks 1, 12 and 24, the end vector and its angle to the corpus's average direction with the corpus percentiles; first re-encodes 16 corpus images as a gate against their stored rows.
Usage: clipviz.py IMAGE OUT_DIR"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

IMAGE, OUT = sys.argv[1], sys.argv[2]
assert os.path.exists(IMAGE), 'image missing'
raw = open(IMAGE, 'rb').read()
import clip    # noqa: E402  (the OpenAI package the corpus encoding used)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLIP, IMG, ids, labels, lens, year    # noqa: E402

X = lens(CLIP).astype('float64')
X /= np.linalg.norm(X, axis=1, keepdims=True)
corpus = (labels['artist'] != '') & (year >= 1000) & (year <= 2026)

m = X[corpus].mean(0)
mdir = m / np.linalg.norm(m)
corpus_angles = np.degrees(np.arccos(np.clip(X[corpus] @ mdir, -1, 1)))

torch.manual_seed(0)
model, preprocess = clip.load('ViT-L/14', device='cpu')
model.eval()


def encode(img):
    with torch.no_grad():
        return model.encode_image(preprocess(img).unsqueeze(0)).float()[0]


# gate: the model reproduces the stored corpus vectors (only cosines are kept; no ids reach logs or output)
rng = np.random.default_rng(0)
cos = []
for k, r in enumerate(rng.choice(len(ids), 16, replace=False)):
    try:
        e = encode(Image.open(f'{IMG}/{ids[r]}.jpg').convert('RGB')).numpy()
    except Exception as err:    # noqa: BLE001
        sys.exit(f'gate image {k + 1} of 16 failed: {type(err).__name__}')
    cos.append(float(e @ X[r] / np.linalg.norm(e)))
gate = {'n': 16, 'cosines': [round(c, 6) for c in cos], 'median': float(np.median(cos)), 'min': float(np.min(cos))}
print('gate', gate, flush=True)
assert gate['median'] >= 0.999 and gate['min'] >= 0.99, 'gate failed'

img = Image.open(IMAGE).convert('RGB')
x = preprocess(img).unsqueeze(0)
blocks = model.visual.transformer.resblocks
assert len(blocks) == 24
grabbed = {}
hooks = [blocks[i].register_forward_hook(lambda mod, inp, out, i=i: grabbed.__setitem__(i, out.detach())) for i in (0, 11, 23)]
with torch.no_grad():
    end = model.encode_image(x).float()[0]
for h in hooks:
    h.remove()
assert end.shape == (768,)
maps = {}
for i, t in grabbed.items():
    t = t[:, 0].float() if t.shape[1] == 1 else t[0].float()    # sequence-first (tokens, 1, width) or batch-first (1, tokens, width)
    assert t.shape == (257, 1024), t.shape
    norms = t.norm(dim=-1).numpy()
    maps[f'block_{i + 1}'] = {'class_token_norm': float(norms[0]), 'patch_norms_16x16': np.round(norms[1:].reshape(16, 16), 4).tolist()}

norm_t = preprocess.transforms[-1]
mean, std = torch.tensor(norm_t.mean).view(3, 1, 1), torch.tensor(norm_t.std).view(3, 1, 1)
crop = ((x[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
os.makedirs(OUT, exist_ok=True)
Image.fromarray(crop).save(f'{OUT}/great_wave_crop_224.png')

unit = (end / end.norm()).numpy()
angle = float(np.degrees(np.arccos(np.clip(unit @ mdir, -1, 1))))
out = {'image_sha256': hashlib.sha256(raw).hexdigest(), 'image_size': list(img.size), 'gate': gate,
       'model': "OpenAI clip package, clip.load('ViT-L/14'), the corpus encoder; CPU, float32",
       'preprocess': 'clip default: resize 224 (bicubic), centre crop 224 (drops the sides of a landscape image), CLIP mean/std',
       'blocks': maps, 'end_vector_length': float(end.norm()), 'end_vector_unit': np.round(unit, 6).tolist(),
       'angle_to_corpus_mean_direction_deg': angle,
       'corpus_angle_percentiles_1_5_50_95_99': np.round(np.percentile(corpus_angles, [1, 5, 50, 95, 99]), 3).tolist(),
       'angle_percentile_in_corpus': float((corpus_angles < angle).mean() * 100),
       'crop_asset': 'great_wave_crop_224.png',
       'stages': {'input': 'the 224 x 224 crop after resize and centre crop (shown de-normalised)',
                  'patches': '16 x 16 patches of 14 x 14 pixels, each embedded as a 1024-number token, plus one class token (257 tokens); a learned position embedding is added and ln_pre applied before block 1',
                  'block_n': 'the residual-stream tokens after transformer block n of 24; a map shows each patch token\'s Euclidean norm',
                  'end_vector': 'the class token after the final LayerNorm (ln_post) and the 1024 -> 768 projection: encode_image',
                  'unit_vector': 'the end vector divided by its length: the point on the sphere'},
       'notes': ['patch and class-token norms are the residual stream (1024 wide) after each block, before the final LayerNorm; '
                 'use one colour scale per block; not attention',
                 'the end vector is 768 wide, after ln_post and the projection; its length does not continue the block-norm chain',
                 'the corpus mean direction is the normalised mean of the unit vectors of the corpus']}
with open(f'{OUT}/clipviz.json', 'w') as f:
    json.dump(out, f)
print('done; end length %.2f; angle to corpus mean %.1f deg (corpus percentile %.0f); class-token norms' % (
    out['end_vector_length'], angle, out['angle_percentile_in_corpus']), {k: round(v['class_token_norm'], 2) for k, v in maps.items()}, flush=True)
