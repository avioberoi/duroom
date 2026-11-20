"""Can a vector become an image? Paints known works from their raw CLIP-L vectors, midpoints of unrelated works and a walk along the record's axis 1 with the image-variations decoder; with a fourth argument n256, a fresh draw of 256 works at guidance 3 and 10 instead.
Usage: decode.py variations OUT_DIR plan | render [n256]   (plan needs pandas, render the decoder's libraries); decode_score.py scores the frames"""
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
sys.path.insert(0, os.environ['DUROOM_REPO'] + '/data')

TAG = sys.argv[4] if len(sys.argv) > 4 else ''
assert TAG in ('', 'n256'), TAG
DECODER, OUT, STEP = sys.argv[1], f'{sys.argv[2]}/{sys.argv[1]}' + (f'_{TAG}' if TAG else ''), sys.argv[3]
assert DECODER == 'variations', DECODER
REEMBED = os.environ['DUROOM_OUT'] + '/reembed/wikiart_catfm.npz'
AXIS = os.environ['DUROOM_OUT'] + '/newaxis/k32/newaxis.json'
SETTINGS = {'base': (None, 3), 'high_guidance': (None, 10)}
if TAG == 'n256':    # guidance 3 and 10 on the same works
    SETTINGS = {'base': (None, 3), 'alt': (None, 10)}
ALPHAS = [0, .05, .1, .15, .2, .25, .3, .4, .6]
BATCH = 4
os.makedirs(f'{OUT}/frames', exist_ok=True)


def noise(n, shape, seed):
    """One draw repeated, so every frame of a seed starts from the same noise and differences come from the vector."""
    return torch.randn((1, *shape), generator=torch.Generator().manual_seed(seed)).repeat(n, 1, 1, 1).cuda()


def load_variations():
    from diffusers import StableDiffusionImageVariationPipeline
    path = sorted(glob.glob(f"{os.environ['HF_HOME']}/hub/models--lambdalabs--sd-image-variations-diffusers/snapshots/*"))[-1]
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(path, safety_checker=None, image_encoder=None, feature_extractor=None, local_files_only=True).to('cuda')
    held = {}
    pipe._encode_image = lambda image, device, n, cfg: torch.cat([torch.zeros_like(held['e']), held['e']]) if cfg else held['e']    # the vector replaces the image encoder's output
    dummy = Image.new('RGB', (224, 224))

    @torch.no_grad()
    def render(v, seed, prompt, guidance):
        held['e'] = v.unsqueeze(1)
        return pipe(image=[dummy] * len(v), guidance_scale=guidance, num_inference_steps=50, latents=noise(len(v), (4, 64, 64), seed)).images
    return render


if STEP == 'plan':
    from data import labels, lens, year
    raw = lens(REEMBED, 'clip_raw').astype('float64')
    named = np.flatnonzero(labels['artist'] != '')
    rng = np.random.default_rng(11)    # fixed draw: the same works in every run
    works = rng.choice(named, 64, replace=False)
    pairs = rng.choice(named, (16, 2), replace=False)
    starts = rng.choice(np.flatnonzero((labels['artist'] != '') & (year < 1900)), 3, replace=False)
    axis = np.array(json.load(open(AXIS))['cutoffs']['1900']['directions']['axis1'])
    assert abs(np.linalg.norm(axis) - 1) < 1e-6
    unit = lambda x: x / np.linalg.norm(x)    # noqa: E731

    if TAG == 'n256':    # a fresh draw; the 64 works of the default run are left out so the two samples are independent
        works = np.random.default_rng(256).choice(np.setdiff1d(named, works), 256, replace=False)
    frames = [{'kind': 'recon', 'rows': [int(r)], 'seed': s, 'setting': 'base', 'v': raw[r]} for s in (0, 1) for r in works]
    frames += [{'kind': 'recon', 'rows': [int(r)], 'seed': 0, 'setting': name, 'v': raw[r]} for name in SETTINGS if name != 'base' for r in (works if TAG else works[:16])]
    if not TAG:
        frames += [{'kind': 'midpoint', 'rows': [int(a), int(b)], 'seed': 0, 'setting': 'base', 'v': unit(unit(raw[a]) + unit(raw[b])) * (np.linalg.norm(raw[a]) + np.linalg.norm(raw[b])) / 2} for a, b in pairs]
        frames += [{'kind': 'walk', 'rows': [int(r)], 'alpha': a, 'seed': 0, 'setting': 'base', 'v': unit(unit(raw[r]) + a * axis) * np.linalg.norm(raw[r])} for r in starts for a in ALPHAS]
    for i, f in enumerate(frames):
        f['path'] = f'{OUT}/frames/{i:04d}.png'
    json.dump([{k: v for k, v in f.items() if k != 'v'} for f in frames], open(f'{OUT}/manifest.json', 'w'), indent=1)
    np.save(f'{OUT}/vectors.npy', np.stack([f['v'] for f in frames]))
    sys.exit()

frames = json.load(open(f'{OUT}/manifest.json'))
vectors = np.load(f'{OUT}/vectors.npy')
assert len(frames) == len(vectors)
for f, v in zip(frames, vectors):
    f['v'] = v
render = load_variations()
todo = [f for f in frames if not os.path.exists(f['path'])]    # a rerun resumes
for key in sorted({(f['seed'], f['setting']) for f in todo}):
    group = [f for f in todo if (f['seed'], f['setting']) == key]
    prompt, guidance = SETTINGS[key[1]]
    for b in range(0, len(group), BATCH):
        part = group[b:b + BATCH]
        images = render(torch.tensor(np.stack([f['v'] for f in part]), dtype=torch.float32).cuda(), key[0], prompt, guidance)
        for f, im in zip(part, images):
            im.save(f['path'] + '.tmp.png')    # a killed job leaves no truncated frame for a rerun to trust
            os.replace(f['path'] + '.tmp.png', f['path'])
        print(key, b + len(part), 'of', len(group), flush=True)
print('done', len(frames), flush=True)
