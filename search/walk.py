"""Generator reach: frames along directions from named starts, painted by the decoder, then read in CLIP, DINOv3 and SigLIP, with the gate, nearest works and sheets.
Usage: walk.py OUT_DIR plan | render | score"""
import json
import os
import sys

import numpy as np
sys.path.insert(0, os.environ['DUROOM_REPO'] + '/data')

OUT, STEP = sys.argv[1], sys.argv[2]
REEMBED = os.environ['DUROOM_OUT'] + '/reembed/wikiart_catfm.npz'
AXIS = os.environ['DUROOM_OUT'] + '/newaxis/k32/newaxis.json'
OBS = os.environ['DUROOM_EMB']
LENSES = ['clip-vit-l-14', 'dinov3-l', 'siglip-so400m']
ANGLES = [0, 5, 10, 15, 20, 30, 40, 50]
RANKS = (1, 50, 300)
BATCH, GUIDANCE, K = 4, 3, 15
os.makedirs(f'{OUT}/frames', exist_ok=True)
os.makedirs(f'{OUT}/sheets', exist_ok=True)
unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)    # noqa: E731
chord = lambda cos: np.sqrt(np.clip(2 - 2 * cos, 0, None))    # noqa: E731


def noise(n, shape, seed):    # one draw repeated, so every frame of a seed starts from the same noise
    import torch
    return torch.randn((1, *shape), generator=torch.Generator().manual_seed(seed)).repeat(n, 1, 1, 1).cuda()


def load_variations():    # the vector replaces the image encoder's output
    import glob
    import torch
    from diffusers import StableDiffusionImageVariationPipeline
    from PIL import Image
    path = sorted(glob.glob(f"{os.environ['HF_HOME']}/hub/models--lambdalabs--sd-image-variations-diffusers/snapshots/*"))[-1]
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(path, safety_checker=None, image_encoder=None, feature_extractor=None, local_files_only=True).to('cuda')
    held = {}
    pipe._encode_image = lambda image, device, n, cfg: torch.cat([torch.zeros_like(held['e']), held['e']]) if cfg else held['e']
    dummy = Image.new('RGB', (224, 224))

    @torch.no_grad()
    def render(v, seed, guidance):
        held['e'] = v.unsqueeze(1)
        return pipe(image=[dummy] * len(v), guidance_scale=guidance, num_inference_steps=50, latents=noise(len(v), (4, 64, 64), seed)).images
    return render


if STEP == 'plan':
    from basis import capped, code
    from data import CLIP, ids, labels, lens, year
    import shutil
    X = unit(lens(CLIP).astype('float64'))
    raw = lens(REEMBED, 'clip_raw').astype('float64')
    raw_norm = np.linalg.norm(raw, axis=1)
    usable = (labels['artist'] != '') & (year >= 1000) & (year <= 2026)
    past, late = np.flatnonzero(usable & (year < 1900)), np.flatnonzero(usable & (year >= 1960))
    rows = np.flatnonzero(usable)
    rng = np.random.default_rng(7)
    starts = np.concatenate([rng.choice(past, 4, replace=False), rng.choice(late, 2, replace=False)])
    assert all((unit(raw[s]) @ X[s]) > 0.999 for s in starts), 'the raw CLIP file and the observer CLIP file disagree on a start'
    ax = json.load(open(AXIS))['cutoffs']['1900']['directions']
    V = code(X[capped(past, 20, np.random.default_rng(0))])[1]
    common = {'shift': np.array(ax['shift']), 'axis1': np.array(ax['axis1']), 'axis2': np.array(ax['axis2']), **{f'past_rank{r}': V[r - 1] for r in RANKS}}
    assert all(d.shape == (X.shape[1],) and abs(np.linalg.norm(d) - 1) < 1e-4 for d in common.values()), 'a direction is not a unit 768-vector'
    mean_dir = unit(X[rows].mean(0))
    median_angle = float(np.median(np.degrees(np.arccos(np.clip(X[rows] @ mean_dir, -1, 1)))))    # how far a typical work sits from the record's mean direction
    sigma = np.tan(np.radians(median_angle)) / np.sqrt(X.shape[1])    # unit(mean + sigma g) sits at about that angle
    frames, meta = [], {'starts': [], 'angles': ANGLES, 'median_raw_norm': float(np.median(raw_norm[rows])), 'median_angle_to_mean_deg': median_angle}

    def add(kind, s, name, t, seed, factor, v, d=None, nb=-1):
        frames.append({'kind': kind, 'start': int(s), 'dir': name, 'angle': t, 'seed': seed, 'factor': factor, 'nb': int(nb), 'v': v, 'd': np.zeros_like(v) if d is None else d,
                       'x': X[s] if s >= 0 else np.zeros_like(v)})

    for s in starts:
        x = X[s]
        sims = X[rows] @ x
        order = rows[np.argsort(-sims)]
        order = order[order != s]    # the start itself out, whatever its sort position
        nb = order[K - 1]    # the 15th other work
        dirs = dict(common)
        dirs['random1'], dirs['random2'] = rng.standard_normal(X.shape[1]), rng.standard_normal(X.shape[1])    # fresh per start
        dirs['to_neighbour15'] = X[nb]
        dirs = {n: unit(d0 - (d0 @ x) * x) for n, d0 in dirs.items()}
        meta['starts'].append({'row': int(s), 'id': int(ids[s]), 'year': int(year[s]), 'artist': labels['artist'][s], 'raw_norm': float(raw_norm[s]),
                               'neighbour15': {'row': int(nb), 'id': int(ids[nb]), 'year': int(year[nb]), 'angle_deg': float(np.degrees(np.arccos(np.clip(X[nb] @ x, -1, 1))))},
                               'direction_cosines': {a: {b: float(dirs[a] @ dirs[b]) for b in dirs if b > a} for a in dirs}})
        add('walk', s, 'none', 0, 0, 1.0, x * raw_norm[s])
        add('reseed', s, 'none', 0, 1, 1.0, x * raw_norm[s])    # second noise, start only
        add('neighbour_real', s, 'to_neighbour15', round(meta['starts'][-1]['neighbour15']['angle_deg'], 1), 0, 1.0, X[nb] * raw_norm[nb], dirs['to_neighbour15'], nb)    # the real picture on that line
        for name, d in dirs.items():
            for t in ANGLES[1:]:
                r = np.radians(t)
                add('walk', s, name, t, 0, 1.0, (np.cos(r) * x + np.sin(r) * d) * raw_norm[s], d)
            if name in ('shift', 'to_neighbour15'):    # second noise along two walks
                for t in (20, 40):
                    r = np.radians(t)
                    add('reseed', s, name, t, 1, 1.0, (np.cos(r) * x + np.sin(r) * d) * raw_norm[s], d)
        if s in starts[[0, 4]]:    # vector length scaled by 0.5 and 2, two starts
            for t in (0, 20):
                r = np.radians(t)
                for f in (0.5, 2.0):
                    add('length', s, 'shift', t, 0, f, (np.cos(r) * x + np.sin(r) * dirs['shift']) * raw_norm[s] * f, dirs['shift'])
    for i in range(24):    # random_cone: the mean direction plus noise, at a typical work's angle from the mean
        add('random_cone', -1, 'random', 0, 0, 1.0, unit(mean_dir + sigma * rng.standard_normal(X.shape[1])) * meta['median_raw_norm'])
    for i in range(12):    # random_sphere: a random point on the whole sphere
        add('random_sphere', -1, 'random', 0, 0, 1.0, unit(rng.standard_normal(X.shape[1])) * meta['median_raw_norm'])
    for i in range(12):    # record_mix: the mean of three random works' directions
        add('record_mix', -1, 'random', 0, 0, 1.0, unit(X[rng.choice(rows, 3, replace=False)].mean(0)) * meta['median_raw_norm'])
    for i, f in enumerate(frames):
        f['path'] = f'{OUT}/frames/{i:04d}.png'
    manifest = [{k: v for k, v in f.items() if k not in ('v', 'd', 'x')} for f in frames]
    vectors = {'v': np.stack([f['v'] for f in frames]), 'd': np.stack([f['d'] for f in frames]), 'x': np.stack([f['x'] for f in frames])}
    if os.path.exists(f'{OUT}/manifest.json'):    # resume: frames whose plan line and vector are unchanged are kept; a changed frame is removed; a changed manifest wipes all
        if os.path.exists(f'{OUT}/vectors.npz') and json.load(open(f'{OUT}/manifest.json')) == manifest and np.load(f'{OUT}/vectors.npz')['v'].shape == vectors['v'].shape:
            moved = np.flatnonzero(~np.all(np.isclose(np.load(f'{OUT}/vectors.npz')['v'], vectors['v'], atol=1e-4), axis=1))
            for i in moved:
                if os.path.exists(frames[i]['path']):
                    os.remove(frames[i]['path'])
            print(f'plan unchanged, frames kept; {len(moved)} frames with changed vectors removed', flush=True)
        else:
            shutil.rmtree(f'{OUT}/frames')
            os.makedirs(f'{OUT}/frames')
            print('plan changed, all frames removed', flush=True)
    np.savez(f'{OUT}/vectors.npz', **vectors)    # vectors first, manifest last: a crash between the two leaves a plan the next run treats as changed
    with open(f'{OUT}/manifest.json', 'w') as fh:
        json.dump(manifest, fh, indent=1)
    cone = [f['v'] for f in frames if f['kind'] == 'random_cone']
    meta['random_cone_angle_to_mean_deg'] = float(np.median(np.degrees(np.arccos(np.clip(unit(np.stack(cone)) @ mean_dir, -1, 1)))))
    with open(f'{OUT}/meta.json', 'w') as fh:
        json.dump(meta, fh, indent=1)
    print(len(frames), 'frames;', {k: sum(f['kind'] == k for f in frames) for k in ('walk', 'neighbour_real', 'reseed', 'length', 'random_cone', 'random_sphere', 'record_mix')},
          '| cone angle to mean', round(meta['random_cone_angle_to_mean_deg'], 1), 'vs record median', round(median_angle, 1), flush=True)
    sys.exit()

frames = json.load(open(f'{OUT}/manifest.json'))
Z = np.load(f'{OUT}/vectors.npz')
assert len(frames) == len(Z['v'])

if STEP == 'render':
    import torch
    render = load_variations()
    todo = [i for i, f in enumerate(frames) if not os.path.exists(f['path'])]    # a rerun resumes
    for seed in sorted({frames[i]['seed'] for i in todo}):
        group = [i for i in todo if frames[i]['seed'] == seed]
        for b in range(0, len(group), BATCH):
            part = group[b:b + BATCH]
            images = render(torch.tensor(Z['v'][part], dtype=torch.float32).cuda(), seed, GUIDANCE)
            for i, im in zip(part, images):
                im.save(frames[i]['path'] + '.tmp.png')
                os.replace(frames[i]['path'] + '.tmp.png', frames[i]['path'])
            print('seed', seed, b + len(part), 'of', len(group), flush=True)
    print('done', len(frames), flush=True)
    sys.exit()

# score
from PIL import Image, ImageDraw    # noqa: E402

from data import ids, labels, lens, year    # noqa: E402

sys.path.insert(0, os.environ['DUROOM_REPO'] + '/encode')
import utils    # noqa: E402

usable = (labels['artist'] != '') & (year >= 1000) & (year <= 2026)
rows, past = np.flatnonzero(usable), np.flatnonzero(usable & (year < 1900))
pos_rows, pos_past = {int(r): j for j, r in enumerate(rows)}, {int(r): j for j, r in enumerate(past)}
start_frame = {f['start']: i for i, f in enumerate(frames) if f['kind'] == 'walk' and f['angle'] == 0}    # each start's own render
for i, f in enumerate(frames):
    im = np.asarray(Image.open(f['path']).convert('L'), dtype='int16')
    f['flat'] = float(np.mean(np.abs(im - np.median(im)) <= 8))
    f['pixel_sd'] = float(im.std())
R = {}
for name in LENSES:
    E = unit(lens(f'{OBS}/{name}.npz').astype('float32'))
    encode = utils.load_encoder(name)[0]
    R[name] = unit(np.stack([encode(f['path']) for f in frames]).astype('float32'))
    sims_all, sims_past = R[name] @ E[rows].T, R[name] @ E[past].T
    for i, f in enumerate(frames):    # the start itself (and, for the real-neighbour frame, that neighbour) is not its own neighbour, so starts and random controls face the same pool
        for r in (f['start'], f.get('nb', -1)):
            if r in pos_rows:
                sims_all[i, pos_rows[r]] = -np.inf
            if r in pos_past:
                sims_past[i, pos_past[r]] = -np.inf
    for i, f in enumerate(frames):
        top = np.argsort(-sims_all[i])[:20]
        f[f'knn15_all_{name}'] = float(chord(np.sort(sims_all[i])[-K:]).mean())
        f[f'knn15_past_{name}'] = float(chord(np.sort(sims_past[i])[-K:]).mean())
        f[f'nearest_{name}'] = {'id': int(ids[rows[top[0]]]), 'year': int(year[rows[top[0]]]), 'chord': float(chord(sims_all[i][top[0]]))}
        f[f'median_year20_{name}'] = float(np.median(year[rows[top]]))
        if f['start'] in start_frame:
            f[f'd_start_{name}'] = float(chord(R[name][i] @ R[name][start_frame[f['start']]]))
# where the render went, in CLIP, against the request; the start's own render on the same direction is the baseline
Vu, D, Xs = unit(Z['v']), Z['d'], Z['x']
Rc = R['clip-vit-l-14'].astype('float64')
for i, f in enumerate(frames):
    f['d_request_clip'] = float(chord(Rc[i] @ Vu[i]))
    if f['start'] in start_frame and np.linalg.norm(D[i]) > 0:
        radial, along = Rc[i] @ Xs[i], Rc[i] @ D[i]
        r0 = Rc[start_frame[f['start']]]
        radial0, along0 = r0 @ Xs[i], r0 @ D[i]
        f['angle_achieved'] = float(np.degrees(np.arctan2(along, radial)))
        f['angle_achieved_start'] = float(np.degrees(np.arctan2(along0, radial0)))    # the start's own render, measured on this direction
        f['angle_achieved_rel'] = f['angle_achieved'] - f['angle_achieved_start']
        f['off_plane_share'] = float(1 - (radial ** 2 + along ** 2))    # share of the render's unit length outside the start-direction plane
        f['off_plane_share_start'] = float(1 - (radial0 ** 2 + along0 ** 2))
    if f['start'] in start_frame:
        f['angle_from_start_vector'] = float(np.degrees(np.arccos(np.clip(Rc[i] @ Xs[i], -1, 1))))
cfg = utils.load_config(os.environ['DUROOM_REPO'] + '/encode/config.yaml')
neg_emb, gate_encode, threshold = utils.load_coherence_gate(cfg)
for f in frames:
    g = np.asarray(utils.coherence_score(gate_encode(f['path']), neg_emb), dtype='float64')
    assert g.shape == (), 'the gate returned more than one number'
    f['gate'] = float(g)
# noise floor: the same vector from two noises, by angle
pairs = {(f['start'], f['dir'], f['angle']): i for i, f in enumerate(frames) if f['kind'] == 'walk'}
for name in LENSES:
    for i, f in enumerate(frames):
        if f['kind'] == 'reseed':
            j = pairs.get((f['start'], f['dir'], f['angle']))
            if j is not None:
                f[f'reseed_chord_{name}'] = float(chord(R[name][i] @ R[name][j]))
with open(f'{OUT}/scores.json', 'w') as fh:
    json.dump({'frames': frames, 'gate_threshold': float(threshold), 'k': K}, fh, indent=1)
# summary by direction type and angle, walks only
family = lambda d: 'record' if d in ('shift', 'axis1', 'axis2') else 'past_basis' if d.startswith('past_rank') else 'random' if d.startswith('random') else 'to_neighbour'    # noqa: E731
summary = {}
for fam in ('record', 'past_basis', 'random', 'to_neighbour'):
    summary[fam] = {}
    for t in ANGLES[1:]:
        sel = [f for f in frames if f['kind'] == 'walk' and f['angle'] == t and family(f['dir']) == fam]
        summary[fam][t] = {'n': len(sel), **{k: float(np.median([f[k] for f in sel])) for k in ('gate', 'flat', 'angle_achieved', 'angle_achieved_rel', 'off_plane_share', 'd_request_clip')},
                           'gate_pass': float(np.mean([f['gate'] >= threshold for f in sel])),
                           **{f'd_start_{nm}': float(np.median([f[f'd_start_{nm}'] for f in sel])) for nm in LENSES},
                           **{f'knn15_all_{nm}': float(np.median([f[f'knn15_all_{nm}'] for f in sel])) for nm in LENSES},
                           **{f'knn15_past_{nm}': float(np.median([f[f'knn15_past_{nm}'] for f in sel])) for nm in LENSES}}
for kind in ('random_cone', 'random_sphere', 'record_mix'):
    ctrl = [f for f in frames if f['kind'] == kind]
    summary[kind] = {'n': len(ctrl), 'gate_pass': float(np.mean([f['gate'] >= threshold for f in ctrl])), 'gate': float(np.median([f['gate'] for f in ctrl])), 'flat': float(np.median([f['flat'] for f in ctrl])),
                     'd_request_clip': float(np.median([f['d_request_clip'] for f in ctrl])), **{f'knn15_all_{nm}': float(np.median([f[f'knn15_all_{nm}'] for f in ctrl])) for nm in LENSES}}
st = [frames[i] for i in start_frame.values()]
summary['starts'] = {'gate': float(np.median([f['gate'] for f in st])), 'gate_pass': float(np.mean([f['gate'] >= threshold for f in st])), 'flat': float(np.median([f['flat'] for f in st])),
                     'd_request_clip': float(np.median([f['d_request_clip'] for f in st])), **{f'knn15_all_{nm}': float(np.median([f[f'knn15_all_{nm}'] for f in st])) for nm in LENSES}}
real = [f for f in frames if f['kind'] == 'neighbour_real']    # the real 15th neighbour rendered: a picture that sits on the walk's line at its own angle
summary['neighbour_real'] = [{k: f[k] for k in ('start', 'angle', 'gate', 'flat', 'angle_achieved', 'angle_achieved_rel', 'off_plane_share', 'd_request_clip', 'd_start_dinov3-l', 'knn15_all_dinov3-l')} for f in real]
summary['noise_floor'] = {nm: {t: float(np.median([f[f'reseed_chord_{nm}'] for f in frames if f'reseed_chord_{nm}' in f and f['angle'] == t])) for t in (0, 20, 40)} for nm in LENSES}
summary['length'] = [{k: f[k] for k in ('start', 'angle', 'factor', 'gate', 'flat', 'd_request_clip', 'd_start_clip-vit-l-14')} for f in frames if f['kind'] == 'length']
with open(f'{OUT}/summary.json', 'w') as fh:
    json.dump(summary, fh, indent=1)
print('gate threshold', threshold, '| starts gate', round(summary['starts']['gate'], 3), 'pass', summary['starts']['gate_pass'],
      '| controls gate pass / flat:', {k: (summary[k]['gate_pass'], round(summary[k]['flat'], 3)) for k in ('record_mix', 'random_cone', 'random_sphere')})
for fam in ('record', 'past_basis', 'random', 'to_neighbour'):
    print(fam, ' '.join(f"{t}:{s['gate_pass']:.2f}/{s['flat']:.2f}/{s['angle_achieved_rel']:.0f}/{s['d_start_dinov3-l']:.2f}" for t, s in summary[fam].items()), '(angle: gate pass / flat / achieved angle above the start baseline / DINOv3 chord from the start)')
print('real neighbours: angle, gate, achieved rel:', [(f['angle'], round(f['gate'], 2), round(f['angle_achieved_rel'], 0)) for f in summary['neighbour_real']])
print('noise floor by angle', {k: {t: round(v, 3) for t, v in d.items()} for k, d in summary['noise_floor'].items()})

# sheets: one per start (rows directions, columns angles), one for the controls
THUMB = 160


def sheet(cells, nrows, ncols, path, labels_left=()):
    W = 90 if labels_left else 0
    im = Image.new('RGB', (W + ncols * THUMB, nrows * (THUMB + 14)), (241, 235, 224))
    draw = ImageDraw.Draw(im)
    for (r, c, f, text) in cells:
        im.paste(Image.open(f['path']).convert('RGB').resize((THUMB, THUMB)), (W + c * THUMB, r * (THUMB + 14)))
        draw.text((W + c * THUMB + 2, r * (THUMB + 14) + THUMB), text, fill=(35, 31, 26))
    for r, lab in enumerate(labels_left):
        draw.text((2, r * (THUMB + 14) + THUMB // 2), lab, fill=(35, 31, 26))
    im.save(path)


dir_names = ['shift', 'axis1', 'axis2', *[f'past_rank{r}' for r in RANKS], 'random1', 'random2', 'to_neighbour15']
for s, i0 in start_frame.items():
    cells = []
    for r, dname in enumerate(dir_names):
        cells.append((r, 0, frames[i0], f"0 g{frames[i0]['gate']:.2f}"))
        for c, t in enumerate(ANGLES[1:], 1):
            f = next(f for f in frames if f['kind'] == 'walk' and f['start'] == s and f['dir'] == dname and f['angle'] == t)
            cells.append((r, c, f, f"{t} g{f['gate']:.2f} f{f['flat']:.2f} a{f['angle_achieved_rel']:.0f}"))
    sheet(cells, len(dir_names), len(ANGLES), f'{OUT}/sheets/start_{s}_{int(year[s])}.png', dir_names)
ctrl = [f for f in frames if f['kind'] in ('record_mix', 'random_cone', 'random_sphere')]
cells = [(i // 8, i % 8, f, f"{f['kind'].split('_')[1]} g{f['gate']:.2f} f{f['flat']:.2f}") for i, f in enumerate(ctrl)]
nctrl = (len(ctrl) + 7) // 8
extra = [f for f in frames if f['kind'] in ('neighbour_real', 'reseed', 'length')]
cells += [(nctrl + i // 8, i % 8, f, f"{f['kind'][:2]}{f['start'] % 10000} {f['dir'][:4]}{f['angle']}{'' if f['factor'] == 1 else ' x%g' % f['factor']} g{f['gate']:.2f}") for i, f in enumerate(extra)]
sheet(cells, nctrl + (len(extra) + 7) // 8, 8, f'{OUT}/sheets/controls.png')
print('done', flush=True)
