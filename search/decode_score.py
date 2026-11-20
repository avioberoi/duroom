"""Re-embeds the rendered frames in each lens and compares them with the source work, its neighbours and the record: distance to the source, rank, nearest works, midpoints, walks.
Usage: decode_score.py DIR [LIMIT=16]   (DIR holds manifest.json from decode.py; LIMIT is how many works each setting is scored on, 0 for all)"""
import json
import os
import sys

import numpy as np
from PIL import Image


DIR = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 16
sys.path.insert(0, os.environ['DUROOM_REPO'] + '/data')
sys.path.insert(0, os.environ['DUROOM_REPO'] + '/encode')
from data import IMG, card, ids, labels, lens, year  # noqa: E402
import utils  # noqa: E402
OBS = os.environ['DUROOM_EMB']
LENSES = ['clip-vit-l-14', 'dinov3-l', 'siglip-so400m']
AXIS = os.environ['DUROOM_OUT'] + '/newaxis/k32/newaxis.json'

frames = json.load(open(f'{DIR}/manifest.json'))
axis = np.array(json.load(open(AXIS))['cutoffs']['1900']['directions']['axis1'], dtype='float32')
idx = lambda **kw: [i for i, f in enumerate(frames) if all(f.get(k) == v for k, v in kw.items())]    # noqa: E731
base0, base1 = idx(kind='recon', setting='base', seed=0), idx(kind='recon', setting='base', seed=1)
rows = np.array([frames[i]['rows'][0] for i in base0])
assert rows.tolist() == [frames[i]['rows'][0] for i in base1]
n = len(ids)
dist = lambda sim: np.sqrt(np.clip(2 - 2 * sim, 0, None))    # noqa: E731
med = lambda x: float(np.median(x))    # noqa: E731
result = {'n_record': n, 'lenses': {}}


def nearest(R, E, m):
    """Top m works for each row of R, by cosine."""
    return np.argsort(-(R @ E.T), axis=1)[:, :m]


for name in LENSES:
    E = lens(f'{OBS}/{name}.npz').astype('float32')
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    encode = utils.load_encoder(name)[0]
    R = np.stack([encode(f['path']) for f in frames]).astype('float32')
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    control = np.stack([encode(f'{IMG}/{ids[r]}.jpg') for r in rows]).astype('float32')
    control /= np.linalg.norm(control, axis=1, keepdims=True)

    r0, s = R[base0], E[rows]
    sim = r0 @ E.T
    own = sim[np.arange(len(rows)), rows]
    rank = (sim > own[:, None]).sum(1)    # works nearer the render than its source; 0 is best
    top = np.sort(s @ E.T, axis=1)[:, -11:][:, ::-1]    # first is the work itself
    d1, d10 = dist(top[:, 1]), dist(top[:, 10])
    rng = np.random.default_rng(0)
    random_rows = rng.choice(n, 2000, replace=False)
    d_random = dist(r0 @ E[random_rows].T)
    d_src = dist(own)
    sim1 = R[base1] @ E.T
    own1 = sim1[np.arange(len(rows)), rows]
    d_src1, rank1 = dist(own1), (sim1 > own1[:, None]).sum(1)    # the second seed, for the paired comparison
    del sim1
    among = (r0 @ s.T)
    rank_among = (among > np.diag(among)[:, None]).sum(1)    # among the 64 sources only, so a shared offset of renders from artworks does not count
    near = [[w for w in row if w != r][:10] for row, r in zip(nearest(r0, E, 11), rows)]    # the source itself does not count
    genre, artist = labels['genre'], labels['artist']
    entry = {
        'control_cos_reencoded_source_median': med((control * s).sum(1)), 'control_cos_min': float((control * s).sum(1).min()),
        'd_source_median': med(d_src), 'd_random_median': med(d_random), 'd_real_to_random_median': med(dist(s @ E[random_rows].T)), 'source_d1_median': med(d1), 'source_d10_median': med(d10),
        'rank_median': med(rank), 'rank_median_share': med(rank) / n, 'share_rank_within_1pct': float((rank < n / 100).mean()),
        'share_nearer_than_own_d10': float((d_src < d10).mean()), 'share_nearer_than_own_d1': float((d_src < d1).mean()),
        'rank_among_64_median': med(rank_among), 'top1_among_64': float((rank_among == 0).mean()), 'chance_top1_among_64': 1 / len(rows),
        'd_seed_median': med(np.linalg.norm(R[base0] - R[base1], axis=1)),
        'same_artist_in_nearest_10': float(np.mean([(artist[near[j]] == artist[r]).mean() for j, r in enumerate(rows)])),
        'same_genre_in_nearest_10': float(np.mean([(genre[near[j]] == genre[r]).mean() for j, r in enumerate(rows) if genre[r]])),
        'per_work': [{'row': int(r), 'id': int(ids[r]), 'd_source': float(d_src[j]), 'rank': int(rank[j]), 'd10': float(d10[j]), 'd_source_seed1': float(d_src1[j]), 'rank_seed1': int(rank1[j]),
                      'd_seed': float(np.linalg.norm(R[base0[j]] - R[base1[j]]))} for j, r in enumerate(rows)],
    }
    entry['settings'] = {}
    for setting in sorted({frames[i]['setting'] for i in idx(kind='recon')}):
        chosen = idx(kind='recon', setting=setting, seed=0)[:LIMIT or None]    # the first LIMIT works, in every setting
        o = np.array([frames[i]['rows'][0] for i in chosen])
        si = R[chosen] @ E.T
        ow = si[np.arange(len(o)), o]
        entry['settings'][setting] = {'d_source_median': med(dist(ow)), 'rank_median': med((si > ow[:, None]).sum(1)), 'rows': o.tolist(), 'd_source': dist(ow).tolist(), 'rank': (si > ow[:, None]).sum(1).tolist()}

    mid = idx(kind='midpoint')
    if mid:    # the n256 run has no midpoints
        parents = np.array([frames[i]['rows'] for i in mid])
        target = E[parents].mean(1)
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        si = R[mid] @ E.T
        p_rank = np.stack([(si > si[np.arange(len(mid)), parents[:, c]][:, None]).sum(1) for c in (0, 1)], axis=1)
        entry['midpoint'] = {'d_to_target_median': med(np.linalg.norm(R[mid] - target, axis=1)), 'parent_d_median': med(dist((E[parents[:, 0]] * E[parents[:, 1]]).sum(1))),
                             'd_nearest_work_median': med(dist(si.max(1))), 'better_parent_rank_median': med(p_rank.min(1)), 'worse_parent_rank_median': med(p_rank.max(1))}

    walks = {}
    for r in sorted({frames[i]['rows'][0] for i in idx(kind='walk')}):
        chosen = sorted(idx(kind='walk', rows=[r]), key=lambda i: frames[i]['alpha'])
        z = R[chosen]
        near = nearest(z, E, 20)
        walks[int(ids[r])] = {'alpha': [frames[i]['alpha'] for i in chosen], 'step': np.linalg.norm(np.diff(z, axis=0), axis=1).tolist(), 'from_start': np.linalg.norm(z - z[0], axis=1).tolist(),
                              'median_year_nearest_20': [float(np.nanmedian(year[m])) for m in near]}
        if name == LENSES[0]:    # the axis lives in CLIP
            walks[int(ids[r])]['loading_change'] = ((z - z[0]) @ axis).tolist()
    entry['walk'] = walks
    result['lenses'][name] = entry
    json.dump(result, open(f'{DIR}/score.json', 'w'), indent=1)    # after each lens, so a failure keeps the earlier ones
    print(name, {k: round(v, 4) for k, v in entry.items() if isinstance(v, float)}, flush=True)
    del encode, E

held = [result['lenses'][x] for x in LENSES[1:]]
result['verdict'] = {'pass': all(e['rank_median_share'] <= .01 for e in held), 'strong': all(e['share_nearer_than_own_d10'] > .5 for e in held)}
print(result['verdict'], flush=True)
json.dump(result, open(f'{DIR}/score.json', 'w'), indent=1)

os.makedirs(f'{DIR}/thumbs', exist_ok=True)
for i, f in enumerate(frames):
    im = Image.open(f['path']).convert('RGB')
    im.thumbnail((360, 360))
    im.save(f'{DIR}/thumbs/f{i:04d}.jpg', quality=85)
    for r in f['rows']:
        card(r, DIR)
print('done', flush=True)
