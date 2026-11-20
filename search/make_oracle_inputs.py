"""Builds inputs/oracle_targets.json and inputs/oracle_inputs.npz under $DUROOM_OUT for oracle.py: far, faithful later works as targets, and the three nearest targets per seed start.
Usage: python make_oracle_inputs.py"""
import json
import os

import numpy as np

B = os.environ['DUROOM_OUT']
O = f'{B}/sphere'
P = np.load(f'{O}/push_emb.npz', allow_pickle=True)
Q = np.load(f'{O}/pos_emb.npz', allow_pickle=True)
unit = lambda X: np.asarray(X, float) / np.linalg.norm(np.asarray(X, float), axis=1, keepdims=True)  # noqa: E731
E = {n: unit(P['emb_' + n]) for n in ('dinov3-l', 'siglip-so400m')}
q99 = {}
for n in E:
    Dm = np.sqrt(np.clip(2 - 2 * E[n] @ E[n].T, 0, None)); np.fill_diagonal(Dm, np.inf); Dm[P['artist'][:, None] == P['artist'][None, :]] = np.inf
    q99[n] = np.percentile(np.sort(Dm, axis=1)[:, :15].mean(1), 99)
# targets: pos works farther from P than P's own q99 in both observers (same-artist rows left out), with CLIP fidelity at most 0.7785
ok = np.ones(512, bool)
for n in E:
    X = unit(Q['emb_' + n]); s = np.sort(np.sqrt(np.clip(2 - 2 * X @ E[n].T, 0, None)), axis=1)[:, :15].mean(1); ok &= s > q99[n]
fid = np.asarray(Q['fidelity_clip'], float); T = ok & (fid <= 0.7785); VT = unit(Q['vec'])[T]; names = np.asarray(Q['artist'])[T]
print('targets', T.sum())
corpus = np.load(os.environ['DUROOM_EMB'] + '/clip-vit-l-14.npz')['embeddings'].astype('float32')
corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
pool_rows = np.load(f'{O}/pools/starts_pre1900.npz', allow_pickle=True)['rows_full']
out = []
for seed in (1, 2, 3):
    tag = f'oracle_s{seed}'
    start_idx = int(pool_rows[np.random.default_rng(seed + 31337).integers(len(pool_rows))])    # the loop's start rule (search_v2 --start seedrow)
    s = unit(corpus[start_idx].copy()[None])[0]
    ang = np.degrees(np.arccos(np.clip(VT @ s, -1, 1))); o = np.argsort(ang)[:3]
    print(f'{tag} start (row {start_idx}): nearest targets at {ang[o[0]]:.1f}, {ang[o[1]]:.1f}, {ang[o[2]]:.1f} deg; median to all {T.sum()} targets {np.median(ang):.1f}; artists {list(names[o])}')
    out.append({'tag': tag, 'start_idx': start_idx, 'start_vec': s.tolist(), 'target_idx_in_pos': [int(i) for i in np.where(T)[0][o]], 'target_angles': [float(ang[i]) for i in o]})
os.makedirs(f'{B}/inputs', exist_ok=True)
json.dump({'targets_pos_index': [int(i) for i in np.where(T)[0]], 'starts': out}, open(f'{B}/inputs/oracle_targets.json', 'w'))
print('saved oracle_targets.json')

j = json.load(open(f'{B}/inputs/oracle_targets.json'))
tidx = np.array(j['targets_pos_index']); targets = np.asarray(Q['vec'], float)[tidx]; targets /= np.linalg.norm(targets, axis=1, keepdims=True)
names = np.asarray(Q['artist'])[tidx].astype(str)
starts = np.array([s['start_vec'] for s in j['starts']], float); tags = np.array([s['tag'] for s in j['starts']]).astype(str)
nearest = np.array([[int(np.where(tidx == t)[0][0]) for t in s['target_idx_in_pos']] for s in j['starts']])
np.savez(f'{B}/inputs/oracle_inputs.npz', starts=starts, start_tags=tags, targets=targets, target_names=names, nearest=nearest)
print('starts', starts.shape, 'targets', targets.shape, 'nearest', nearest.tolist(), 'names', [list(names[r]) for r in nearest])
