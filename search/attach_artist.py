"""Adds artist, vec (from the P shards, aligned by path) and fidelity_clip (render-to-request CLIP chord) to an embed_images.py output.
Usage: python attach_artist.py EMB_NPZ SHARD_DIR"""
import glob
import sys

import numpy as np

emb_file, shard_dir = sys.argv[1], sys.argv[2]
z = dict(np.load(emb_file, allow_pickle=True))
shards = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f'{shard_dir}/shard[0-9].npz'))]
by_path = {}
for s in shards:
    for p, a, v in zip(s['paths'], s['artist'], s['vec']):
        by_path[str(p)] = (str(a), np.asarray(v, dtype=np.float32))
paths = [str(p) for p in z['paths']]
missing = [p for p in paths if p not in by_path]
assert not missing, f'{len(missing)} embedded paths are not in the shards, e.g. {missing[:2]}'
z['artist'] = np.array([by_path[p][0] for p in paths])
z['vec'] = np.stack([by_path[p][1] for p in paths])
if 'emb_clip-vit-l-14' in z:
    R = np.asarray(z['emb_clip-vit-l-14'], dtype=np.float64)
    V = z['vec'].astype(np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    z['fidelity_clip'] = np.sqrt(np.clip(2 - 2 * (R * V).sum(1), 0, None)).astype(np.float32)
    print('P fidelity (render-to-request chord, CLIP): median %.3f, q90 %.3f, q99 %.3f' % tuple(np.percentile(z['fidelity_clip'], (50, 90, 99))))
np.savez(emb_file, **z)
print('attached artist and vec to', emb_file, 'rows', len(paths), 'artists', len(np.unique(z['artist'])))
