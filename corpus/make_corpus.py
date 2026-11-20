"""WIKIART_DIR layout from the Kaggle images (kaggle.com/datasets/yangyuwang/wikiart-images) and this folder's files; images matched by URL.
Usage: python make_corpus.py KAGGLE_DIR WIKIART_DIR"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src, out = sys.argv[1], sys.argv[2]
for d in ('artwork_images', 'artist_data', 'artwork_data'):
    os.makedirs(f'{out}/{d}', exist_ok=True)

ids = np.loadtxt(f'{HERE}/ids.txt', dtype=np.int64)
meta = pd.read_csv(f'{HERE}/metadata.csv.gz', low_memory=False)
meta.to_csv(f'{out}/artist_data/artwork_data_merged.csv', index=False)
np.savez(f'{out}/artwork_data/artwork_style_embeddings.npz', ids=ids)
np.savetxt(f'{out}/artwork_data/delete_ids.txt', np.loadtxt(f'{HERE}/delete_ids.txt', dtype=np.int64), fmt='%d')

by_url = {}
for u, n in zip(meta.image_url, pd.to_numeric(meta.image_n, errors='coerce')):
    if isinstance(u, str) and not np.isnan(n):
        by_url.setdefault(u, int(n))
want = set(ids.tolist())
linked, key_agrees, no_url = 0, 0, 0
for js in glob.glob(f'{src}/artwork_picture/*/*.json'):
    jpg = js[:-5] + '.jpg'
    if not os.path.exists(jpg):
        continue
    info = json.load(open(js))
    n = by_url.get(info.get('url'))
    if n is None:
        no_url += 1
        continue
    key_agrees += int(str(info.get('key', '')).lstrip('0') or '0') == n
    if n in want and not os.path.exists(f'{out}/artwork_images/{n}.jpg'):
        os.symlink(os.path.abspath(jpg), f'{out}/artwork_images/{n}.jpg')
        linked += 1

missing = sorted(want - {int(os.path.basename(p)[:-4]) for p in glob.glob(f'{out}/artwork_images/*.jpg')})
print(f'linked {linked}; key equals image_n for {key_agrees}; images whose URL is not in the metadata {no_url}')
print(f'works in ids.txt with no image: {len(missing)}' + (f' (first: {missing[:10]})' if missing else ''))
