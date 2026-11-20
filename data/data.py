"""The corpus, shared by every script: filtered rows, labels, lens files, thumbnails."""
import os

import numpy as np
import pandas as pd
from PIL import Image

ART = os.environ['WIKIART_DIR'] + '/artwork_data'
META = os.environ['WIKIART_DIR'] + '/artist_data/artwork_data_merged.csv'
IMG = os.environ['WIKIART_DIR'] + '/artwork_images'
CLIP = os.environ['DUROOM_EMB'] + '/clip-vit-l-14.npz'

all_ids = np.load(f'{ART}/artwork_style_embeddings.npz')['ids']
keep = ~np.isin(all_ids, np.loadtxt(f'{ART}/delete_ids.txt', dtype=int))
ids = all_ids[keep]

_m = pd.read_csv(META, low_memory=False).dropna(subset=['image_n'])
_m['image_n'] = _m.image_n.astype(int)
_m = _m.drop_duplicates('image_n', keep='first').set_index('image_n').reindex(ids)
_first = lambda s: s.fillna('').astype(str).str.split('|').str[0].str.strip()  # noqa: E731
labels = {'artist': _m.Artist_name.fillna('').astype(str).values, 'style': _first(_m.Style).values, 'genre': _first(_m.Genre).values}
title, year = _m['Artwork Name'].fillna('').astype(str).values, pd.to_numeric(_m.Year, errors='coerce').values


def lens(file, key='embeddings'):
    z = np.load(file)
    assert (z['ids'] == all_ids).all()      # row i must be the same artwork in every file
    return z[key][keep]


def card(i, out, dist=None):
    i, dst = int(i), f'{out}/thumbs/{ids[i]}.jpg'
    os.makedirs(f'{out}/thumbs', exist_ok=True)
    if not os.path.exists(dst):
        im = Image.open(f'{IMG}/{ids[i]}.jpg').convert('RGB')
        im.thumbnail((420, 420))
        im.save(dst, quality=88)
    return {'id': int(ids[i]), 'title': title[i][:70], 'year': None if np.isnan(year[i]) else int(year[i]), 'dist': dist,
            **{k: v[i] for k, v in labels.items()}}
