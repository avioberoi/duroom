"""Fan data: leftover share to angle (theta = asin(sqrt(share))), ring positions of six thumbnails, from leftover/clip-vit-l-14_k32_cap0/leftover.json, to figures/data/q3_fan.json.
Usage: python derive_q3_fan.py"""
import json
import math
import os
from pathlib import Path

from PIL import Image
from _artwork_export import card

OUT = Path(os.environ['DUROOM_OUT'])
HERE = OUT / 'figures' / 'data'
HERE.mkdir(parents=True, exist_ok=True)
IMAGES = Path(os.environ['WIKIART_DIR']) / 'artwork_images'
COMMON = OUT / 'leftover' / 'clip-vit-l-14_k32_cap0' / 'leftover.json'
RING_SEP = 14.5           # drawn: minimum angular gap between thumbnails on the outer ring, degrees
RING_LO, RING_HI = 7.5, 85.0


def write(name, data, folder=HERE):
    text = json.dumps(data, ensure_ascii=False, indent=1) + '\n'
    assert '"id":' not in text and '"ids":' not in text
    (folder / f'{name}.json').write_text(text, encoding='utf-8')


def theta(share):
    return math.degrees(math.asin(math.sqrt(share)))


def ring(angles):
    """Ring angles for thumbnails: measured order kept, pushed apart to >= RING_SEP (pool adjacent violators), clipped to [RING_LO, RING_HI]; a drawn layout."""
    n = len(angles)
    order = sorted(range(n), key=lambda i: (angles[i], i))
    q = [angles[i] - k * RING_SEP for k, i in enumerate(order)]
    blocks = []                                   # [sum, count]
    for v in q:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    flat = [b[0] / b[1] for b in blocks for _ in range(b[1])]
    lo, hi = RING_LO, RING_HI - (n - 1) * RING_SEP
    out = [0.0] * n
    for k, i in enumerate(order):
        out[i] = min(max(flat[k], lo), hi) + k * RING_SEP
    return out


d = json.loads(COMMON.read_text(encoding='utf-8'))
assert d['k'] == 32 and d['cap'] == 0
p = d['share_percentiles_1_10_50_90_99']
rows = {}
for lst in d['cards'].values():
    for r in lst:
        rows[(r['artist'], r['year'], r['title'])] = r

WANT = [  # (artist slug, year, title, short label)
    ('george-romney', 1764, 'Mrs Margaret Ainslie', 'Romney'),
    ('richard-hamilton', 1971, 'A Beautiful Little Shit for Dieter', 'Hamilton'),
    ('ellsworth-kelly', 2000, 'Red Yellow Blue', 'Kelly'),
    ('alexander-calder', 1963, 'POLYGONS ON TRIANGLES', 'Calder'),
    ('mimmo-rotella', 2003, 'Modi', 'Rotella'),
    ('noma-bar', 2018, 'Sex & Power, MeToo', 'Noma Bar'),
]
works = []
for artist, year, title, short in WANT:
    r = rows[(artist, year, title)]
    c = card(r, IMAGES, share=r['share'], share_past=r['share_past'])
    if artist == 'alexander-calder':
        c['title'] = 'Polygons on Triangles'
    with Image.open(HERE.parent / c['asset']) as im:
        c['size'] = list(im.size)
    c.update(short=short, angle=theta(r['share']), angle_past=theta(r['share_past']))
    works.append(c)
a_ring = ring([w['angle'] for w in works])
b_ring = ring([w['angle_past'] for w in works])
for w, a, b in zip(works, a_ring, b_ring):
    w['ring'] = a
    w['ring_past'] = b

out = {
    'encoder': 'CLIP', 'k': d['k'], 'n': d['n'],
    'reference': {
        'median': {'share': p[2], 'angle': theta(p[2])},
        'cut': {'share': p[4], 'angle': theta(p[4])},
        'first': {'share': p[0], 'angle': theta(p[0])},
    },
    'ring_sep': RING_SEP, 'ring_range': [RING_LO, RING_HI],
    'definition': 'Each work is drawn as a point on a quarter circle at its measured angle theta, the angle between its '
                  'offset from the average and the plane of the 32 directions. Its leftover share is sin^2(theta). '
                  'The basis from the works before 1900 includes those works and takes its own average. '
                  'The ring positions of the thumbnails are drawn, not measured.',
    'works': works,
}
write('q3_fan', out)

for w in works:
    print(f"{w['short']:9s} {w['angle']:5.1f} -> {w['angle_past']:5.1f}   ring {w['ring']:5.1f} -> {w['ring_past']:5.1f}")
for k, v in out['reference'].items():
    print(k, round(v['share'], 4), round(v['angle'], 2))
