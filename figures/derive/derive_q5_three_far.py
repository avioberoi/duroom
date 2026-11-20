"""Display data for the three-ways-to-be-far figure: ring radii, neighbour rings, centre angle, from distance/clip-vit-l-14/{distance.json,scores.npz}, q5_far.json and measured_sphere.json, to figures/data/q5_three_far.json.
Usage: python derive_q5_three_far.py"""
import json
import math
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ['DUROOM_OUT'])
HERE = OUT / 'figures' / 'data'
SRC = OUT / 'distance' / 'clip-vit-l-14'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))


def write(name, data):
    text = json.dumps(data, ensure_ascii=False, indent=1) + '\n'
    assert '"id":' not in text and '"ids":' not in text
    (HERE / f'{name}.json').write_text(text, encoding='utf-8')


distance = read(SRC / 'distance.json')
far = read(HERE / 'q5_far.json')
sphere = read(HERE / 'measured_sphere.json')
clip = next(e for e in far['encoders'] if e['encoder'] == 'CLIP')
examples = {e['score_key']: e for e in clip['examples']}

with np.load(SRC / 'scores.npz') as z:
    ids = z['ids']; d10 = z['d10_all']; local = z['local_all']; centre = z['centre']
    nb_all = z['neighbours_all']

# The scored works are the named works dated 1000-2026; the other rows only serve as neighbours.
scored = ~np.isnan(d10)
n = int(scored.sum())
assert n == distance['n'] == sphere['population']
angle = lambda c: np.degrees(2 * np.arcsin(np.asarray(c) / 2))   # chord between unit vectors -> angle


def rank(values, i):
    """1 = farthest among the scored works."""
    return int(1 + np.sum(values[scored] > values[i]))


def row_of(key, score_key):
    """Join the q5_far.json card to its distance.json card and its scores.npz row."""
    card = distance['cards'][key][0]
    public = examples[score_key]
    assert (card['title'], card['year']) == (public['title'], public['year'])
    (i,) = np.flatnonzero(ids == card['id'])
    assert np.isclose(d10[i], public['absolute']) and np.isclose(local[i], public['local'])
    assert np.isclose(centre[i], public['centre'])
    assert (HERE.parent / public['asset']).exists()
    work = {k: public[k] for k in ('artist', 'title', 'year', 'asset', 'source')}
    return int(i), work


def in_lists(i, neighbours):
    """How many of the work's ten neighbours have the work among their own ten."""
    return [bool(i in set(neighbours[j])) for j in neighbours[i]]


def place(ring, own, inside, start, lo=None):
    """Drawn positions of the ten neighbours under the measured ring constraints; angles are evenly spaced because the directions are not stored."""
    own = np.asarray(own)
    if lo is None:
        lo = min(ring, max(r for r, a in zip(own, inside) if not a) + .03)
    radii = lo + (ring - lo) * np.arange(10) / 9
    radii = np.where(inside, np.minimum(radii, own), radii)
    assert radii[-1] == ring
    assert np.all(radii <= ring + 1e-12)
    assert all((r >= o) if not a else (r <= o) for r, o, a in zip(radii, own, inside))
    t = np.radians(start + 36 * np.arange(10))
    return [[round(float(r * math.cos(a)), 4), round(float(r * math.sin(a)), 4)] for r, a in zip(radii, t)]


i_ice, ice = row_of('top_d10_not_local', 'd10_all')
i_fire, fire = row_of('top_local_not_d10', 'local_all')
i_alston, alston = row_of('top_centre_not_local', 'centre')

corpus_d10 = d10[scored]; corpus_local = local[scored]; corpus_angle = angle(centre[scored])
# The centre panel stands on the sphere figure: its percentiles must reproduce exactly.
for p, value in sphere['percentiles'].items():
    assert abs(np.percentile(corpus_angle, float(p)) - value) < 1e-3, p

# Absolute: Ice Hockey's ring among the corpus's median and 99th-percentile rings.
own_ice = d10[nb_all[i_ice]]
inside_ice = in_lists(i_ice, nb_all)
absolute = {'key': 'absolute', 'work': ice, 'ring': float(d10[i_ice]), 'rank': rank(d10, i_ice),
            'neighbour_rings': [round(float(r), 4) for r in own_ice],
            'neighbour_median': float(np.median(own_ice)), 'local': float(local[i_ice]),
            'local_rank': rank(local, i_ice),
            'layout': {'drawn': True, 'neighbours': place(float(d10[i_ice]), own_ice, inside_ice, 112)}}

# Local: the Forest Fire's ring against its ten neighbours' own rings (neighbours by any artist).
own_fire = d10[nb_all[i_fire]]
inside_fire = in_lists(i_fire, nb_all)
assert not any(inside_fire)                      # it is in none of its neighbours' own ten
assert np.isclose(d10[i_fire] / np.median(own_fire), local[i_fire])
fire_layout = place(float(d10[i_fire]), own_fire, inside_fire, 99)
local_panel = {'key': 'local', 'work': fire, 'ring': float(d10[i_fire]),
               'neighbour_rings': [round(float(r), 4) for r in own_fire],
               'neighbour_median': float(np.median(own_fire)), 'ratio': float(local[i_fire]),
               'rank': rank(local, i_fire), 'absolute_rank': rank(d10, i_fire),
               'absolute_share': rank(d10, i_fire) / n, 'in_neighbour_lists': int(sum(inside_fire)),
               'layout': {'drawn': True, 'neighbours': fire_layout}}

# Centre: Alston's chord to the unit average direction, read as the polar angle on the sphere.
centre_panel = {'key': 'centre', 'work': alston, 'distance': float(centre[i_alston]),
                'angle': float(angle(centre[i_alston])), 'rank': rank(centre, i_alston),
                'local_rank': rank(local, i_alston), 'layout': {'drawn': True, 'side': 'right'}}
assert abs(2 * math.sin(math.radians(centre_panel['angle']) / 2) - centre_panel['distance']) < 1e-9

write('q5_three_far', {
    'encoder': 'CLIP', 'n': n, 'k': distance['k'],
    'definition': ('Absolute: distance to the 10th nearest work. Local: that distance divided by the median of '
                   'its ten neighbours\u2019 own 10th-neighbour distances; neighbours by any artist count. Centre: '
                   'distance to the unit average direction, read on the sphere as the angle 2\u00b7asin(d/2).'),
    'drawn': ('Ring radii, chord lengths and angles are measured. Where the neighbours sit in the plane and the '
              'side of the pole are drawn, not stored.'),
    'corpus': {'absolute': {'median': float(np.median(corpus_d10)), 'p99': float(np.percentile(corpus_d10, 99))},
               'local': {'median': float(np.median(corpus_local)), 'p99': float(np.percentile(corpus_local, 99))},
               'centre_angle': {'median': sphere['percentiles']['50'], 'p99': sphere['percentiles']['99'],
                                'nearest': sphere['percentiles']['0'], 'farthest': sphere['percentiles']['100']}},
    'panels': [absolute, local_panel, centre_panel]})
print('wrote q5_three_far.json')
