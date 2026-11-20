"""Gallery of far, faithful pictures with no real work within the manifold line: start, painted, nearest real work in SigLIP, fact block beside; the Fra Angelico start is named only, the other start is shown. Reads figures/marks/SHOW-off1900.csv.
Usage: python search_finds_off1900_v5.py"""
import os

import marks_v1 as am
import search_finds_off1900_v4 as v4
from search_finds_v1 import FS, make
from _style import *

COSSIERS = str(HERE / 'finds' / 'start_s6.jpg')
is_cossiers = lambda r: 'cossiers' in r['start_artist'].lower()  # noqa: E731


def start_cell(P, r, x, top, cell, fs):
    if is_cossiers(r):
        P.image(COSSIERS, x, top + cell, cell, cell)
        return cell + 0.03 + P.block(am.off_start(r), x, top + cell + 0.03, cell, fs)
    return P.block(am.off_start(r) + [[('(named only)', False)]], x, top - 0.02, cell, fs)


def desk():
    W, gap, cell = 8.8, 0.2, 1.95
    xs = [0, cell + gap, 2 * (cell + gap), 3 * (cell + gap)]
    fw = W - xs[3]
    P, fs = am.MPlate(W), FS['desk']
    y = 0.0
    for x, h in zip(xs, v4.HEADS):
        P.text(x, y + 0.16, h, fs['head'], MUTE)
    y += 0.30
    for i, r in enumerate(am.OFF):
        if i:
            P.rule(y + 0.02); y += 0.26
        top = y
        hs = start_cell(P, r, 0, top, cell, fs['credit'])
        P.image(r['picture'], xs[1], top + cell, cell, cell)
        P.image(r['nearest_image'], xs[2], top + cell, cell, cell)
        hn = cell + P.block(am.off_nearest(r), xs[2], top + cell + 0.03, cell, fs['credit'])
        hf = P.block(v4.facts(r), xs[3], top - 0.02, fw, fs['credit'] - 0.3, MUTE)
        y = top + max(hs, cell, hn, hf) + 0.30
    return P, y - 0.12


def mob():
    W, gap = 3.9, 0.14
    half = (W - gap) / 2
    P, fs = am.MPlate(W), FS['mob']
    y = 0.0
    prev = None
    for r in am.OFF:
        if prev is not None and r['start'] != prev:
            P.rule(y + 0.02); y += 0.30
        P.text(0, y + 0.15, 'painted by the search', fs['head'], MUTE)
        top = y + 0.24
        P.image(r['picture'], 0, top + W, W, W)
        y = top + W + 0.03 + P.block(v4.facts(r), 0, top + W + 0.03, W, fs['found'] - 0.5, MUTE) + 0.22
        P.text(0, y + 0.15, 'starting work', fs['head'], MUTE)
        P.text(half + gap, y + 0.15, 'nearest real work', fs['head'], MUTE)
        top = y + 0.24
        h1 = start_cell(P, r, 0, top, half, fs['credit'])
        P.image(r['nearest_image'], half + gap, top + half, half, half)
        h2 = half + 0.03 + P.block(am.off_nearest(r), half + gap, top + half + 0.03, half, fs['credit'])
        y = top + max(h1, h2) + 0.42
        prev = r['start']
    return P, y - 0.2


if __name__ == '__main__':
    for nm, fn in (('search_finds_off1900_v5', desk), ('search_finds_off1900_v5-mobile', mob)):
        assert os.environ.get('OFF5_ITER') or not (HERE / f'{nm}.svg').exists(), nm
        P, H = fn(); make(nm, P, H)
