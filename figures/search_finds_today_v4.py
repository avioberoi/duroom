"""Gallery of the 5 pictures of the run against today's corpus: start, painted, nearest real work, with the fact block (and a reader-supplied line) beside.
Usage: python search_finds_today_v4.py"""
import os

import picture_facts_figs_v2 as pf2          # installs the fact lines and the reader-supplied lines on search_finds_v1.credit_runs
import search_finds_v1 as sf
from _style import *

ROWS = sf.C_ROWS
HEADS = ('starting work', 'painted by the search', 'nearest real work, any date', 'round, and the nearest real works')


def desk():
    W, gap = 8.8, 0.2
    cell = 1.95
    xs = [0, cell + gap, 2 * (cell + gap), 3 * (cell + gap)]
    fw = W - xs[3]
    P, fs = sf.Plate(W), sf.FS['desk']
    y = 0.0
    for x, h in zip(xs, HEADS):
        P.text(x, y + 0.16, h, fs['head'], MUTE)
    y += 0.30
    prev = None
    for i, r in enumerate(ROWS):
        if i:
            P.rule(y + 0.02); y += 0.26
        top = y
        for x, role in zip(xs[:3], ('start', 'found', 'nearest')):
            P.image(r[role], x, top + cell, cell, cell)
        hs = [P.credits(r, role, x, top + cell + 0.03, cell, fs['credit']) for x, role in zip((xs[0], xs[2]), ('start', 'nearest'))]
        hf = P.credits(r, 'found', xs[3], top - 0.02, fw, fs['credit'] - 0.3)     # the fact block, level with the pictures
        y = max(top + cell + 0.03 + max(hs), top + hf) + 0.30
    return P, y - 0.12


if __name__ == '__main__':
    for nm, fn in (('search_finds_today_v4', desk), ('search_finds_today_v4-mobile', lambda: sf.mob_rows(ROWS, 'nearest', sf.HEADS['C']))):
        assert os.environ.get('TODAY4_ITER') or not (HERE / f'{nm}.svg').exists(), nm
        P, H = fn(); sf.make(nm, P, H)
