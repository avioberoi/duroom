"""Main-text gallery: one drawn row per start, 4 columns (start, painted, later work, fact block beside); the phone layout is search_finds_v1's.
Usage: imported by picture_facts_figs_v2.py"""
import picture_facts_figs_v1 as pf          # installs the fact lines on search_finds_v1.credit_runs
import search_finds_v1 as sf
from _style import *

DRAWN = ['057', 'p_s8_1219', 'p_s9_1784', '045']     # 1 per start, drawn at random (seed 20261119)
ROWS = [r for p in DRAWN for r in sf.A_ROWS if pf.pid_of(r['found']) == p]
assert [pf.pid_of(r['found']) for r in ROWS] == DRAWN
HEADS = ('starting work, before 1900', 'painted by the search', 'real work dated 1900 or later', 'round, and the nearest real works')


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
    for i, r in enumerate(ROWS):
        if i:
            P.rule(y + 0.02); y += 0.26
        top = y
        for x, role in zip(xs[:3], ('start', 'found', 'later')):
            P.image(r[role], x, top + cell, cell, cell)
        hs = [P.credits(r, role, x, top + cell + 0.03, cell, fs['credit']) for x, role in zip((xs[0], xs[2]), ('start', 'later'))]
        hf = P.credits(r, 'found', xs[3], top - 0.02, fw, fs['credit'] - 0.3)     # the fact block, level with the pictures
        y = max(top + cell + 0.03 + max(hs), top + hf) + 0.30
    return P, y - 0.12
