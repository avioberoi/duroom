"""Bands of a tall gallery PDF: cuts on the thin rules between rows.
Usage: imported by print_split_v4.py"""
import os
from pathlib import Path

import fitz

import print_split_v2 as v2

HERE = Path(os.environ['DUROOM_OUT']) / 'figures'


def bands_of(name):
    src = HERE / f'{name}.pdf'
    h = fitz.open(src)[0].rect.height
    ys = v2.rules_y(src)
    cuts, top = [], 0.0
    while h - top > v2.MAXH:
        ok = [y for y in ys if top + 100 < y <= top + v2.MAXH]
        assert ok, f'{name}: no rule between {top:.0f} and {top + v2.MAXH:.0f} pt'
        cuts.append(ok[-1]); top = ok[-1]
    return h, list(zip([0.0] + cuts, cuts + [h]))
