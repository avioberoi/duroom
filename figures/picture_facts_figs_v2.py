"""Adds a reader-supplied line per picture (figures/marks/byeye_v1.json: {"lines": {picture id: {"line": text, "shown_work": true if the line is about the real work shown}}}) as the last line of each fact block; without that file no line is added.
Usage: python picture_facts_figs_v2.py  (writes search_finds_main_v2)"""
import json
import os

import picture_facts_figs_v1 as pf
import search_finds_v1 as sf
from _style import *

BYEYE = HERE / 'marks' / 'byeye_v1.json'
if BYEYE.exists():
    BY = json.load(open(BYEYE, encoding='utf-8'))['lines']
else:
    print(f'no extra lines: {BYEYE} not supplied')
    BY = {}
_facts = pf.fact_blocks
USED = []


def fact_blocks(pid, large_leftover=False):
    b = _facts(pid, large_leftover)
    e = BY.get(pid)
    if e and e['shown_work']:
        USED.append(pid)
        b = b + [[(f'By eye: “{e["line"]}”', False)]]
    return b


pf.fact_blocks = fact_blocks   # pf.credit_runs (installed on search_finds_v1) calls it at draw time


def check(name):
    assert os.environ.get('BYEYE_ITER') or not (HERE / f'{name}.svg').exists(), name
    return name


if __name__ == '__main__':
    import search_finds_main_v1 as sm
    P, H = sm.desk(); sf.make(check('search_finds_main_v2'), P, H)
    P, H = sf.mob_rows(sm.ROWS, 'later', sf.HEADS['A']); sf.make(check('search_finds_main_v2-mobile'), P, H)
    print('extra lines drawn for', sorted(set(USED)))
