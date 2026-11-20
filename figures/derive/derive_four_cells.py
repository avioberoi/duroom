"""Data of the four-cells figure: gap and far scores of every kind of picture (grid_v1_rows.csv) and the four-cell counts of every group (grid_v1.txt), to figures/data/four_cells.json.
Usage: python derive_four_cells.py"""
import csv
import json
import os
import sys
from pathlib import Path

OUT = Path(os.environ['DUROOM_OUT'])
HERE = OUT / 'figures' / 'data'
HERE.mkdir(parents=True, exist_ok=True)
GRID = OUT / 'tests12' / 'grid'
sys.path.insert(0, os.environ['DUROOM_REPO'] + '/measure')
from local_rescore import BARS, FID

KEYS = [  # key, groups of grid_v1 in the order drawn
    ('p2', ['1 P2 (before 1900)']),
    ('pos', ['1 pos (1900-1969)']),
    ('today1970', ['1 today, 1970 on']),
    ('rand_cone', ['2 random requests, cone']),
    ('rand_sphere', ['2 random requests, sphere']),
    ('noise', [f'3 noise requests, kind {k}' for k in 'abcde']),
    ('reanchor', ['4 V, all candidates (5 starts)']),
    ('free', [f'5 free run main_s{s}' for s in (1, 2, 3)]),
]
COUNTED = {'noise': ['3 noise requests, all kinds']}   # groups the table line of which is not one of the groups above

rows = {}
for r in csv.DictReader(open(GRID / 'grid_v1_free_rows.csv', encoding='utf-8')):
    rows.setdefault(r['group'], []).append(r)

counts = {}
for line in open(GRID / 'grid_v1_free.txt', encoding='utf-8'):
    parts = line.rstrip('\n').split('|')
    if len(parts) == 7 and parts[0][:1].isdigit():
        counts[parts[0][:92].strip()] = (int(parts[0][92:]), [int(v) for v in parts[3].split()])

points, table, sampled = {}, {}, {}
for key, groups in KEYS:
    pts = [[float(r['gap']), float(r['d_dino']), float(r['d_siglip'])] for g in groups for r in rows[g]]
    lines = [counts[g] for g in COUNTED.get(key, groups)]
    points[key] = pts
    table[key] = {'n': sum(n for n, _ in lines), 'FF_FN_NF_NN': [sum(c[i] for _, c in lines) for i in range(4)]}
    sampled[key] = any(r['sampled'] == '1' for g in groups for r in rows[g])
json.dump({'columns': ['gap', 'd_dino', 'd_siglip'], 'faithful_bar_gap': FID,
           'far_levels_p99': {'dinov3': BARS[0], 'siglip': BARS[1]},
           'table': table, 'sampled': sampled, 'points': points},
          open(HERE / 'four_cells.json', 'w'))
print('four_cells.json:', {k: len(v) for k, v in points.items()})
