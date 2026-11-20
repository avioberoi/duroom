"""Data of the ways-of-asking figures and the dated-split figure: far and faithful pictures by way of asking (ways_of_asking.csv) and by the dates of the real works within the manifold line (dated_split.csv), from the readouts of 07 and the grid.
Usage: python derive_ways_dated.py"""
import csv
import os
import re
from pathlib import Path

OUT = Path(os.environ['DUROOM_OUT'])
HERE = OUT / 'figures' / 'data'
HERE.mkdir(parents=True, exist_ok=True)
T12 = OUT / 'tests12'
NOSEARCH = T12 / 'nosearch' / 'nosearch_v5_rows.csv'
ORACLE = T12 / 'nosearch' / 'nosearch_oracle_v1_rows.csv'
GRID = T12 / 'grid' / 'grid_v1.txt'
DATED = T12 / 'onoff' / 'out' / 'onoff_dated_v2.txt'
WALK = T12 / 'onoff' / 'out' / 'onoff_walk_v2_rtx.txt'


def pooled(path):
    """title of each table of the walk readout -> pooled counts of the columns W (single walk), V (search), C1, C2"""
    lines = open(path, encoding='utf-8').read().splitlines()
    out = {}
    for i, line in enumerate(lines):
        if re.fullmatch(r'\s+W\s+V\s+C1\s+C2', line):
            end = next(j for j in range(i + 1, len(lines)) if lines[j].strip().startswith('pooled'))
            out[lines[i - 1].strip()] = [int(v.strip('*')) for v in lines[end].split()[1:]]
    return out


def grid_counts(path):
    """group of the grid table -> (n, far and faithful, far and not faithful) at the 99th-percentile level and the faithful bar"""
    out = {}
    for line in open(path, encoding='utf-8'):
        parts = line.rstrip('\n').split('|')
        if len(parts) == 7 and parts[0][:1].isdigit():
            c = [int(v) for v in parts[3].split()]
            out[parts[0][:92].strip()] = (int(parts[0][92:]), c[0], c[1])
    return out


def far_and_faithful(rows):
    far = [r for r in rows if r['alone_far'] == '1']
    return len(rows), len(far), sum(r['faithful'] == '1' for r in far)


ns = list(csv.DictReader(open(NOSEARCH, encoding='utf-8')))
by_group = lambda g: [r for r in ns if r['group'] == g]
by_family = lambda f, s: [r for r in ns if (r['family'], r['sub']) == (f, s)]
oracle = [r for r in csv.DictReader(open(ORACLE, encoding='utf-8')) if r['target'] != '-1']
walk, grid = pooled(WALK), grid_counts(GRID)
FAR, FF = 'far from P alone (run-file readings), among all 1,920', 'FF = far and exact-faithful (exact gap <= FID)'

ways = [
    ('midpoints of two works', 'recipe', *far_and_faithful(by_group('midpoint of two works (48)'))),
    ("between two regions' centres", 'recipe', *far_and_faithful(by_group("line between two regions' centres (36)"))),
    ('35-degree steps from a work', 'recipe', *far_and_faithful(by_group('35 deg step past a work (480)'))),
    ('lines past a work', 'recipe', *far_and_faithful(by_group('lines past a real work or toward the sparse edge (450)'))),
    ('random requests inside the region', 'recipe', *far_and_faithful(by_family('prior_h200', 'cone'))),
    ('random requests on the whole sphere', 'recipe', *far_and_faithful(by_family('prior_h200', 'sphere'))),
]
ways.append(('all six recipes', 'recipe_total', *[sum(w[i] for w in ways) for i in (2, 3, 4)]))
ways.append(('lines aimed at real later works', 'ceiling', *far_and_faithful(oracle)))
ways.append(('single walk with selection', 'baseline', 9600, walk[FAR][0], walk[FF][0]))
for way, col in (('no-step loop', 3), ('random-score loop', 2), ('search', 1)):
    ways.append((way, 'loop', 9600, walk[FAR][col], walk[FF][col]))
for way, group in (('real works before 1900 held out (repainted)', '1 P2 (before 1900)'),
                   ('real works of 1900 to 1969 (repainted)', '1 pos (1900-1969)'),
                   ('real works from 1970 on (repainted)', '1 today, 1970 on')):
    n, ffc, fnc = grid[group]
    ways.append((way, 'real', n, ffc + fnc, ffc))
with open(HERE / 'ways_of_asking.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['way', 'kind', 'n', 'far', 'far_and_faithful'])
    w.writerows(ways)

# dated split, each work dated by its own year: the loops from the dated readout, the single walk from the walk readout
dated = {}
for line in open(DATED, encoding='utf-8'):
    m = re.match(r'\s+(V|C1|C2) pooled FF( and LL)?\s+n\s+(\d+) \|.*B one reader PRE/SINCE/UND/NONE (\d+)/(\d+)/(\d+)/(\d+)', line)
    if m:
        dated[m[1], 'large_leftover' if m[2] else 'all'] = [int(m[3])] + [int(m[i]) for i in (4, 5, 6, 7)]
rows = []
for subset in ('all', 'large_leftover'):
    for loop, key in (('search', 'V'), ('random-score loop', 'C1'), ('no-step loop', 'C2')):
        rows.append(['own', subset, loop] + dated[key, subset])
rows.append(['own', 'all', 'single walk', walk[FF][0]] + [walk[t][0] for t in (
    'among FF, one reader, nothing masked: PRE-NEAR (a work dated before 1900 near)', 'among FF, one reader: SINCE',
    'among FF, one reader: UNDATED-ONLY', 'among FF, one reader: NONE (off)')])
rows.append(['own', 'large_leftover', 'single walk', walk['FF and LL (large leftover)'][0], '',
             walk['LEAD ROW: FF and SINCE (one reader) and LL'][0], '', ''])
with open(HERE / 'dated_split.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['dating', 'subset', 'loop', 'far_and_faithful', 'pre1900_near', 'later_only_near', 'undated_only', 'none'])
    w.writerows(rows)
print('ways_of_asking.csv', len(ways), 'rows; dated_split.csv', len(rows), 'rows')
