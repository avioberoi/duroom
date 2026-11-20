"""Distinct hits of the search and its two controls per starting work and in total, from the evaluator table sphere/eval/bound/conf_table.out (controls_evaluate.py table), to figures/data/q10_confirm.json.
Usage: python derive_q10_confirm.py"""
import json
import os
import re
from pathlib import Path

OUT = Path(os.environ['DUROOM_OUT'])
HERE = OUT / 'figures' / 'data'
HERE.mkdir(parents=True, exist_ok=True)
TABLE = OUT / 'sphere' / 'eval' / 'bound' / 'conf_table.out'
STARTS = {6: 'Cossiers, 1641', 7: 'Fra Angelico, 1419', 8: 'Sisley, 1885', 9: 'Van Gogh, 1890', 10: 'Degas, 1890'}   # the 5 starting works by seed
ROLES = ('V', 'C1', 'C2')

hits = {}
lines = open(TABLE, encoding='utf-8').read().splitlines()
for i, line in enumerate(lines):
    m = re.match(r'RUN \S+: .*tag \w+_s(\d+) \(role (V|C1|C2)\);', line)
    if m:
        exact = re.search(r'evaluator-admitted \d+ DF \d+ DH (\d+) ', lines[i + 1])
        hits[int(m[1]), m[2]] = int(exact[1])
starts = [{'label': label, 'distinct_hits': {r: hits[s, r] for r in ROLES}} for s, label in STARTS.items()]
totals = {r: sum(x['distinct_hits'][r] for x in starts) for r in ROLES}
out = {'arms': {'V': 'the search', 'C1': "random scores with the search's step sizes", 'C2': 'round trips with no search'},
       'starts': starts, 'totals': {'distinct_hits': totals}}
(HERE / 'q10_confirm.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print('q10_confirm.json: totals', totals)
