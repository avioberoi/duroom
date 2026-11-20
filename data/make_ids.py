"""Builds smooth_ids.csv (role,id,partner) for smooth.py: the 512 later works of pos_emb.npz by role (passer, unfaithful, nonpasser), the 3 search starts and the 9 oracle (target, start) pairs.
Usage: make_ids.py OUT_CSV   (e.g. $DUROOM_OUT/inputs/smooth_ids.csv; reads oracle_targets.json, pos_emb.npz and the pools in $DUROOM_OUT)"""
import json
import os
import sys

import numpy as np

B = os.environ['DUROOM_OUT']
O = f'{B}/sphere'
o = json.load(open(f'{B}/inputs/oracle_targets.json'))
z = np.load(f'{O}/pos_emb.npz', allow_pickle=True)
s = np.load(f'{O}/pools/starts_pre1900.npz', allow_pickle=True)
pool_ids = np.load(f'{O}/pools/pos_1900_1969_cap3.npz', allow_pickle=True)['ids']
pos = np.array([int(p.split('push_')[1].split('.')[0]) for p in z['paths']])
passers = list(o['targets_pos_index'])
unfaith = [21, 276]
non = [i for i in range(len(pos)) if i not in passers and i not in unfaith]
start_id = {st['start_idx']: int(s['ids'][np.flatnonzero(s['rows_full'] == st['start_idx'])[0]]) for st in o['starts']}
rows = [('passer', pos[i], '') for i in passers] + [('unfaithful', pos[i], '') for i in unfaith] + [('nonpasser', pos[i], '') for i in non]
rows += [('start', v, '') for v in start_id.values()]
rows += [('oracle', pos[t], start_id[st['start_idx']]) for st in o['starts'] for t in st['target_idx_in_pos']]
n = {r: sum(x[0] == r for x in rows) for r in ('passer', 'unfaithful', 'nonpasser', 'start', 'oracle')}
assert len(set(pos)) == 512
assert set(int(i) for i in pool_ids) >= set(int(i) for i in pos)
with open(sys.argv[1], 'w', newline='\n') as f:
    f.write('role,id,partner\n' + ''.join(f'{r},{int(i)},{p}\n' for r, i, p in rows))
print(n)
