"""The record in numbers: rows kept, named and dated works, artists, styles, genres, works per decade.
Usage: counts.py OUT_JSON"""
import json
import sys

import numpy as np

from data import all_ids, ids, labels, year

named = (labels['artist'] != '') & (year >= 1000) & (year <= 2026)      # the rule used throughout: an artist and a year from 1000 to 2026
dec = (year[named] // 10 * 10).astype(int)
out = {'rows': int(len(all_ids)), 'kept': int(len(ids)), 'dropped': int(len(all_ids) - len(ids)), 'named_dated': int(named.sum()),
       'year_parsed': int((~np.isnan(year)).sum()), 'artist_given': int((labels['artist'] != '').sum()),
       **{f'{k}s_named': int(len(np.unique(v[named & (v != '')]))) for k, v in labels.items()},     # distinct non-empty labels among named dated works
       **{f'{k}s_all': int(len(np.unique(v[v != '']))) for k, v in labels.items()},
       'no_style_named': int((named & (labels['style'] == '')).sum()), 'no_genre_named': int((named & (labels['genre'] == '')).sum()),
       'by_decade': {int(d): int(c) for d, c in zip(*np.unique(dec, return_counts=True))},
       **{f'top_{k}s_named': [[s, int(c)] for s, c in sorted(zip(*np.unique(v[named & (v != '')], return_counts=True)), key=lambda x: -x[1])[:15]]
          for k, v in labels.items()},
       'rule': 'style and genre are the first tag before |; artist is the raw name; decades are year // 10, all of them'}
assert sum(out['by_decade'].values()) == out['named_dated'] and len(np.unique(all_ids)) == len(all_ids)
with open(sys.argv[1], 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'by_decade'}))
