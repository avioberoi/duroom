"""Which works do the three encoders agree are far? Per score (leftover share, 10th-neighbour distance, local score, centre distance): the share of the top 1%, 5% and 10% two encoders have in common against chance, and the works in all three top sets with era and label mix and cards.
Usage: agree.py OUT_DIR LEFTOVER_DIR DISTANCE_DIR   (the folders holding <lens>_k32_cap0/leftover.npz and <lens>/scores.npz)"""
import json
import os
import sys

import numpy as np

from data import card, ids, labels, year

OUT, LEFT, DIST = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
LENSES = ('clip-vit-l-14', 'dinov3-l', 'siglip-so400m')
artist, style, genre = labels['artist'], labels['style'], labels['genre']
usable = (artist != '') & (year >= 1000) & (year <= 2026)
rows = np.flatnonzero(usable)
scores = {}
for lens in LENSES:
    z = np.load(f'{LEFT}/{lens}_k32_cap0/leftover.npz')
    assert (z['ids'] == ids[rows]).all()
    scores[('leftover', lens)] = z['share']
    z = np.load(f'{DIST}/{lens}/scores.npz')
    assert (z['ids'] == ids).all()
    for s in ('d10_all', 'local_all', 'centre'):
        scores[(s, lens)] = z[s][rows]
for k, v in scores.items():
    assert np.isfinite(v).all(), k    # a NaN would sort last and an inf would enter a top set
result = {'n': int(len(rows)), 'lenses': LENSES, 'scores': {}}
for s in ('leftover', 'd10_all', 'local_all', 'centre'):
    out = {}
    for frac in (0.01, 0.05, 0.10):
        n = int(len(rows) * frac)
        top = {lens: set(np.argsort(-scores[(s, lens)])[:n]) for lens in LENSES}
        pairs = {f'{a}_{b}': len(top[a] & top[b]) / n for i, a in enumerate(LENSES) for b in LENSES[i + 1:]}
        allthree = top[LENSES[0]] & top[LENSES[1]] & top[LENSES[2]]
        out[f'top{int(frac * 100)}pct'] = {'pairs_shared': pairs, 'chance': frac, 'chance_all_three': frac ** 2, 'all_three': len(allthree), 'all_three_share': len(allthree) / n}
        if frac == 0.01:
            idx = np.array(sorted(allthree, key=lambda i: -sum(scores[(s, lens)][i] for lens in LENSES)), dtype=int)    # int even when empty
            eras = {era: int(sum(1 for i in idx if lo <= year[rows[i]] < hi)) for era, (lo, hi) in (('before_1900', (0, 1900)), ('1900_1959', (1900, 1960)), ('1960_on', (1960, 3000)))}
            lab = lambda v: sorted(((n2, int(c)) for n2, c in zip(*np.unique(v[rows[idx]][v[rows[idx]] != ''], return_counts=True))), key=lambda x: -x[1])[:6]    # noqa: E731
            cards, seen = [], {}
            for i in idx:
                a = artist[rows[i]]
                if seen.get(a, 0) < 2:
                    seen[a] = seen.get(a, 0) + 1
                    cards.append({**card(rows[i], OUT), **{lens: float(scores[(s, lens)][i]) for lens in LENSES}})
                if len(cards) == 16:
                    break
            out['all_three_top1pct'] = {'eras': eras, 'styles': lab(style), 'genres': lab(genre), 'cards': cards}
    rank = lambda a: np.argsort(np.argsort(a))    # noqa: E731
    out['spearman'] = {f'{a}_{b}': float(np.corrcoef(rank(scores[(s, a)]), rank(scores[(s, b)]))[0, 1]) for i, a in enumerate(LENSES) for b in LENSES[i + 1:]}
    result['scores'][s] = out
    print(s, 'top 1% shared by pair', {k: round(v, 2) for k, v in out['top1pct']['pairs_shared'].items()}, '| all three', out['top1pct']['all_three'], '(%.0f%%)' % (100 * out['top1pct']['all_three_share']), '| spearman', {k: round(v, 2) for k, v in out['spearman'].items()}, flush=True)
json.dump(result, open(f'{OUT}/agree.json', 'w'), indent=1)
print('done', flush=True)
