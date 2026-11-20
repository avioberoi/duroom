"""Median leftover share by decade under the whole-corpus basis and the basis from the works before 1900, plus four works with their share under each, from leftover/clip-vit-l-14_k32_cap0/leftover.json, to figures/data/q3_two_pasts.json.
Usage: python derive_q3_two_pasts.py"""
import json
import os
from pathlib import Path

from _artwork_export import card

OUT = Path(os.environ['DUROOM_OUT'])
HERE = OUT / 'figures' / 'data'
HERE.mkdir(parents=True, exist_ok=True)
COMMON = OUT / 'leftover' / 'clip-vit-l-14_k32_cap0' / 'leftover.json'
FIRST_DECADE = 1500     # earlier decades hold few works and sit outside the drawn range


def write(name, data):
    text = json.dumps(data, ensure_ascii=False, indent=1) + '\n'
    assert '"id":' not in text and '"ids":' not in text
    (HERE / f'{name}.json').write_text(text, encoding='utf-8')


d = json.loads(COMMON.read_text(encoding='utf-8'))
assert d['k'] == 32 and d['cap'] == 0

decades = [{'decade': a, 'corpus': b, 'past': c, 'n': n} for a, b, c, n in d['by_decade'] if a >= FIRST_DECADE]
assert all(r['n'] >= 50 for r in decades)
by = {r['decade']: r for r in decades}

rows = {}
for lst in d['cards'].values():
    for r in lst:
        rows[(r['artist'], r['year'], r['title'])] = r
WANT = [  # (artist slug, year, title in the source, title as printed, short label)
    ('alexander-calder', 1963, 'POLYGONS ON TRIANGLES', 'Polygons on Triangles', 'Calder'),
    ('ellsworth-kelly', 2000, 'Red Yellow Blue', 'Red Yellow Blue', 'Kelly'),
    ('ayse-erkmen', 2009, 'Bluish', 'Bluish', 'Erkmen'),
    ('richard-hamilton', 1971, 'A Beautiful Little Shit for Dieter', 'A Beautiful Little Shit for Dieter', 'Hamilton'),
]
works = []
for artist, year, title, shown, short in WANT:
    r = rows[(artist, year, title)]
    c = card(r, COMMON.parent / 'thumbs', share=r['share'], share_past=r['share_past'])
    c['title'] = shown
    c['short'] = short
    works.append(c)

out = {
    'encoder': 'CLIP', 'k': d['k'], 'n': d['n'], 'min_works_per_decade': 50,
    'cutoff_year': 1900,
    'definition': 'Median leftover share of the works of each decade, measured against the average and 32 directions of '
                  'the whole corpus (corpus) and of the works before 1900 only (past). Before 1900 the past basis is '
                  'fit on those same works. Each stem joins a work\'s share under the corpus basis to its share under the past basis.',
    'decades': decades,
    'works': works,
}
write('q3_two_pasts', out)

for w in works:
    print(f"{w['short']:9s} {w['year']} {w['share']:.3f} -> {w['share_past']:.3f}  {w['asset']}")
print(len(decades), 'decades', by[1960])
