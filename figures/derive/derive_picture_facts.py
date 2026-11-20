"""Fact record of each painted picture of the galleries (round, nearest works per observer, label), from the pf_facts outputs tests12/pf/pf_facts_v1.json and pf_facts_v2.json, to figures/data/picture_facts_v1.json.
Usage: python derive_picture_facts.py"""
import json
import os
import re
from pathlib import Path

OUT = Path(os.environ['DUROOM_OUT'])
HERE = OUT / 'figures' / 'data'
HERE.mkdir(parents=True, exist_ok=True)
PF = OUT / 'tests12' / 'pf'
OBSERVERS = {'dinov3-l': 'DINOv3', 'siglip-so400m': 'SigLIP'}


def name(slug):
    return ' '.join(w.capitalize() for w in slug.split('-'))


def work(w):
    return {'artist': name(w['artist']), 'title': w['title'], 'year': w['year'], 'chord': round(w['chord'], 6)}


records = {r['pid']: r for r in json.load(open(PF / 'pf_facts_v1.json', encoding='utf-8'))}
records.update({r['pid']: r for r in json.load(open(PF / 'pf_facts_v2.json', encoding='utf-8'))})   # the off-the-manifold pictures, read again with their shown works as corpus rows
pictures = {}
for pid, r in records.items():
    obs = {}
    for key, label in OBSERVERS.items():
        x = r[key]
        obs[label] = {'line': x['line'], 'shown_chord': round(x['shown_chord'], 6), 'shown_within': x['shown_within'],
                      'nearest': work(x['nearest']), 'nearest_is_shown': x['nearest_is_shown'], 'pre_within': x['nearest_pre']['within']}
    pictures[pid] = {'round': int(re.search(r'iter(\d+)_', r['image'])[1]), 'label': r['label'],
                     'key_observer': OBSERVERS[r['key_observer']], 'obs': obs}
json.dump({'pictures': pictures}, open(HERE / 'picture_facts_v1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('picture_facts_v1.json:', len(pictures), 'pictures')
