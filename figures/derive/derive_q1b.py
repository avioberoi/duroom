"""Aggregate decade coordinates and histograms (decade means in the first two new directions) from newaxis_check/k32/newaxis.json, to figures/data/q1b.json.
Usage: python derive_q1b.py"""
import json
import os
from pathlib import Path
import numpy as np

OUT = Path(os.environ['DUROOM_OUT'])
HERE = OUT / 'figures' / 'data'
HERE.mkdir(parents=True, exist_ok=True)
SOURCE = OUT / 'newaxis_check' / 'k32' / 'newaxis.json'
raw = json.loads(SOURCE.read_text(encoding='utf-8'))
cutoff = 1900
c = raw['cutoffs'][str(cutoff)]
past = np.array([r for r in c['by_decade'] if r[0] < cutoff])
rms = float(np.sqrt(np.average(past[:,2]**2, weights=past[:,3])))
plane = [dict(zip(('decade','shift','axis1','axis2','n'),row)) for row in c['by_decade_plane']]
sizes = {r['decade']:r['n'] for r in plane}
distributions = {}
for axis in ('shift','axis1'):
    assert c['distribution_outside_bins'][axis] == [0,0]
    distributions[axis] = []
    for decade, quantiles, counts in c['by_decade_distribution'][axis]:
        if 1850 <= decade <= 2010:
            assert len(counts)+1 == len(c['distribution_bins'])
            assert sum(counts) == sizes[decade]
            edges = np.array(c['distribution_bins'])
            approximate_mean = np.dot(counts,(edges[:-1]+edges[1:])/2)/sizes[decade]
            exact_mean = next(r[axis] for r in plane if r['decade']==decade)
            assert abs(approximate_mean-exact_mean) <= np.diff(edges).max()/2
            distributions[axis].append({'decade':decade,'n':sizes[decade],
                                        'counts':counts})
    assert [r['decade'] for r in distributions[axis]] == list(range(1850,2020,10))

s1, s2 = dict(c['by_style']), dict(c['by_style_axis2'])
g1, g2 = dict(c['by_genre']), dict(c['by_genre_axis2'])
assert all(s1[n]>0 for n in ('Minimalism','Light and Space','Color Field Painting'))
assert g1['installation']>0
assert all(s2[n]>0 for n in ('Synthetic Cubism','Cubism','Futurism'))
assert g2['installation']<0
assert any(r['genre']=='portrait' and 1910<=r['year']<1920 for r in c['axis1_later']['low'])
assert any(r['genre']=='performance' for r in c['axis2_later']['low'])

out = {
    'lens':'CLIP','cutoff':cutoff,'k':raw['k'],'cap':raw['cap'],
    'direction_seed':c['runs'][0]['seed'],
    'population':'Decade summaries use all dated works, with at least 50 works per decade; cap applies to fitting directions.',
    'plane':plane,'label_decades':[1900,1930,1950,1970,2010],
    'past_axis1_rms':rms,'rms_n':int(past[:,3].sum()),'past_n':c['n_past'],
    'rms_method':'sqrt(sum(n_decade * rms_axis1_decade**2) / sum(n_decade)); available pre-cutoff decades',
    'distribution_bins':c['distribution_bins'],'distributions':distributions,
    'axis_labels':{
        'axis1_positive':'Minimal, spatial, installation','axis1_negative':'No clear kind',
        'axis2_positive':'Cubism, Futurism','axis2_negative':'Installation, performance'},
    'label_evidence':{
        'axis1_positive_styles':{n:s1[n] for n in ('Minimalism','Light and Space','Color Field Painting')},
        'axis2_positive_styles':{n:s2[n] for n in ('Synthetic Cubism','Cubism','Futurism')},
        'installation_axis1':g1['installation'],'installation_axis2':g2['installation'],
        'negative_examples':'1910s portraits occur among axis1_later.low; performance occurs among axis2_later.low. These are examples, not category-mean claims.'}
}
(HERE/'q1b.json').write_text(json.dumps(out,indent=1),encoding='utf-8')
print(f'q1b.json: {len(plane)} decade means; {len(distributions["shift"])} histograms per axis; past RMS {rms:.6f} from {out["rms_n"]}/{out["past_n"]} past works')
