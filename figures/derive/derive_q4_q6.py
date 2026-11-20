"""Display data for the three-far figures and the straight-route figure: figures/data/q5_far.json from distance/<encoder>/distance.json and figures/data/q5_walk.json from corner/corner.json.
Usage: python derive_q4_q6.py"""
import json
import os
from pathlib import Path
import numpy as np
from _artwork_export import card

OUT=Path(os.environ['DUROOM_OUT'])
HERE=OUT/'figures'/'data'
HERE.mkdir(parents=True,exist_ok=True)
IMAGES=Path(os.environ['WIKIART_DIR'])/'artwork_images'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def write(name,data):
    text=json.dumps(data,ensure_ascii=False,indent=1)+'\n'
    assert '"id":' not in text and '"ids":' not in text
    (HERE/f'{name}.json').write_text(text,encoding='utf-8')
ENCODERS=[('CLIP','clip-vit-l-14'),('DINOv3','dinov3-l'),('SigLIP','siglip-so400m')]
distance={label:read(OUT/'distance'/slug/'distance.json') for label,slug in ENCODERS}

far=[]
for label,slug in ENCODERS:
    d=distance[label]
    era_order=['before_1900','1900_1959','1960_on']
    metrics_keys=[('d10_all','Absolute'),('local_all','Local'),('centre','Average direction')]
    era_rows=[]
    for key,title in metrics_keys:
        counts=[d['top1pct_by_era'][key][era] for era in era_order];total=sum(counts)
        assert total==d['n']//100
        era_rows.append({'key':key,'label':title,'counts':counts,'shares':[n/total for n in counts],'total':total})
    counts=[d['era_sizes'][era] for era in era_order]
    assert sum(counts)==d['n']
    era_rows.append({'key':'corpus','label':'Whole corpus','counts':counts,'shares':[n/d['n'] for n in counts],'total':d['n']})
    examples=[]
    for key,title,score in [('top_d10_not_local','Absolute, not local','d10_all'),('top_local_not_d10','Local, not absolute','local_all'),('top_centre_not_local','Centre, not local','centre')]:
        r=d['cards'][key][0]
        examples.append(card(r,IMAGES,selection=title,score_key=score,score=r[score],absolute=r['d10_all'],local=r['local_all'],centre=r['centre']))
    far.append({'encoder':label,'eras':era_rows,'examples':examples,
        'overlap':[{'pair':name,'share':d['top1pct_overlap'][key]} for name,key in [('Absolute / local','d10_all_local_all'),('Absolute / centre','d10_all_centre'),('Local / centre','local_all_centre')]]})
write('q5_far',{'encoders':far,'era_labels':['Before 1900','1900–1959','1960 onward'],'k':distance['CLIP']['k'],
    'definition':'Absolute: distance to the tenth neighbour. Local: absolute distance divided by the neighbours\u2019 median absolute distance. Centre: distance to the unit average direction, not to the plain mean inside the sphere.'})

walk=read(OUT/'corner'/'corner.json')
settings=[]
for cap in walk['caps']:
    key=f'cap{cap}_full';s=walk['settings'][key]
    routes={}
    for name in ['straight','path']:
        routes[name]=[{'progress':r['progress']['mean'],'count':r['within_R']['mean'],'sd':r['within_R']['sd'],
                       'label':r.get('label',''),'t':r.get('t')} for r in s[name]]
    path=sorted(routes['path'],key=lambda r:r['progress'])
    ratios=[{'progress':r['progress'],'ratio':r['count']/float(np.interp(r['progress'],[p['progress'] for p in path],[p['count'] for p in path]))} for r in routes['straight']]
    settings.append({'cap':cap,'reach':s['R'],**routes,'ratios':ratios})
write('q5_walk',{'cutoff':walk['cut'],'window':walk['window'],'seeds':walk['seeds'],'default_cap':20,'settings':settings,
    'ratio_definition':'Ratio of seed-mean counts, with path means linearly interpolated at the straight route\u2019s mean progress; not the mean of per-seed ratios.',
    'reach_definition':'Reach is the median distance to a tenth neighbour by another artist, estimated separately in each draw; counts within reach include all works.'})
print('Derived q5_far.json and q5_walk.json')
