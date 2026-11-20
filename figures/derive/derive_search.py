"""Figure data for the direction-response figure (curves from the decoder walk) and the retrodiction figures, to figures/data/search_walk.json and search_retro.json.
Usage: python derive_search.py"""
import json
import os
from pathlib import Path
import numpy as np

OUT=Path(os.environ['DUROOM_OUT'])
HERE=OUT/'figures'/'data'
HERE.mkdir(parents=True,exist_ok=True)
RUN=OUT/'walk'

def read(name): return json.loads((RUN/name).read_text(encoding='utf-8'))
def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [clean(v) for v in value]
    if isinstance(value,(np.floating,float)): return round(float(value),6) if np.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,np.bool_): return bool(value)
    return value
def write(name,value):
    (HERE/(name+'.json')).write_text(json.dumps(clean(value),ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')

def walk():
    meta=read('meta.json'); scores=read('scores.json')['frames']
    dirs={'shift':'After-1900 shift','axis1':'New axis 1','axis2':'New axis 2',
          'past_rank1':'Past basis: direction 1','past_rank50':'Past basis: direction 50',
          'past_rank300':'Past basis: direction 300','random1':'Random direction 1',
          'random2':'Random direction 2','to_neighbour15':'Toward the 15th neighbour'}
    curves=[]
    for key,label in dirs.items():
        vals=[]
        for angle in meta['angles']:
            rows=[f for f in scores if f['kind']=='walk' and f['dir']==key and f['angle']==angle and f['seed']==0]
            if rows: vals.append({'angle':angle,'achieved':float(np.median([f['angle_achieved_rel'] for f in rows])),
                'chord':float(np.median([f['d_request_clip'] for f in rows])),'n':len(rows)})
        curves.append({'key':key,'label':label,'points':vals})
    write('search_walk',{'curves':curves,'angles':meta['angles'],
        'note':'Decoded images; zero degrees is a repainting. Movement along the request is relative to this start repainting, not total image distance.'})


def retro():
    cuts=[]
    for cutoff in (1800,1850,1900,1950):
        d=json.loads((OUT/'retro'/f'clip-vit-l-14_cut{cutoff}'/'retro.json').read_text(encoding='utf-8'))
        rows=[{'decade':int(year),'n':v['n'],'q99':v.get('abs_others_above_q99'),
            'q95':v.get('abs_others_above_q95'),'auc':v.get('abs_others_auc_vs_past')}
            for year,v in d['by_decade'].items() if int(year)>=cutoff]
        genres=[{'label':k,'n':v['n'],'share':v['abs_others_above_q99']} for k,v in d['later_by_genre'].items() if v['n']>=200]
        cuts.append({'cutoff':cutoff,'n_past':d['n_past'],'n_later':d['n_later'],
            'bar99':d['bars']['abs_others']['99'],'bar95':d['bars']['abs_others']['95'],
            'decades':sorted(rows,key=lambda x:x['decade']),'genres':sorted(genres,key=lambda x:-x['share'])})
    write('search_retro',{'cuts':cuts,'encoder':'CLIP','note':'Stored artwork vectors, artist-excluded neighbours, full-size reference. These are not the decoder search bars.'})

if __name__=='__main__':
    walk();retro()
    print('Derived the walk curves and the retrodiction tables')
