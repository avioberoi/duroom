"""Sphere sample, polar-angle histogram and percentiles from figdata/figdata.json (data/figdata.py), to figures/data/measured_sphere.json.
Usage: python derive_measured_visuals.py"""
import json
import os
from pathlib import Path
import numpy as np

OUT=Path(os.environ['DUROOM_OUT'])
HERE=OUT/'figures'/'data'
HERE.mkdir(parents=True,exist_ok=True)
d=json.loads((OUT/'figdata'/'figdata.json').read_text(encoding='utf-8'))
def write(name, data):
    (HERE/f'{name}.json').write_text(json.dumps(data,separators=(',',':'))+'\n',encoding='utf-8')

s=d['sphere']
angles=np.array(s['sample_angle_deg'])
assert len(angles)==len(s['sample_xy'])==s['n_sample']
edges=np.arange(0,182,2)
counts,_=np.histogram(angles,edges)
assert counts.sum()==s['n_sample']
write('measured_sphere',{
    'provenance':d['provenance'],'population':s['n_population'],'sample':s['n_sample'],
    'hist_edges':edges.tolist(),'hist_counts':counts.tolist(),'xy':s['sample_xy'],
    'angles':s['sample_angle_deg'],
    'percentiles':s['angle_percentiles_population'], 'mean_norm':s['mean_vector_norm'],
    'variance_share':sum(s['variance_share_dir1_dir2']),
})
print('Derived measured sphere; counts checked')
