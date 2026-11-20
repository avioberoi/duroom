"""Figure data for the CLIP pass of one public-domain print: patch norms per block, end vector, angle to the corpus mean, from clipviz/clipviz.json (data/clipviz.py) and measured_sphere.json, to figures/data/image_to_vector.json.
Usage: python derive_image_to_vector.py"""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

OUT=Path(os.environ['DUROOM_OUT'])
HERE=OUT/'figures'/'data'
HERE.mkdir(parents=True,exist_ok=True)
raw=json.loads((OUT/'clipviz'/'clipviz.json').read_text(encoding='utf-8'))
assert raw['gate']['median']>=0.999 and raw['gate']['min']>=0.99
unit=np.array(raw['end_vector_unit'])
assert np.isclose(np.linalg.norm(unit),1,atol=2e-5)
length=raw['end_vector_length']
crop=OUT/'clipviz'/raw['crop_asset']
size=Image.open(crop).size
assert size==(224,224)
stages=[]
for name,block in raw['blocks'].items():
    values=np.array(block['patch_norms_16x16'])
    assert values.shape==(16,16) and np.all(values>0)
    stages.append({'block':int(name.split('_')[-1]),'norms':values.tolist(),
                   'min':float(values.min()),'max':float(values.max()),
                   'class_norm':block['class_token_norm']})
angle=np.deg2rad(raw['angle_to_corpus_mean_direction_deg'])
sphere=json.loads((HERE/'measured_sphere.json').read_text())
out={
    'kind':'Measured encoding of the public-domain Commons reproduction of The Great Wave.',
    'provenance':{'model':raw['model'],'gate':raw['gate'],'stages':raw['stages'],
                  'notes':raw['notes'],'preprocess':raw['preprocess']},
    'lens':'CLIP','dimension':len(unit),
    'crop_asset':f'../clipviz/{raw["crop_asset"]}','crop_size':list(size),
    'source_size':raw['image_size'],'patch_grid':16,'patch_size':14,'token_dimensions':1024,
    'blocks':stages,'unit_values':unit.tolist(),'raw_values_rounded':(unit*length).tolist(),
    'length':length,'sample_indices':[0,1,2],
    'angle_deg':raw['angle_to_corpus_mean_direction_deg'],
    'angle_percentile':raw['angle_percentile_in_corpus'],
    'sphere_point':[float(np.cos(angle)),float(np.sin(angle))],
    'corpus_mean_length':sphere['mean_norm'],
}
(HERE/'image_to_vector.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(f'Measured encoder: {len(stages)} captured blocks, {len(unit)} coordinates; corpus gate passed')
