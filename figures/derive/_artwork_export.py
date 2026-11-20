"""Export credited 320 px thumbnails of corpus works (read from $WIKIART_DIR/artwork_images) to $DUROOM_OUT/figures/assets/works.
Usage: imported by the derive scripts"""
import hashlib
import json
import re
import unicodedata
import os
from pathlib import Path
from PIL import Image, ImageOps

FIG=Path(os.environ['DUROOM_OUT'])/'figures'
HERE=FIG/'data'
ASSETS=FIG/'assets/works'
ASSETS.mkdir(parents=True,exist_ok=True)
MANIFEST=FIG/'artwork-sources.json'
MANIFEST.parent.mkdir(parents=True,exist_ok=True)
used=json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}
def register(x):
    if isinstance(x,dict):
        if 'asset' in x and 'source_sha256' in x:
            used[Path(x['asset']).stem]=x['source_sha256']
        for value in x.values():register(value)
    elif isinstance(x,list):
        for value in x:register(value)
for path in HERE.glob('*.json'):
    register(json.loads(path.read_text(encoding='utf-8')))
MANIFEST.write_text(json.dumps(used,indent=2),encoding='utf-8')

NAMES={'hans-holbein-the-younger':'Hans Holbein the Younger','tracy-168':'Tracy 168',
       'lee-quinones':'Lee Quiñones','ayse-erkmen':'Ayşe Erkmen','john-mccracken':'John McCracken',
       'piero-di-cosimo':'Piero di Cosimo','nicholas-of-verdun':'Nicholas of Verdun',
       'vincent-van-gogh':'Vincent van Gogh','dante-gabriel-rossetti':'Dante Gabriel Rossetti'}

LOCK=MANIFEST.with_suffix('.lock')

class _ManifestLock:
    """Exclusive lock file held while the manifest is re-read, a slug chosen and the manifest written."""
    def __enter__(self):
        import os, time
        for _ in range(600):
            try:
                self.fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY);return self
            except FileExistsError:
                time.sleep(0.1)
        raise TimeoutError(f'{LOCK} held for over a minute; remove it if no export is running')
    def __exit__(self,*exc):
        import os
        os.close(self.fd);os.remove(LOCK)

def card(row,folder,**values):
    source=folder/f'{row["id"]}.jpg'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    title=f'{row["artist"]}-{row.get("year") or "undated"}-{row["title"]}'
    ascii_title=unicodedata.normalize('NFKD',title).encode('ascii','ignore').decode().lower()
    base=re.sub('[^a-z0-9]+','-',ascii_title).strip('-')[:140].rstrip('-')
    with _ManifestLock():
        if MANIFEST.exists():  # another process may have added slugs since import
            used.update(json.loads(MANIFEST.read_text(encoding='utf-8')))
        slug=base;suffix=2
        while slug in used and used[slug]!=digest:
            slug=f'{base}-{suffix}';suffix+=1
        used[slug]=digest
        MANIFEST.write_text(json.dumps(used,indent=2),encoding='utf-8')
    asset=ASSETS/f'{slug}.jpg'
    with Image.open(source) as original:
        im=ImageOps.exif_transpose(original).convert('RGB')
        im.thumbnail((320,320),Image.Resampling.LANCZOS)
        im.save(asset,quality=88,optimize=True)
    return {'artist':NAMES.get(row['artist'],row['artist'].replace('-',' ').title()),
            'title':row['title'],'year':row.get('year'),'asset':f'assets/works/{asset.name}',
            'source':'WikiArt reproduction',**values}

def first_artists(rows,n=3):
    result=[];seen=set()
    for row in rows:
        if row['artist'] in seen:continue
        seen.add(row['artist']);result.append(row)
        if len(result)==n:break
    return result
