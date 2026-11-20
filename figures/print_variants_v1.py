"""Print version of a figure PDF: the paper colour F1EBE0 becomes white in the page and every form XObject; every other colour and raster is unchanged. Writes NAME-print.pdf, refuses to overwrite.
Usage: python print_variants_v1.py warm NAME..."""
import re
import os
import sys
from pathlib import Path

import fitz
import numpy as np
import pikepdf

HERE = Path(os.environ['DUROOM_OUT']) / 'figures'
WARM = (0.945098, 0.921569, 0.878431)          # F1EBE0
TOL = 0.004
NUM = rb'(-?[0-9]*\.?[0-9]+)'
COLOR = re.compile(NUM + rb'\s+' + NUM + rb'\s+' + NUM + rb'\s+(rg|RG|sc|SC|scn|SCN)\b')


def _swap(data):
    n = [0]

    def rep(m):
        v = [float(m.group(i)) for i in (1, 2, 3)]
        if all(abs(a - b) <= TOL for a, b in zip(v, WARM)):
            n[0] += 1
            return b'1 1 1 ' + m.group(4)
        return m.group(0)
    return COLOR.sub(rep, data), n[0]


def _streams(pdf):
    """every content stream: pages' contents, and every Form XObject / Tiling pattern stream in the file"""
    seen = set()
    for pg in pdf.pages:
        c = pg.obj.get('/Contents')
        for s in (c if isinstance(c, pikepdf.Array) else [c]):
            if s is not None and s.objgen not in seen:
                seen.add(s.objgen)
                yield s
    for obj in pdf.objects:
        if isinstance(obj, pikepdf.Stream) and obj.objgen not in seen:
            st = obj.get('/Subtype')
            if st == '/Form' or obj.get('/PatternType') == 1:
                seen.add(obj.objgen)
                yield obj


def warm_to_white(name):
    src, dst = HERE / f'{name}.pdf', HERE / f'{name}-print.pdf'
    assert src.exists(), src
    assert not dst.exists(), f'{dst} exists: refusing to overwrite'
    pdf = pikepdf.open(src)
    total = 0
    for s in _streams(pdf):
        data = s.read_bytes()
        new, n = _swap(data)
        if n:
            s.write(new)
            total += n
    pdf.save(dst)
    return total


def warm_pixels(path):
    """pixels within 3/255 of F1EBE0 on a 1x render (raster images included)"""
    pm = fitz.open(path)[0].get_pixmap(dpi=72)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, :3].astype(int)
    return int((np.abs(a - np.array([241, 235, 224])).max(2) <= 3).sum()), pm.w * pm.h


if __name__ == '__main__':
    mode, names = sys.argv[1], sys.argv[2:]
    for nm in names:
        if mode == 'warm':
            n = warm_to_white(nm)
            b, a = warm_pixels(HERE / f'{nm}.pdf'), warm_pixels(HERE / f'{nm}-print.pdf')
            print(f'{nm}: {n} colour operators F1EBE0 -> white; paper-colour pixels {b[0]} -> {a[0]} (of {b[1]})')
