"""Splits a tall gallery PDF into page parts on the rules between rows; each part keeps only its own rows' images, downsampled to 200 dpi at print size. Writes NAME-print4_p1.pdf, _p2 ... and refuses to overwrite.
Usage: python print_split_v4.py NAME..."""
import io
import os
import sys
from pathlib import Path

import fitz
import pikepdf

import print_split_v3 as v3

HERE = Path(os.environ['DUROOM_OUT']) / 'figures'


def part(name, y0, y1, i):
    src = HERE / f'{name}.pdf'
    dst = HERE / f'{name}-print4_p{i}.pdf'
    assert not dst.exists(), f'{dst} exists: refusing to overwrite'
    doc = fitz.open(src)
    pg = doc[0]
    w, h = pg.rect.width, pg.rect.height
    for info in pg.get_image_info():
        r = fitz.Rect(info['bbox'])
        assert not (r.y0 < y0 - 0.5 < r.y1 - 1) and not (r.y0 + 1 < y1 + 0.5 < r.y1), f'{name}: an image straddles a cut at {y0:.0f} or {y1:.0f}'
    if y0 > 0:
        pg.add_redact_annot(fitz.Rect(0, 0, w, y0 - 0.5))
    if y1 < h:
        pg.add_redact_annot(fitz.Rect(0, y1 + 0.5, w, h))
    pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE, graphics=fitz.PDF_REDACT_LINE_ART_NONE, text=fitz.PDF_REDACT_TEXT_REMOVE)
    kept = len(pg.get_images())
    doc.rewrite_images(dpi_threshold=250, dpi_target=200, quality=85)
    pdf = pikepdf.open(io.BytesIO(doc.tobytes(garbage=4, deflate=True)))
    p = pdf.pages[0]
    mb = [float(x) for x in p.obj.MediaBox]
    assert mb[1] == 0 and abs(mb[3] - h) < 1, mb
    box = pikepdf.Array([mb[0], h - y1, mb[2], h - y0])
    p.obj.MediaBox = box; p.obj.CropBox = box
    pdf.save(dst, compress_streams=True, object_stream_mode=pikepdf.ObjectStreamMode.generate)
    return dst.name, round(y1 - y0), kept, dst.stat().st_size


if __name__ == '__main__':
    for nm in sys.argv[1:]:
        h, bands = v3.bands_of(nm)
        n_all = len(fitz.open(HERE / f'{nm}.pdf')[0].get_images())
        outs = [part(nm, y0, y1, i) for i, (y0, y1) in enumerate(bands, 1)]
        assert sum(o[2] for o in outs) >= n_all, f'{nm}: images lost ({sum(o[2] for o in outs)} of {n_all})'
        print(f'{nm}: {h:.0f} pt, {n_all} images -> ' + '; '.join(f'{o[0]} {o[1]} pt, {o[2]} img, {o[3] / 1e6:.2f} MB' for o in outs))
