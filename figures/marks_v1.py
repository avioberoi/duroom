"""Helpers of the gallery figures: the plate class (MPlate), title block, off-the-manifold rows and genre bars with a real work beside each; reads figures/marks/SHOW-off1900.csv and SHOW-genres.csv (supplied by the reader).
Usage: imported by search_finds_off1900_v5.py, search_genres_v3.py"""
import csv

from PIL import Image

from search_finds_v1 import FS, Plate, make, runs_wrap, width_in
from _style import *


def rd(n):
    if not (HERE / 'marks' / n).exists():
        raise SystemExit(f'missing {HERE / "marks" / n}: supply the input file')
    return list(csv.DictReader(open(HERE / 'marks' / n, encoding='utf-8')))


OFF, GEN = rd('SHOW-off1900.csv'), rd('SHOW-genres.csv')
NAMES = {'salvador-dali': 'Salvador Dalí'}
name = lambda slug: NAMES.get(slug, ' '.join(w.capitalize() for w in slug.split('-')))


class MPlate(Plate):
    """Plate with pictures from absolute paths, downscaled, returning the drawn width; plus bars, dots, right text."""
    def image(self, path, x, y_bottom, w_box, h_box, px=600, top=False):
        im = Image.open(path).convert('RGB'); im.thumbnail((px, px))
        iw, ih = im.size; k = min(w_box / iw, h_box / ih); w, h = iw * k, ih * k
        self.ops.append(('image', im, x, (y_bottom - h_box) if top else (y_bottom - h), w, h))
        return w

    def rtext(self, x_right, y, s, size, color=INK, italic=False):
        self.text(x_right - width_in(s, size, italic), y, s, size, color, italic)

    def block(self, runs_list, x, y_top, w, size, color=INK):
        """Wrapped credit blocks (each a list of (text, italic) runs); returns the height used."""
        lh = size * 1.32 / 72; y = y_top
        for runs in runs_list:
            for line in runs_wrap(runs, size, w):
                y += lh
                for word, it, dx in line:
                    self.text(x + dx, y, word, size, color, it)
        return y - y_top

    def figure(self, H):
        extra = [op for op in self.ops if op[0] in ('bar', 'dot', 'seg')]
        self.ops = [op for op in self.ops if op[0] not in ('bar', 'dot', 'seg')]
        fig = Plate.figure(self, H); ax = fig.axes[0]
        for op in extra:
            if op[0] == 'bar':
                _, x0, x1, ya, yb = op
                ax.fill_between([x0, x1], [ya, ya], [yb, yb], color=PALE, lw=0, zorder=1)
            elif op[0] == 'dot':
                _, x, y = op
                ax.plot([x], [y], 'o', ms=4.2, color=ACCENT, mec='none', zorder=3)
            else:
                _, xa, xb, ya, yb = op
                ax.plot([xa, xb], [ya, yb], color=HAIR, lw=.6, solid_capstyle='butt', zorder=2)
        ax.set_xlim(0, self.W); ax.set_ylim(H, 0)
        return fig


def title_block(P, y, title, sub, W, size=12.5):
    for line in runs_wrap([(title, False)], size, W):
        y += size * 1.3 / 72
        for word, it, dx in line:
            P.text(dx, y, word, size)
    y += 0.06
    y += P.block([[(sub, False)]], 0, y, W, 9.0, MUTE)
    return y + 0.22


for r in OFF:
    assert r['observer'] == 'SigLIP' and float(r['chord']) > float(r['line']) and not r['start_image']


def off_start(r):
    return [[(f"{name(r['start_artist'])}, {r['start_year']}", False)], [(r['start_title'], True)]]


def off_nearest(r):
    return [[(f"{name(r['nearest_artist'])}, {r['nearest_year']}", False)], [(r['nearest_title'], True)]]


def genre_rows():
    from search_figures import data
    rows = next(c for c in data('retro')['cuts'] if c['cutoff'] == 1900)['genres']
    rows = rows[:4] + [r for r in rows[4:-5] if r['label'] == 'abstract'] + rows[-5:]   # as search_genres
    pick = {}
    for r in GEN:
        pick.setdefault(r['genre'], []).append(r)
    out = []
    for r in rows:
        g = r['label']
        works = pick.get(g, [])
        w = None if g.startswith('nude') else works[1 if g == 'installation' else 0]
        if w is not None:
            assert abs(float(w['share_over_bar']) - r['share']) < 1e-6, g
        out.append((g.replace('nude painting (nu)', 'nude'), r['share'], w))
    return out


G_TITLE = '70% of later installations are over the bar, 1% of flower paintings'


def g_credit(w):
    return [[(f"{name(w['artist'])}, {w['year']}", False)], [(w['title'], True)]]
