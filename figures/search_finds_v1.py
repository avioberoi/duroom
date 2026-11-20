"""Gallery figures of pictures the search painted: each row the starting work, the painted picture and a real work; plates drawn in inches (Plate), shared by the other gallery scripts. Reads data/search_finds_v1.csv and search_finds_today_v1.csv (derive_search_finds.py).
Usage: python search_finds_v1.py  (writes search_find_pair_v2)"""
import base64
import csv
import io
import re

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from PIL import Image

from _style import *

FAMILY = [SERIF, 'Book Antiqua']
FONTS = {False: FontProperties(family=FAMILY), True: FontProperties(family=FAMILY, style='italic')}
A_ROWS = list(csv.DictReader(open(HERE / 'data' / 'search_finds_v1.csv', encoding='utf-8')))
C_ROWS = list(csv.DictReader(open(HERE / 'data' / 'search_finds_today_v1.csv', encoding='utf-8')))
B_ROW = [r for r in A_ROWS if r['found'].endswith('found_026.jpg')]
assert len(B_ROW) == 1

HEADS = {'A': ('starting work, before 1900', 'painted by the search', 'real work dated 1900 or later'),
         'C': ('starting work', 'painted by the search', 'nearest real work')}
FS = {'desk': dict(head=8.8, credit=8.6, found=8.6), 'mob': dict(head=9.2, credit=9.0, found=9.0)}


def width_in(text, size, italic=False):
    return TextPath((0, 0), text, size=size, prop=FONTS[italic]).get_extents().width / 72 if text else 0.0


def space_in(size):
    return width_in('a a', size) - width_in('aa', size)


def segments(line):
    """A line's words grouped into runs of one style, each drawn as one text so the font sets its own word spaces."""
    segs = []
    for w, it, glued in line:
        if segs and segs[-1][1] == it:
            segs[-1][0] += ' ' + w
        else:
            segs.append([w, it, bool(segs) and glued and not w[0].isalnum()])
    return segs


def line_width(line, size):
    x = 0.0
    for i, (s, it, glue) in enumerate(segments(line)):
        x += (0 if glue else space_in(size)) if i else 0
        x += width_in(s, size, it)
    return x


def runs_wrap(runs, size, limit):
    """Greedy wrap of (text, italic) runs into lines of (text, italic, x) segments, by measured width."""
    words = []
    for text, it in runs:
        for k, w in enumerate(text.split(' ')):
            if w:
                words.append((w, it, k == 0 and not text.startswith(' ')))
    lines, cur = [], []
    for wd in words:
        if cur and line_width(cur + [wd], size) > limit:
            lines.append(cur); cur = []
        cur.append(wd)
    if cur:
        lines.append(cur)
    out = []
    for line in lines:
        x, placed = 0.0, []
        for i, (s, it, glue) in enumerate(segments(line)):
            x += (0 if glue else space_in(size)) if i else 0
            placed.append((s, it, x)); x += width_in(s, size, it)
        out.append(placed)
    return out


def credit_runs(r, role):
    if role == 'found':
        return [[(f"painted by the search, round {int(r['round'])}", False)]]
    a, t, y = r[f'{role}_artist'], r[f'{role}_title'], r[f'{role}_year']
    return [[(a, False)], [(t, True), (f', {y}', False)]]


class Plate:
    """Collects drawing operations in inches from the top-left, then draws them on one axes."""
    def __init__(self, W):
        self.W, self.ops, self.y = W, [], 0.0

    def text(self, x, y, s, size, color=INK, italic=False):
        self.ops.append(('text', x, y, s, size, color, italic))

    def image(self, path, x, y_bottom, w_box, h_box):
        im = Image.open(HERE / path).convert('RGB'); iw, ih = im.size
        k = min(w_box / iw, h_box / ih); w, h = iw * k, ih * k
        self.ops.append(('image', im, x, y_bottom - h, w, h))   # flush left, bottom-aligned: the credit sits under it

    def rule(self, y, x0=0.0, x1=None):
        self.ops.append(('rule', x0, self.W if x1 is None else x1, y))

    def credits(self, r, role, x, y_top, w, fs):
        """Credits under a picture; returns their height."""
        lh = fs * 1.32 / 72
        y = y_top
        for block in credit_runs(r, role):
            size = fs
            for line in runs_wrap(block, size, w):
                y += lh
                for word, it, dx in line:
                    self.text(x + dx, y, word, size, MUTE if role == 'found' else INK, it)
        return y - y_top

    def figure(self, H):
        fig = plt.figure(figsize=(self.W, H))
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, self.W); ax.set_ylim(H, 0); ax.axis('off')
        for op in self.ops:
            if op[0] == 'image':
                _, im, x, y, w, h = op
                ax.imshow(im, extent=(x, x + w, y + h, y), interpolation='none', aspect='auto')
            elif op[0] == 'rule':
                _, x0, x1, y = op
                ax.plot([x0, x1], [y, y], color=HAIR, lw=.6, solid_capstyle='butt')
            else:
                _, x, y, s, size, color, it = op
                ax.text(x, y, s, fontsize=size, color=color, fontfamily=FAMILY, fontstyle='italic' if it else 'normal',
                        ha='left', va='baseline')
        ax.set_xlim(0, self.W); ax.set_ylim(H, 0)
        return fig


def desk_rows(rows, third, heads, title=None):
    W, gap = 8.8, 0.22
    cell = (W - 2 * gap) / 3
    P, fs = Plate(W), FS['desk']
    y = 0.0
    if title:
        P.text(0, 0.24, title, 12.5); y = 0.42
    xs = [0, cell + gap, 2 * (cell + gap)]
    for x, h in zip(xs, heads):
        P.text(x, y + 0.16, h, fs['head'], MUTE)
    y += 0.30
    prev = None
    for r in rows:
        if prev is not None and r['start'] != prev:
            P.rule(y + 0.02); y += 0.26
        top = y
        for x, role in zip(xs, ('start', 'found', third)):
            P.image(r[role] if role != 'found' else r['found'], x, top + cell, cell, cell)
        hs = [P.credits(r, role, x, top + cell + 0.03, cell, fs['credit']) for x, role in zip(xs, ('start', 'found', third))]
        y = top + cell + 0.03 + max(hs) + 0.30
        prev = r['start']
    return P, y - 0.12


def mob_rows(rows, third, heads, title=None):
    W, gap = 3.9, 0.14
    half = (W - gap) / 2
    P, fs = Plate(W), FS['mob']
    y = 0.0
    if title:
        P.text(0, 0.24, title, 12.5); y = 0.42
    prev = None
    for r in rows:
        if prev is not None and r['start'] != prev:
            P.rule(y + 0.02); y += 0.30
        P.text(0, y + 0.15, heads[1], fs['head'], MUTE)
        top = y + 0.24
        P.image(r['found'], 0, top + W, W, W)
        y = top + W + 0.03 + P.credits(r, 'found', 0, top + W + 0.03, W, fs['found']) + 0.22
        P.text(0, y + 0.15, heads[0], fs['head'], MUTE)
        P.text(half + gap, y + 0.15, heads[2], fs['head'], MUTE)
        top = y + 0.24
        P.image(r['start'], 0, top + half, half, half)
        P.image(r[third], half + gap, top + half, half, half)
        hs = [P.credits(r, 'start', 0, top + half + 0.03, half, fs['credit']),
              P.credits(r, third, half + gap, top + half + 0.03, half, fs['credit'])]
        y = top + half + 0.03 + max(hs) + 0.42
        prev = r['start']
    return P, y - 0.2


def pair_desk(r):
    P, H = desk_rows([r], 'later', HEADS['A'])
    return P, H


def jpeg_svg(name):
    """Store the embedded pictures as JPEG and drop matplotlib's pixelated rendering hint."""
    path = HERE / f'{name}.svg'
    text = path.read_text(encoding='utf-8')
    text = text.replace(' style="image-rendering:crisp-edges;image-rendering:pixelated"', '')

    def jpeg(m):
        im = Image.open(io.BytesIO(base64.b64decode(re.sub(r'\s+', '', m.group(1)))))
        buf = io.BytesIO(); im.convert('RGB').save(buf, 'JPEG', quality=84, optimize=True, progressive=True)
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

    text = re.sub(r'data:image/png;base64,([A-Za-z0-9+/=\s]+)', jpeg, text)
    assert 'image/png' not in text and 'pixelated' not in text
    path.write_text(text, encoding='utf-8')


def make(name, P, H):
    save(P.figure(H), name)
    jpeg_svg(name)


if __name__ == '__main__':
    P, H = pair_desk(B_ROW[0]); make('search_find_pair_v2', P, H)
    P, H = mob_rows([B_ROW[0]], 'later', HEADS['A']); make('search_find_pair_v2-mobile', P, H)
