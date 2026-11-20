"""Genre chart: one upright bar per genre (share of later works above the 99th-percentile bar) with a real work of that genre under it. Reads data/search_retro.json and figures/marks/SHOW-genres.csv.
Usage: python search_genres_v3.py"""
from marks_v1 import G_TITLE, MPlate, g_credit, genre_rows, runs_wrap, title_block
from search_finds_v1 import make, width_in
from _style import *

G_SUB_V3 = ('WikiArt genres with at least 200 later works: the 4 highest, the 5 lowest, and abstract. Each bar is the share '
            'of later works above the 99th-percentile bar; under it, a real work of that genre dated 1900 or later.')


def desk():
    W = 8.8
    P = MPlate(W)
    y = title_block(P, 0.0, G_TITLE, G_SUB_V3, W)
    rows = genre_rows()
    n = len(rows); col = W / n; inner = col - 0.1
    hb, k = 1.9, 1.9 / 75                       # bar area height; 75% = 1.9 in
    base = y + 0.25 + hb
    P.ops.append(('seg', 0, W - 0.1, base, base))
    lab_h = 0.0
    for i, (g, share, w) in enumerate(rows):
        x = i * col
        top = base - 100 * share * k
        P.ops.append(('bar', x, x + 0.46, top, base)); P.ops.append(('dot', x + 0.23, top))
        s = f'{100 * share:.0f}%'
        P.text(x + 0.23 - width_in(s, 8.6) / 2, top - 0.09, s, 8.6)
        lab_h = max(lab_h, P.block([[(g, False)]], x, base + 0.04, inner, 9.0))
    yt = base + 0.04 + lab_h + 0.12
    box = inner
    for i, (g, share, w) in enumerate(rows):
        if w is None:
            continue
        x = i * col
        P.image(w['image'], x, yt + box, box, box, px=300, top=True)
        P.block(g_credit(w), x, yt + box + 0.04, inner, 7.6)
    return P, yt + box + 0.04 + 4 * 7.6 * 1.32 / 72 + 0.08


if __name__ == '__main__':
    assert not (HERE / 'search_genres_v3.svg').exists()
    P, H = desk(); make('search_genres_v3', P, H)
