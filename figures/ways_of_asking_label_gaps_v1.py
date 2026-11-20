"""Smallest gap between any two texts of a figure (px at 200 and 96 dpi), any text over a bar, and the smallest font, for vert v4 and vert v5, desktop and phone.
Usage: python -B ways_of_asking_label_gaps_v1.py"""
import math
import sys

import matplotlib.text as mtext
from matplotlib.patches import Rectangle


def measure(fig, label):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in fig.findobj(mtext.Text):
        s = t.get_text().strip()
        if not s or not t.get_visible():
            continue
        bb = t.get_window_extent(r)
        if bb.width < 1:
            continue
        items.append((s.replace("\n", " / "), bb, t.get_fontsize()))
    best = (1e9, None, None)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            dx = max(0, b.x0 - a.x1, a.x0 - b.x1)
            dy = max(0, b.y0 - a.y1, a.y0 - b.y1)
            g = math.hypot(dx, dy)
            if g < best[0]:
                best = (g, items[i][0], items[j][0])
    # text over a drawn bar (filled or open); the dashed pending outline holds its own word, so it is skipped
    over = []
    for ax in fig.axes:
        for p in ax.patches:
            if not isinstance(p, Rectangle) or p.get_linestyle() != "-":
                continue
            pb = p.get_window_extent(r)
            for s, bb, _ in items:
                if bb.x0 < pb.x1 and bb.x1 > pb.x0 and bb.y0 < pb.y1 and bb.y1 > pb.y0:
                    over.append(s)
    smallest = min(f for _, _, f in items)
    fw, fh = fig.get_size_inches()
    out = (f"{label}: {len(items)} texts; min gap {best[0]:.1f} px @200dpi = {best[0] * 0.48:.1f} px @96dpi "
           f"between [{best[1]}] and [{best[2]}]; text over a bar: {over or 'none'}; smallest font {smallest:.1f} pt; fig {fw:.2f}x{fh:.2f} in")
    return out, best[0]


if __name__ == "__main__":
    import ways_of_asking_vert_v4 as v4
    import ways_of_asking_vert_v5 as v5
    for lab, fn in (("v4 desktop (reference)", v4.desktop), ("v4 mobile (reference)", v4.mobile),
                    ("v5 desktop", v5.desktop), ("v5 mobile", v5.mobile)):
        print(measure(fn(), lab)[0])
