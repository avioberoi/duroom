"""Short ladder of six upright bars in the order no search, single walk, DuROOM search, the same search scored in CLIP, random-score loop, no-step loop (CLIP row as in ways_of_asking_vert_v6).
Usage: python ways_of_asking_ladder_v4.py"""
import os

import ways_of_asking_vert_v6 as v6     # adds the CLIP row (count read from r4_readout_v2.txt) to W.ROWS, and CLIP_TINT
import ways_of_asking_titles_v1  # noqa: F401
import ways_of_asking_ladder_v1 as ladder
import ways_of_asking_vert_v2 as V
from _style import *

_orig_draw_bar = V.draw_bar


def draw_bar(ax, x, w, style, width):
    if style == "search2":
        s = V.share(w)
        ax.bar(x, s, width=width, bottom=0, zorder=3, color=v6.CLIP_TINT, edgecolor=ACCENT, linewidth=1.0)
        return s
    return _orig_draw_bar(ax, x, w, style, width)


V.draw_bar = draw_bar   # ladder_v1.draw calls V.draw_bar

ORDER = ["all six recipes", "single walk with selection", "search", v6.CLIP, "random-score loop", "no-step loop"]
NAMES = ["no search, 6 kinds", "single walk", "DuROOM search", "the same search,\nfar scored in CLIP", "random-score loop", "no-step loop"]
STYLES = ["total", "loop", "search", "search2", "loop", "loop"]
ladder.ROWS = list(zip(NAMES, ORDER, STYLES))


def desktop():
    return ladder.desktop()


def mobile():
    keep = ladder.TITLE
    ladder.TITLE = "With the 6 kinds of request and no search,\n0.5% of the pictures are far and faithful;\nDuROOM search paints 46%"
    try:
        H = 3.4 + 0.2
        return ladder.draw((3.9, H), [0.11, 0.25 * 3.4 / H, 0.87, 0.46 * 3.4 / H], 7.2, 8.4, 6.4, 0.5, 11.0, 3, True,
                           ["no search", "single\nwalk", "DuROOM\nsearch", "same\nsearch,\nCLIP", "random-\nscore loop", "no-step\nloop"])
    finally:
        ladder.TITLE = keep


if __name__ == "__main__":
    for name, fn in (("ways_of_asking_ladder_v4", desktop), ("ways_of_asking_ladder_v4-mobile", mobile)):
        assert os.environ.get("LADDER4_ITER") or not (HERE / f"{name}.svg").exists(), name
        save(fn(), name)
