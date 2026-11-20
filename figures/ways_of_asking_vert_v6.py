"""Far and faithful pictures by way of asking as vertical bars, as ways_of_asking_vert_v5 with a bar for the same search scored in CLIP (the search's accent at a lighter tint) after DuROOM search; its count is read from r4_readout_v2.txt.
Usage: python ways_of_asking_vert_v6.py"""
import os
import re
from pathlib import Path

import ways_of_asking_vert_v5 as base
from ways_of_asking_vert_v5 import vert, W, share, REAL, FREE, YMAX, chart, GAP
from _style import *

CLIP = "search scored in CLIP"
assert CLIP not in W.ROWS
R4_TXT = Path(os.environ["DUROOM_OUT"]) / "tests12" / "more" / "r4_readout_v2.txt"
if not R4_TXT.exists():
    raise SystemExit(f"missing {R4_TXT}: run measure/r4_readout_v2.py (run/09) with all 10 CLIP-scored runs")
m = re.search(r"^\s+CLIP-scored search\s+FF [0-9/]+ = ([0-9]+) ;", R4_TXT.read_text(encoding="utf-8"), re.M)
if m is None:
    raise SystemExit(f"{R4_TXT}: no complete 'CLIP-scored search' line (all 5 starts)")
W.ROWS[CLIP] = {"way": CLIP, "kind": "search_clip", "n": "9600", "far_and_faithful": m.group(1)}
CLIP_TINT = "#C89B8D"   # ACCENT at 45% over PAPER


def draw_slot(ax, x, w, style, width):
    if style == "search2":
        s = share(w)
        ax.bar(x, s, width=width, bottom=0, zorder=3, color=CLIP_TINT, edgecolor=ACCENT, linewidth=1.0)
        return s
    return vert.draw_bar(ax, x, w, style, width)


base.draw_slot = draw_slot   # v5.chart looks draw_slot up in its module at call time

ROWS = [("no search,\n6 kinds of\nrequest", "all six recipes", "total"),
        ("single walk\nwith the\nscore", "single walk with selection", "loop"),
        ("DuROOM\nsearch\n(evolutionary\nsearch with\nthe score)", "search", "search"),
        ("the same\nsearch, far\nscored in\nCLIP", CLIP, "search2"),
        ("random-\nscore loop\n(selection\nat random)", "random-score loop", "loop"),
        ("no-step loop\n(selection\nat random,\nno variation)", "no-step loop", "loop"),
        None,
        ("free search,\n3 other\nstarts", FREE, "loop"),
        ("lines aimed\nat real later\nworks", "lines aimed at real later works", "ceiling"),
        None,
        ("before 1900,\nheld out", REAL[0], "real"),
        ("1900 to\n1969", REAL[1], "real"),
        ("from 1970\non", REAL[2], "real")]
SHORT = ["no\nsearch", "single\nwalk", "DuROOM\nsearch", "same\nsearch,\nCLIP\nscored", "random\nscore", "no\nstep",
         None, "free\nsearch", "lines to\nlater\nworks", None, "before\n1900", "1900\nto 1969", "1970\non"]
assert len(ROWS) == len(SHORT) and all((r is None) == (s is None) for r, s in zip(ROWS, SHORT))


def desktop():
    fig, ax, xs, bracket, below = chart(ROWS, (8.8, 5.4), [0.05, 0.30, 0.945, 0.55], 7.6, 10, 7.6, 0.62, GAP)
    bracket(1, 3, "single walk and the 2 searches,\n9,600 pictures each")
    bracket(4, 5, "DuROOM search's 2 controls,\n9,600 pictures each")
    bracket(7, 7, "own runs:\nno restart,\nno faithful\ncheck")
    bracket(8, 8, "hatched:\nuses later\nart")
    bracket(10, 12, "real works repainted, for scale")
    return fig


def mobile():
    rows = [None if r is None else (sh, r[1], r[2]) for r, sh in zip(ROWS, SHORT)]
    fig, ax, xs, bracket, below = chart(rows, (3.9, 6.3), [0.07, 0.33, 0.925, 0.49], 6.0, 8.4, 5.3, 0.66, 0.15,
                                        title_fs=11.5, title_lines=2, two_line_count=True, adv=ADV_MOBILE)
    bracket(1, 3, "walk, 2 searches")
    bracket(4, 5, "2 controls")
    bracket(7, 7, "own\nruns")
    bracket(8, 8, "later\nart")
    bracket(10, 12, "real works, for scale")
    return fig


ADV_MOBILE = {1: 1.03, 2: 1.14, 3: 1.2, 4: 1.12, 5: 1.04}

if __name__ == "__main__":
    for name, fn in (("ways_of_asking_vert_v6", desktop), ("ways_of_asking_vert_v6-mobile", mobile)):
        assert os.environ.get("VERT6_ITER") or not (HERE / f"{name}.svg").exists(), name
        save(fn(), name)
