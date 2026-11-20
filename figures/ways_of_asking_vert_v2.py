"""Far and faithful pictures by way of asking as vertical bars (the ladder with the real works for scale, and the plain version); data and checks are ways_of_asking_v4's.
Usage: python ways_of_asking_vert_v2.py"""
import os

from matplotlib.transforms import ScaledTranslation

import ways_of_asking_v4 as W
from _style import *

num, share = W.num, W.share
REAL = W.REAL
YMAX = 50

# (label, csv way, style)
LADDER = [("no search,\n6 kinds of\nrequest", "all six recipes", "total"),
          ("lines aimed\nat real later\nworks", "lines aimed at real later works", "ceiling"),
          ("a single walk\nwith selection", "single walk with selection", "loop"),
          ("random-score\nloop", "random-score loop", "loop"),
          ("no-step\nloop", "no-step loop", "loop"),
          ("DuROOM\nsearch", "search", "search"),
          None,
          ("before 1900,\nheld out", REAL[0], "real"),
          ("1900 to\n1969", REAL[1], "real"),
          ("from 1970\non", REAL[2], "real")]
SHORT = ["no\nsearch", "lines to\nlater\nworks", "single\nwalk", "random\nscore", "no\nstep", "DuROOM\nsearch",
         None, "before\n1900", "1900\nto 1969", "1970\non"]

PLAIN = [("midpoints\nof 2\nworks", "midpoints of two works", "small"),
         ("line\nthrough 2\nregions'\ncentres", "between two regions' centres", "small"),
         ("35° steps\nfrom a\nwork", "35-degree steps from a work", "small"),
         ("lines past\na work", "lines past a work", "small"),
         ("random,\ninside the\nregion", "random requests inside the region", "small"),
         ("random,\nwhole\nsphere", "random requests on the whole sphere", "small"),
         ("all 6\nkinds", "all six recipes", "total"),
         None,
         ("lines\naimed at\nreal later\nworks", "lines aimed at real later works", "ceiling"),
         ("single\nwalk", "single walk with selection", "loop"),
         ("random-\nscore\nloop", "random-score loop", "loop"),
         ("no-step\nloop", "no-step loop", "loop"),
         ("DuROOM\nsearch", "search", "search"),
         None,
         ("before\n1900", REAL[0], "real"),
         ("1900 to\n1969", REAL[1], "real"),
         ("from\n1970 on", REAL[2], "real")]

TITLE = "46% of DuROOM search's pictures are far and faithful; with no search, 0.5%"
SUB = "Far and faithful, as a percent of the pictures each way of asking painted. Under each name, the count."


def rate_text(s):
    return "0%" if s == 0 else (f"{s:.1f}%" if s < 10 else f"{s:.0f}%")


def count_text(w, two_lines):
    t = f"{num(w, 'far_and_faithful'):,} of {num(w, 'n'):,}"
    return t.replace(" of ", "\nof ") if two_lines else t


def draw_bar(ax, x, w, style, width):
    s = share(w)
    kw = dict(width=width, bottom=0, zorder=3, linewidth=0)
    if style == "small":
        ax.bar(x, s, color=W.MID, **kw)
    elif style in ("total", "loop"):
        ax.bar(x, s, color=INK, **kw)
    elif style == "ceiling":
        ax.bar(x, s, color=PAPER, edgecolor=INK, hatch="////", **{**kw, "linewidth": 0.8})
    elif style == "search":
        ax.bar(x, s, color=ACCENT, **kw)
    elif style == "real":
        ax.bar(x, s, color=PAPER, edgecolor=MUTE, **{**kw, "linewidth": 1.0})
    if s < 0.6:   # a near-zero is a short tick on the baseline, so the bar is not read as missing
        ax.plot([x - width / 2, x + width / 2], [0, 0], color=W.MID if style == "small" else INK, lw=2.2, zorder=4,
                solid_capstyle="butt")
    return s


def chart(rows, size, axrect, fs_name, fs_rate, fs_count, width, title_fs=12.5, title_lines=1, two_line_count=False):
    fig = plt.figure(figsize=size, dpi=200)
    t = TITLE.replace("; ", ";\n") if title_lines == 2 else TITLE
    fig.text(0.0, 0.995, t, fontsize=title_fs, color=INK, ha="left", va="top", linespacing=1.15)
    sub_y = 0.995 - (title_lines * title_fs * 1.25 + 6) / 72 / size[1]
    sub = SUB.replace(" of the pictures", "\nof the pictures").replace(" painted. ", " painted.\n") if title_lines == 2 else SUB
    fig.text(0.0, sub_y, sub, fontsize=9.3, color=MUTE, ha="left", va="top", linespacing=1.2)
    ax = fig.add_axes(axrect)
    xs, x = [], 0.0
    for r in rows:
        x += 0.75 if r is None else 1.0
        xs.append(None if r is None else x)
    xmin, xmax = 0.35, x + 0.55
    ax.set_xlim(xmin, xmax); ax.set_ylim(0, YMAX * 1.08)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    for tk in range(0, YMAX + 1, 10):
        ax.plot([xmin, xmax], [tk, tk], color=HAIR, lw=0.6, zorder=1)
        ax.text(xmin, tk, f"{tk}% ", fontsize=fs_count, color=MUTE, ha="right", va="center")
    ax.plot([xmin, xmax], [0, 0], color=MUTE, lw=0.8, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    name_lines = max(r[0].count("\n") + 1 for r in rows if r)
    name_off = -5
    count_off = name_off - name_lines * fs_name * 1.2 - 3
    below = count_off - (2 if two_line_count else 1) * fs_count * 1.15 - 8   # where brackets start
    for r, xx in zip(rows, xs):
        if r is None:
            continue
        label, w, style = r
        s = draw_bar(ax, xx, w, style, width)
        ax.text(xx, s + 1.2, rate_text(s), fontsize=fs_rate, color=INK, ha="center", va="bottom",
                fontweight="bold" if style == "search" else "normal")
        ax.annotate(label, (xx, 0), xytext=(0, name_off), textcoords="offset points", fontsize=fs_name,
                    color=MUTE if style == "small" else INK, ha="center", va="top", linespacing=1.1, annotation_clip=False)
        ax.annotate(count_text(w, two_line_count), (xx, 0), xytext=(0, count_off), textcoords="offset points",
                    fontsize=fs_count, color=MUTE, ha="center", va="top", linespacing=1.05, annotation_clip=False)

    def bracket(i0, i1, text, extra=0):
        a, b = xs[i0] - width / 2, xs[i1] + width / 2
        off = below - extra
        shift = ax.transData + ScaledTranslation(0, off / 72, fig.dpi_scale_trans)
        ax.plot([a, b], [0, 0], color=HAIR, lw=0.8, clip_on=False, transform=shift)
        ax.annotate(text, ((a + b) / 2, 0), xytext=(0, off - 3), textcoords="offset points", fontsize=fs_count,
                    color=MUTE, ha="center", va="top", annotation_clip=False)
    return fig, ax, xs, bracket, below


def ladder_desktop():
    fig, ax, xs, bracket, below = chart(LADDER, (8.8, 5.3), [0.06, 0.29, 0.93, 0.56], 9.0, 10, 8.0, 0.62)
    bracket(3, 5, "DuROOM search and its 2 controls, 9,600 pictures each")
    bracket(7, 9, "real works repainted, for scale")
    bracket(1, 1, "hatched: uses later art")
    return fig


def ladder_mobile():
    rows = [None if r is None else (sh, r[1], r[2]) for r, sh in zip(LADDER, SHORT)]
    fig, ax, xs, bracket, below = chart(rows, (3.9, 6.1), [0.09, 0.33, 0.89, 0.50], 6.6, 8.4, 6.0, 0.66,
                                        title_fs=11.5, title_lines=2, two_line_count=True)
    bracket(3, 5, "search and 2 controls")
    bracket(7, 9, "real works, for scale")
    return fig


def plain_desktop():
    fig, ax, xs, bracket, below = chart(PLAIN, (8.8, 5.8), [0.05, 0.36, 0.94, 0.50], 8.0, 9.0, 7.4, 0.62, two_line_count=True)
    bracket(0, 6, "no search: 6 kinds of request not aimed at a later work")
    bracket(10, 12, "the search and its 2 controls")
    bracket(14, 16, "real works repainted, for scale")
    return fig


if __name__ == "__main__":
    for name, fn in (("ways_of_asking_vert_v2a", ladder_desktop), ("ways_of_asking_vert_v2a-mobile", ladder_mobile),
                     ("ways_of_asking_vert_v2b", plain_desktop)):
        assert os.environ.get("VERT_ITER") or not (HERE / f"{name}.svg").exists(), name
        save(fn(), name)
