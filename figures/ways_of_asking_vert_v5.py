"""Far and faithful pictures by way of asking as vertical bars in the order no search, single walk, DuROOM search, an empty slot for the search scored in CLIP, the two controls, then free search, lines aimed at real later works and the real works repainted.
Usage: python ways_of_asking_vert_v5.py"""
import os

from matplotlib.transforms import ScaledTranslation

import ways_of_asking_titles_v1  # noqa: F401  (import sets the title on ways_of_asking_vert_v2)
import ways_of_asking_vert_v2 as vert
import ways_of_asking_v4 as W
from _style import *

FREE = "free search"
if FREE not in W.ROWS:   # the csv has no free-search row; add it once, in memory
    W.ROWS[FREE] = {"way": FREE, "kind": "free", "n": "5760", "far": "4094", "far_and_faithful": "0"}
PENDING = "far scored in CLIP"   # an empty slot, not a csv row

num, share, REAL = W.num, W.share, W.REAL
YMAX = vert.YMAX

# (name, csv way or PENDING, style)
ROWS = [("no search,\n6 kinds of\nrequest", "all six recipes", "total"),
        ("single walk\nwith the\nscore", "single walk with selection", "loop"),
        ("DuROOM\nsearch\n(sampling\nwith the\nscore)", "search", "search"),
        ("far scored\nin CLIP", PENDING, "pending"),
        ("random-\nscore loop\n(sampling,\nno score)", "random-score loop", "loop"),
        ("no-step\nloop (no\nscore, no\nstep)", "no-step loop", "loop"),
        None,
        ("free search,\n3 other\nstarts", FREE, "loop"),
        ("lines aimed\nat real later\nworks", "lines aimed at real later works", "ceiling"),
        None,
        ("before 1900,\nheld out", REAL[0], "real"),
        ("1900 to\n1969", REAL[1], "real"),
        ("from 1970\non", REAL[2], "real")]
SHORT = ["no\nsearch", "single\nwalk", "DuROOM\nsearch", "far\nscored\nin\nCLIP", "random\nscore", "no\nstep",
         None, "free\nsearch", "lines to\nlater\nworks", None, "before\n1900", "1900\nto 1969", "1970\non"]
assert len(ROWS) == len(SHORT) and all((r is None) == (s is None) for r, s in zip(ROWS, SHORT))


def draw_slot(ax, x, w, style, width):
    """a data bar, or the empty dashed slot"""
    if style == "pending":
        ax.add_patch(plt.Rectangle((x - width / 2, 0), width, YMAX, fill=False, edgecolor=W.MID, lw=0.8, ls=(0, (3, 2.5)), zorder=2))
        return None
    return vert.draw_bar(ax, x, w, style, width)


def chart(rows, size, axrect, fs_name, fs_rate, fs_count, width, gap, title_fs=12.5, title_lines=1, two_line_count=False, adv=None):
    """vert_v2.chart with an empty slot, a settable gap and lowered brackets"""
    fig = plt.figure(figsize=size, dpi=200)
    t = vert.TITLE.replace("; ", ";\n") if title_lines == 2 else vert.TITLE
    fig.text(0.0, 0.995, t, fontsize=title_fs, color=INK, ha="left", va="top", linespacing=1.15)
    sub_y = 0.995 - (title_lines * title_fs * 1.25 + 6) / 72 / size[1]
    sub = vert.SUB.replace(" of the pictures", "\nof the pictures").replace(" painted. ", " painted.\n") if title_lines == 2 else vert.SUB
    fig.text(0.0, sub_y, sub, fontsize=9.3, color=MUTE, ha="left", va="top", linespacing=1.2)
    ax = fig.add_axes(axrect)
    xs, x = [], 0.0
    for r in rows:
        x += gap if r is None else (adv or {}).get(len(xs), 1.0)   # adv: wider advance into a slot where 2 wide names meet
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
        s = draw_slot(ax, xx, w, style, width)
        ax.annotate(label, (xx, 0), xytext=(0, name_off), textcoords="offset points", fontsize=fs_name,
                    color=INK, ha="center", va="top", linespacing=1.1, annotation_clip=False)
        if style == "pending":
            ax.text(xx, YMAX / 2, "pending", fontsize=fs_count, color=MUTE, ha="center", va="center", rotation=90,
                    zorder=5, bbox=dict(facecolor=PAPER, edgecolor="none", pad=0.8))
            continue
        ax.text(xx, s + 1.2, vert.rate_text(s), fontsize=fs_rate, color=INK, ha="center", va="bottom",
                fontweight="bold" if style == "search" else "normal")
        ax.annotate(vert.count_text(w, two_line_count), (xx, 0), xytext=(0, count_off), textcoords="offset points",
                    fontsize=fs_count, color=MUTE, ha="center", va="top", linespacing=1.05, annotation_clip=False)

    def bracket(i0, i1, text, extra=0):
        a, b = xs[i0] - width / 2, xs[i1] + width / 2
        off = below - extra
        shift = ax.transData + ScaledTranslation(0, off / 72, fig.dpi_scale_trans)
        ax.plot([a, b], [0, 0], color=HAIR, lw=0.8, clip_on=False, transform=shift)
        ax.annotate(text, ((a + b) / 2, 0), xytext=(0, off - 3), textcoords="offset points", fontsize=fs_count,
                    color=MUTE, ha="center", va="top", annotation_clip=False)
    return fig, ax, xs, bracket, below


GAP = 0.4


def desktop():
    fig, ax, xs, bracket, below = chart(ROWS, (8.8, 5.4), [0.05, 0.30, 0.945, 0.55], 7.6, 10, 7.6, 0.62, GAP)
    bracket(1, 2, "single walk and DuROOM search,\n9,600 pictures each")
    bracket(4, 5, "DuROOM search's 2 controls,\n9,600 pictures each")
    bracket(7, 7, "own runs:\nno restart,\nno faithful\ncheck")
    bracket(8, 8, "hatched:\nuses later\nart")
    bracket(10, 12, "real works repainted, for scale")
    return fig


def mobile():
    rows = [None if r is None else (sh, r[1], r[2]) for r, sh in zip(ROWS, SHORT)]
    fig, ax, xs, bracket, below = chart(rows, (3.9, 6.3), [0.07, 0.33, 0.925, 0.49], 6.0, 8.4, 5.3, 0.66, 0.15,
                                        title_fs=11.5, title_lines=2, two_line_count=True, adv={1: 1.03, 2: 1.14, 3: 1.2, 4: 1.05, 5: 1.04})
    bracket(1, 2, "walk, search")
    bracket(4, 5, "2 controls")
    bracket(7, 7, "own\nruns")
    bracket(8, 8, "later\nart")
    bracket(10, 12, "real works, for scale")
    return fig


if __name__ == "__main__":
    for name, fn in (("ways_of_asking_vert_v5", desktop), ("ways_of_asking_vert_v5-mobile", mobile)):
        assert os.environ.get("VERT5_ITER") or not (HERE / f"{name}.svg").exists(), name
        save(fn(), name)
