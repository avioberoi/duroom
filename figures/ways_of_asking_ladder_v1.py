"""Short ladder of five upright bars: far and faithful as a percent of the pictures each way of asking painted, the search in the accent; bars from ways_of_asking_vert_v2.
Usage: python ways_of_asking_ladder_v1.py"""
import os

import ways_of_asking_vert_v2 as V
from _style import *

ROWS = [("no search, 6 kinds", "all six recipes", "total"),
        ("single walk", "single walk with selection", "loop"),
        ("random-score loop", "random-score loop", "loop"),
        ("no-step loop", "no-step loop", "loop"),
        ("DuROOM search", "search", "search")]
TITLE = "With no search, 0.5% of the pictures are far and faithful; DuROOM search paints 46%"
SUB = "Far and faithful, as a percent of the pictures each way of asking painted; the count under each name."
YMAX = 50


def draw(size, axrect, fs_name, fs_rate, fs_count, width, title_fs, title_lines, two_line_count, names):
    fig = plt.figure(figsize=size, dpi=200)
    t = TITLE.replace("; ", ";\n") if title_lines == 2 else TITLE
    fig.text(0.0, 0.995, t, fontsize=title_fs, color=INK, ha="left", va="top", linespacing=1.15)
    sub_y = 0.995 - (title_lines * title_fs * 1.25 + 5) / 72 / size[1]
    fig.text(0.0, sub_y, SUB if title_lines == 1 else SUB.replace(" of the pictures", "\nof the pictures").replace("; the count", ";\nthe count"),
             fontsize=8.8, color=MUTE, ha="left", va="top", linespacing=1.2)
    ax = fig.add_axes(axrect)
    xs = [1.0 + i for i in range(len(ROWS))]
    xmin, xmax = 0.45, xs[-1] + 0.55
    ax.set_xlim(xmin, xmax); ax.set_ylim(0, YMAX * 1.12)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    for tk in (0, 25, 50):
        ax.plot([xmin, xmax], [tk, tk], color=HAIR, lw=0.6, zorder=1)
        ax.text(xmin, tk, f"{tk}% ", fontsize=fs_count, color=MUTE, ha="right", va="center")
    ax.plot([xmin, xmax], [0, 0], color=MUTE, lw=0.8, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    nl = max(n.count("\n") + 1 for n in names)
    for (label, w, style), xx, nm in zip(ROWS, xs, names):
        s = V.draw_bar(ax, xx, w, style, width)
        ax.text(xx, s + 1.5, V.rate_text(s), fontsize=fs_rate, color=INK, ha="center", va="bottom", zorder=5,
                fontweight="bold" if style == "search" else "normal", bbox=dict(facecolor=PAPER, edgecolor="none", pad=0.6))
        ax.annotate(nm, (xx, 0), xytext=(0, -4), textcoords="offset points", fontsize=fs_name, color=INK, ha="center", va="top",
                    linespacing=1.1, annotation_clip=False)
        ax.annotate(V.count_text(w, two_line_count), (xx, 0), xytext=(0, -4 - nl * fs_name * 1.2 - 2), textcoords="offset points",
                    fontsize=fs_count, color=MUTE, ha="center", va="top", linespacing=1.05, annotation_clip=False)
    return fig


def desktop():
    return draw((8.8, 2.1), [0.06, 0.25, 0.93, 0.50], 9.0, 9.6, 7.8, 0.42, 12.0, 1, False, [r[0] for r in ROWS])


def mobile():
    return draw((3.9, 3.4), [0.11, 0.25, 0.87, 0.46], 7.2, 8.4, 6.4, 0.5, 11.0, 2, True,
                ["no search", "single\nwalk", "random-\nscore loop", "no-step\nloop", "DuROOM\nsearch"])


if __name__ == "__main__":
    for nm, fn in (("ways_of_asking_ladder_v1", desktop), ("ways_of_asking_ladder_v1-mobile", mobile)):
        assert os.environ.get("LADDER_ITER") or not (HERE / f"{nm}.svg").exists(), nm
        save(fn(), nm)
