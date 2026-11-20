"""Far and faithful pictures by way of asking: the share of each way's pictures, with count over total on every mark; the data and checks of the vertical figures. Reads data/ways_of_asking.csv.
Usage: python ways_of_asking_v4.py"""
import csv

from _panels import *

MID = "#A99E8C"
ROWS = {r["way"]: r for r in csv.DictReader(open(HERE / "data" / "ways_of_asking.csv", encoding="utf-8"))}
num = lambda w, k: int(ROWS[w][k])
RECIPES = ["midpoints of two works", "between two regions' centres", "35-degree steps from a work", "lines past a work",
           "random requests inside the region", "random requests on the whole sphere"]
for k in ("n", "far", "far_and_faithful"):
    assert sum(num(w, k) for w in RECIPES) == num("all six recipes", k), k
LOOPS = ["no-step loop", "random-score loop", "search"]
REAL = ["real works before 1900 held out (repainted)", "real works of 1900 to 1969 (repainted)",
        "real works from 1970 on (repainted)"]
assert ROWS["single walk with selection"]["kind"] == "baseline"
for w, r in ROWS.items():
    if r["n"]:
        assert 0 <= num(w, "far_and_faithful") <= num(w, "far") <= num(w, "n"), w

# (kind, csv way or None, label, style); kind "head" is a group heading
LAYOUT = [("head", None, "No search: 6 kinds of request not aimed at a later work", None)]
SHOWN = {"midpoints of two works": "midpoints of 2 works",
         "between two regions' centres": "on the line through 2 regions' centres"}
LAYOUT += [("row", w, SHOWN.get(w, w), "small") for w in RECIPES]
LAYOUT += [("row", "all six recipes", "all 6 kinds", "total"),
           ("head", None, "Using later art: a ceiling, not a request from the past", None),
           ("row", "lines aimed at real later works", "lines aimed at real later works", "ceiling"),
           ("head", None, "A single walk with selection, 9,600 pictures", None),
           ("row", "single walk with selection", "a single walk with selection", "loop"),
           ("head", None, "DuROOM search and its 2 controls, 9,600 pictures each", None),
           ("row", "no-step loop", "the no-step loop", "loop"),
           ("row", "random-score loop", "the random-score loop", "loop"),
           ("row", "search", "the search", "search"),
           ("head", None, "Real works repainted, for scale", None),
           ("row", REAL[0], "works before 1900, held out", "real"),
           ("row", REAL[1], "works of 1900 to 1969", "real"),
           ("row", REAL[2], "works from 1970 on", "real")]
XMAX = 50
HALO = dict(zorder=5, bbox=dict(facecolor=PAPER, edgecolor="none", pad=0.8))   # paper behind printed text: no gridline through it


def share(w):
    return 100 * num(w, "far_and_faithful") / num(w, "n")


def value_text(w):
    k, n = num(w, "far_and_faithful"), num(w, "n")
    if k == 0:
        return f"0 of {n:,}"
    s = share(w)
    return f"{k:,} of {n:,} ({s:.1f} in 100)" if s < 10 else f"{k:,} of {n:,} ({s:.0f} in 100)"


def bar(ax, y, w, style, h):
    s = share(w)
    kw = dict(height=h, left=0, zorder=3, linewidth=0)
    if style == "small":
        ax.barh(y, s, color=MID, **kw)
    elif style == "total":
        ax.barh(y, s, color=INK, **kw)
    elif style == "ceiling":
        ax.barh(y, s, color=PAPER, edgecolor=INK, hatch="////", **{**kw, "linewidth": 0.8})
    elif style == "loop":
        ax.barh(y, s, color=INK, **kw)
    elif style == "search":
        ax.barh(y, s, color=ACCENT, **kw)
    elif style == "real":
        ax.barh(y, s, color=PAPER, edgecolor=MUTE, **{**kw, "linewidth": 1.0})
    if num(w, "far_and_faithful") == 0:   # a zero is a short tick on the baseline, so the row is not read as missing
        ax.plot([0, 0], [y - h / 2, y + h / 2], color=MID, lw=1.6, zorder=4, solid_capstyle="butt")
    return s


def axis(ax, top, bottom, fs):
    ax.set_xlim(0, XMAX * 1.62 if fs < 10 else XMAX * 1.0); ax.set_ylim(bottom, top)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ticks = [0, 10, 20, 30, 40, 50]
    for t in ticks:
        ax.plot([t, t], [bottom, top], color=HAIR, lw=0.6, zorder=1)
    ax.set_xticks(ticks); ax.tick_params(axis="x", length=0, labelsize=fs, colors=MUTE, pad=2)


def desktop():
    step = {"head": 1.25, "row": 1.0, "pending": 1.45}
    ys, y = [], 0.0
    for kind, *_ in LAYOUT:
        y += step[kind]; ys.append(y)
    fig = plt.figure(figsize=(8.8, 6.9), dpi=200)
    ax = fig.add_axes([0.335, 0.075, 0.655, 0.855])
    fig.text(0.0, 0.995, "Far and faithful pictures, by way of asking", fontsize=12.5, color=INK, ha="left", va="top")
    fig.text(0.0, 0.958, "The share of each way's pictures that are far and faithful, on one scale; the count is printed "
             "on every bar.", fontsize=9.5, color=MUTE, ha="left", va="top")
    for (kind, w, label, style), y in zip(LAYOUT, ys):
        if kind == "head":
            ax.text(-0.505 * XMAX, y + 0.12, label, fontsize=10, color=INK, ha="left", va="center", fontweight="bold",
                    clip_on=False, zorder=5, bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5))
            continue
        if kind == "pending":
            ax.add_patch(plt.Rectangle((0, y - 0.3), XMAX, 0.6, fill=False, edgecolor=MID, lw=0.8, ls=(0, (3, 2.5)),
                                       zorder=2))
            ax.text(XMAX / 2, y, "not run", fontsize=9, color=MUTE, ha="center", va="center", **HALO)
            ax.text(-0.02 * XMAX, y, label, fontsize=9.6, color=INK, ha="right", va="center", clip_on=False)
            continue
        small = style == "small"
        fs = 8.6 if small else 9.6
        ax.text(-0.02 * XMAX, y, label, fontsize=fs, color=MUTE if small else INK, ha="right", va="center", clip_on=False)
        s = bar(ax, y, w, style, 0.48 if small else 0.62)
        ax.text(s + 0.9, y, value_text(w), fontsize=fs, color=INK if not small else MUTE, ha="left", va="center", **HALO,
                fontweight="bold" if style in ("total", "search") else "normal")
    axis(ax, 0.75, ys[-1] + 0.7, 9)
    ax.set_xlim(0, XMAX * 1.62)
    ax.text(XMAX / 2, ys[-1] + 1.55, "far and faithful, in 100 pictures", fontsize=9, color=MUTE, ha="center", va="top",
            clip_on=False)
    return fig


def mobile():
    # phone: each row's label and count sit on one line above its bar
    step = {"head": 2.0, "row": 1.32, "pending": 1.9}
    ys, y = [], 0.0
    for kind, *_ in LAYOUT:
        y += step[kind]; ys.append(y)
    fig = plt.figure(figsize=(3.9, 10.6), dpi=200)
    ax = fig.add_axes([0.02, 0.045, 0.96, 0.855])
    fig.text(0.0, 0.997, "Far and faithful pictures,\nby way of asking", fontsize=12.5, color=INK, ha="left", va="top",
             linespacing=1.15)
    fig.text(0.0, 0.952, "The share of each way's pictures that are far\nand faithful, on one scale; the count is printed\n"
             "on every bar.", fontsize=9.5, color=MUTE, ha="left", va="top", linespacing=1.2)
    for (kind, w, label, style), y in zip(LAYOUT, ys):
        if kind == "head":
            ax.text(0, y - 0.3, label if len(label) < 40 else label.replace(": " if ": " in label else ", ", ":\n" if ": " in label else ",\n", 1), fontsize=10, color=INK,
                    ha="left", va="center", fontweight="bold", linespacing=1.1, zorder=5,
                    bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5))
            continue
        if kind == "pending":
            ax.add_patch(plt.Rectangle((0, y - 0.28), XMAX, 0.56, fill=False, edgecolor=MID, lw=0.8, ls=(0, (3, 2.5)),
                                       zorder=2))
            ax.text(XMAX / 2, y, "not run", fontsize=9, color=MUTE, ha="center", va="center", **HALO)
            ax.text(0, y - 0.43, label, fontsize=9.6, color=INK, ha="left", va="bottom", **HALO)
            continue
        small = style == "small"
        fs = 8.8 if small else 9.6
        ax.text(0, y - 0.38, label, fontsize=fs, color=MUTE if small else INK, ha="left", va="bottom", **HALO)
        s = bar(ax, y, w, style, 0.38 if small else 0.46)
        ax.text(s + 1.2, y, value_text(w), fontsize=fs, color=MUTE if small else INK, ha="left", va="center", **HALO,
                fontweight="bold" if style in ("total", "search") else "normal")
    axis(ax, 0.9, ys[-1] + 0.6, 9)
    ax.set_xlim(0, XMAX * 1.95)
    ax.text(XMAX / 2, ys[-1] + 1.45, "far and faithful, in 100 pictures", fontsize=9, color=MUTE, ha="center", va="top",
            clip_on=False)
    return fig


if __name__ == "__main__":
    finish(desktop(), "ways_of_asking_v4", False)
    finish(mobile(), "ways_of_asking_v4", True)
