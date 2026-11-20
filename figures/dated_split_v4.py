"""Where the far and faithful pictures sit among real works by date, per loop, in four parts by the dates of the real works within the manifold line; reads data/dated_split.csv.
Usage: python dated_split_v4.py"""
import csv

from _panels import *
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

MID = "#A99E8C"
ROWS = list(csv.DictReader(open(HERE / "data" / "dated_split.csv", encoding="utf-8")))
LOOPS = [("search", "the search"), ("random-score loop", "the random-score loop"), ("no-step loop", "the no-step loop"),
         ("single walk", "the single walk")]
REST = ("rest", "not split into parts (counted only in the accent part)", DEEP, MUTE)
PARTS = [("pre1900_near", "a work dated before 1900 within the manifold line", MID, INK),
         ("later_only_near", "none dated before 1900 within the manifold line, a work dated 1900 or later within it", ACCENT, PAPER),
         ("undated_only", "only an undated work within the manifold line", PALE, INK),
         ("none", "no real work within the manifold line", INK, PAPER)]
D = {}
for r in ROWS:
    if r["dating"] != "own":
        continue
    v = {k: int(r[k]) if r[k] != "" else None for k, *_ in PARTS}
    n = int(r["far_and_faithful"])
    if None in v.values():   # only the walk's large-leftover row, where the accent part alone was counted
        assert r["loop"] == "single walk" and r["subset"] == "large_leftover" and v["later_only_near"] is not None, r
        v = {k: x for k, x in v.items() if x is not None}
        v["rest"] = n - sum(v.values())
    assert sum(v.values()) == n, r
    D[r["subset"], r["loop"]] = (n, v)
PANELS = [("all", "All far and faithful pictures"), ("large_leftover", "Those with a large leftover share")]


def share_text(k, n):
    s = 100 * k / n
    return f"{s:.1f}" if s < 10 else f"{s:.0f}"


def pct_text(k, n):
    """0 prints 0%; under 10 one decimal; 10 or more a whole number."""
    if k == 0:
        return "0%"
    s = 100 * k / n
    return f"{s:.1f}%" if round(s, 1) < 10 else f"{s:.0f}%"


def text_w(s, fs, bold=False):
    """Width of a string in points."""
    w, _, _ = TextToPath().get_text_width_height_descent(s, FontProperties(family=plt.rcParams["font.family"], size=fs,
                                                                           weight="bold" if bold else "normal"), False)
    return w


def two_part(ax, x, y, main, cnt, fs, fs_cnt, ipu, ha, va, col, cnt_col, bold=False, zorder=4, bbox=None):
    """Main label and its small count side by side, anchored at x by ha; widths measured, in data units."""
    gap_in = 0.045
    wm = text_w(main, fs, bold) / 72 / ipu
    wc = text_w(cnt, fs_cnt) / 72 / ipu
    tot = wm + gap_in / ipu + wc
    x0 = x - tot / 2 if ha == "center" else (x if ha == "left" else x - tot)
    ax.text(x0, y, main, fontsize=fs, color=col, ha="left", va=va, zorder=zorder, fontweight="bold" if bold else "normal",
            bbox=bbox)
    ax.text(x0 + wm + gap_in / ipu, y, cnt, fontsize=fs_cnt, color=cnt_col, ha="left", va=va, zorder=zorder, bbox=bbox)
    return tot


def draw_bar(ax, y, n, v, h, fs, inch_per_unit, fat):
    """One bar at height y, in percent of the loop's pictures; share and small count inside each part when they fit, else outside, alternating above/below."""
    x, outside = 0.0, 0
    fc = fs - 2.2   # the small count
    for key, _, col, txt in PARTS + [REST]:
        if key not in v:
            continue
        k = v[key]
        w = 100 * k / n
        if k:
            ax.barh(y, w, left=x, height=h, color=col, edgecolor=PAPER if key != "rest" else MID, linewidth=1.0, zorder=3,
                    hatch="..." if key == "rest" else None)
        main = pct_text(k, n) + (" not split" if key == "rest" else "")
        cnt = f"{k:,}"
        need = (text_w(main, fs, key == "later_only_near") + text_w(cnt, fc) + 0.045 * 72 + 6) / 72 / inch_per_unit
        if k and w >= need:
            two_part(ax, x + w / 2, y, main, cnt, fs, fc, inch_per_unit, "center", "center",
                     txt if key != "rest" else INK, txt if key != "rest" else INK, bold=key == "later_only_near",
                     bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.2) if key == "rest" else None)   # a paper patch behind the hatched part's label
        else:
            above = outside % 2 == 0
            xx = min(max(x + w / 2, 1.5), 98.5)
            ax.plot([xx, xx], [y - h / 2, y - h / 2 - 0.12] if above else [y + h / 2, y + h / 2 + 0.12], color=MUTE, lw=0.6,
                    zorder=4)
            two_part(ax, xx, y - h / 2 - 0.15 if above else y + h / 2 + 0.15, main, cnt, fs - 0.6, fc - 0.6, inch_per_unit,
                     "center" if 3 < xx < 97 else ("left" if xx <= 3 else "right"), "bottom" if above else "top", INK, MUTE)
            outside += 1
        x += w
    return x


def key(fig, spots, fs):
    """The four parts, as swatches with their words, at the given figure positions (two by two on desktop)."""
    for (x, y), (_, label, col, _) in zip(spots, PARTS):
        fig.patches.append(plt.Rectangle((x, y - 0.006), 0.022, 0.012, transform=fig.transFigure, color=col, figure=fig))
        fig.text(x + 0.03, y, label, fontsize=fs, color=INK, ha="left", va="center")


def desktop():
    fig = plt.figure(figsize=(8.8, 7.6), dpi=200)
    fig.text(0.0, 0.996, "Where the far and faithful pictures sit among real works, by date", fontsize=12.5, color=INK,
             ha="left", va="top")
    fig.text(0.0, 0.960, "Each loop's far and faithful pictures, split by the real works within the manifold line of them, each work "
             "dated by its own year.\nEach part is a percent of its loop's far and faithful pictures; the count is printed small.", fontsize=9.3, color=MUTE, ha="left", va="top",
             linespacing=1.25)
    key(fig, [(0.0, 0.886), (0.0, 0.857), (0.6, 0.886), (0.6, 0.857)], 9.3)
    axes = [fig.add_axes([0.2, 0.468, 0.785, 0.32]), fig.add_axes([0.2, 0.061, 0.785, 0.32])]
    fig.text(0.2, 0.012, "The walk's second strip: only the accent part was counted; the rest is not split into parts.",
             fontsize=8.6, color=MUTE, ha="left", va="bottom")
    for ax, (sub, title) in zip(axes, PANELS):
        ax.set_xlim(0, 100); ax.set_ylim(3.75, -0.75); ax.axis("off")
        ax.text(-25.5, -0.95, title, fontsize=10.5, color=INK, fontweight="bold", ha="left", va="bottom", clip_on=False)
        ipu = 0.785 * 8.8 / 100
        for i, (loop, name) in enumerate(LOOPS):
            n, v = D[sub, loop]
            ax.text(-1.5, i, f"{name}\n{n:,} pictures", fontsize=9.6, color=INK, ha="right", va="center",
                    linespacing=1.2, clip_on=False)
            draw_bar(ax, i, n, v, 0.5, 9, ipu, True)
    return fig


def mobile():
    fig = plt.figure(figsize=(3.9, 11.6), dpi=200)
    fig.text(0.0, 0.9975, "Where the far and faithful pictures\nsit among real works, by date", fontsize=12.5, color=INK,
             ha="left", va="top", linespacing=1.15)
    fig.text(0.0, 0.957, "Each loop's far and faithful pictures, split\nby the real works within the manifold line of them, each\n"
             "work dated by its own year. Each part is a percent\nof its loop's far and faithful pictures; count small.", fontsize=9.5, color=MUTE, ha="left", va="top",
             linespacing=1.2)
    y = 0.872
    for _, label, col, _ in PARTS:
        lines = label.replace(", a work", ",\na work") if len(label) > 40 else label
        fig.patches.append(plt.Rectangle((0.0, y - 0.0033), 0.05, 0.0066, transform=fig.transFigure, color=col, figure=fig))
        fig.text(0.07, y, lines, fontsize=9.5, color=INK, ha="left", va="center", linespacing=1.15)
        y -= 0.0215 + (0.0133 if "\n" in lines else 0)
    axes = [fig.add_axes([0.02, 0.43, 0.96, 0.335]), fig.add_axes([0.02, 0.05, 0.96, 0.335])]
    fig.text(0.0, 0.004, "The walk's second strip: only the accent part\nwas counted; the rest is not split into parts.",
             fontsize=8.8, color=MUTE, ha="left", va="bottom", linespacing=1.15)
    for ax, (sub, title) in zip(axes, PANELS):
        ax.set_xlim(0, 100); ax.set_ylim(8.0, -1.0); ax.axis("off")
        ax.text(0, -0.95, title, fontsize=10.5, color=INK, fontweight="bold", ha="left", va="bottom", clip_on=False)
        ipu = 0.96 * 3.9 / 100
        for i, (loop, name) in enumerate(LOOPS):
            n, v = D[sub, loop]
            yy = i * 2.25 + 0.75
            ax.text(0, yy - 0.88,f"{name}, {n:,} pictures", fontsize=9.6, color=INK, ha="left", va="bottom")
            draw_bar(ax, yy, n, v, 0.55, 9, ipu, False)
    return fig


if __name__ == "__main__":
    finish(desktop(), "dated_split_v4", False)
    finish(mobile(), "dated_split_v4", True)
