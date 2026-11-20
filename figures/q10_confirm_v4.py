"""Distinct hits of the search and its two controls, by starting work and in total; draws q10_confirm_v3's panel.
Usage: python q10_confirm_v4.py"""
from matplotlib.lines import Line2D

import q10_confirm_v3 as q
from _style import *

T = q.D["totals"]["distinct_hits"]
assert q.ARMS == {"V": "the search", "C1": "the random-score loop", "C2": "the no-step loop"}, q.ARMS
TITLE = f"The search found {T['V']} distinct hits, the random-score loop {T['C1']}, the no-step loop {T['C2']}"
SUB = "Hits by starting work, 5 starts, and all 5 together"


def draw(mobile=False):
    if mobile:
        fig = plt.figure(figsize=(3.9, 6.4))
        gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.5], left=0.35, right=0.95, top=0.74, bottom=0.05, hspace=0.32)
    else:
        fig = plt.figure(figsize=(7.2, 4.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.5], left=0.2, right=0.98, top=0.77, bottom=0.06, hspace=0.45)
    a1, t1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    q.panel(a1, t1, "distinct_hits", "", mobile, True)
    a1.set_title("")
    t1.set_yticklabels(["All 5"], fontsize=q.FS)
    fig.text(0.01, 0.995, "The search found " + str(T["V"]) + " distinct hits,\nthe random-score loop " + str(T["C1"])
             + ", the no-step loop " + str(T["C2"]) if mobile else TITLE, fontsize=12 if not mobile else 11.5, color=INK,
             ha="left", va="top", linespacing=1.15)
    fig.text(0.01, 0.905 if mobile else 0.93, SUB, fontsize=q.FS, color=MUTE, ha="left", va="top")
    handles = [Line2D([], [], color=q.STYLE[a]["color"], marker=q.STYLE[a]["marker"], ls="", markersize=7,
                      markeredgewidth=1.2, markerfacecolor=q.STYLE[a]["color"] if q.STYLE[a]["fill"] else PAPER,
                      label=q.ARMS[a]) for a in q.ORDER]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.875 if mobile else 0.875),
               ncol=1 if mobile else 3, frameon=False, fontsize=q.FS, handletextpad=0.4, columnspacing=1.6)
    save(fig, "q10_confirm_v4" + ("-mobile" if mobile else ""))


if __name__ == "__main__":
    draw(False)
    draw(True)
