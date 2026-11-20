"""Dot-chart panel of hits per start for the search and its two controls; drawing helper of q10_confirm_v4. Reads data/q10_confirm.json.
Usage: imported by q10_confirm_v4.py"""
import textwrap
from matplotlib.ticker import MaxNLocator
from _style import *

D = load_data("q10_confirm")
RENAME = {"random scores with the search's step sizes": "the random-score loop", "round trips with no search": "the no-step loop"}
D["arms"] = {k: RENAME.get(v, v) for k, v in D["arms"].items()}
ARMS = D["arms"]
ORDER = ["V", "C1", "C2"]
STYLE = {"V": dict(color=ACCENT, fill=True, marker="o"), "C1": dict(color=INK, fill=False, marker="s"), "C2": dict(color=INK, fill=False, marker="^")}
FS = 10.0


def values(cell):
    """The three counts of a cell, or None if any is still null."""
    v = [cell[a] for a in ORDER]
    return None if any(x is None for x in v) else v


def panel(ax, axt, key, title, mobile, labels):
    """Five start rows on ax (own scale); the 'All five' totals row on axt (own, wider scale, so the totals do not squash the rows)."""
    rows = [(s["label"], values(s[key])) for s in D["starts"]]
    total = values(D["totals"][key])
    known = [x for _, v in rows if v for x in v]
    xmax = max(max(known) * 1.12, 10) if known else 40
    off = (0.32, 0.0, -0.32)       # vertical offsets so equal or close counts never hide one another
    ax.set_xlim(-xmax * 0.03, xmax)
    ax.set_ylim(4.6, -0.6)
    for y, (_, v) in zip(range(5), rows):
        ax.plot([0, xmax], [y, y], color=HAIR, lw=0.5, ls=(0, (1, 2)), zorder=1)
        if v is None:
            ax.text(xmax / 2, y, "pending", fontsize=FS, color=MUTE, style="italic", ha="center", va="center",
                    bbox=dict(boxstyle="square,pad=0.25", fc=PAPER, ec=HAIR, lw=0.5), zorder=5)
            continue
        ax.plot([min(v), max(v)], [y, y], color=INK, lw=0.7, zorder=2)
        for a, x, dy in zip(ORDER, v, off):
            st = STYLE[a]
            ax.scatter([x], [y + dy], s=46, marker=st["marker"], facecolor=st["color"] if st["fill"] else PAPER,
                       edgecolor=st["color"], linewidth=1.2, zorder=4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_yticks(range(5))
    ax.set_yticklabels([r[0] for r in rows] if labels else [""] * 5, fontsize=FS)
    ax.grid(axis="x", color=HAIR, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=3, width=0.6, labelsize=FS, colors=INK)
    ax.set_title(chr(10).join(textwrap.wrap(title, 30 if mobile else 36)), fontsize=FS + 1, loc="left", color=INK, pad=8)
    tmax = max(total) * 1.18 if total else 40
    axt.set_xlim(-tmax * 0.03, tmax)
    axt.set_ylim(1.3, -1.3)
    axt.plot([0, tmax], [0, 0], color=HAIR, lw=0.5, ls=(0, (1, 2)), zorder=1)
    if total is None:
        axt.text(tmax / 2, 0, "pending", fontsize=FS, color=MUTE, style="italic", ha="center", va="center",
                 bbox=dict(boxstyle="square,pad=0.25", fc=PAPER, ec=HAIR, lw=0.5), zorder=5)
    else:
        axt.plot([min(total), max(total)], [0, 0], color=INK, lw=0.7, zorder=2)
        for a, x, dy in zip(ORDER, total, (-0.62, 0.62, -0.62)):
            st = STYLE[a]
            axt.scatter([x], [0], s=46, marker=st["marker"], facecolor=st["color"] if st["fill"] else PAPER,
                        edgecolor=st["color"], linewidth=1.2, zorder=4)
            axt.text(x, dy, f"{x:,}", fontsize=FS, color=INK, ha="center", va="center", zorder=5)
    axt.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axt.set_yticks([0])
    axt.set_yticklabels(["All five"] if labels else [""], fontsize=FS)
    axt.grid(axis="x", color=HAIR, lw=0.5)
    axt.set_axisbelow(True)
    axt.tick_params(axis="x", length=3, width=0.6, labelsize=FS, colors=INK)
    for x_ in (ax, axt):
        x_.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            x_.spines[side].set_visible(False)
        x_.spines["bottom"].set_color(HAIR)
