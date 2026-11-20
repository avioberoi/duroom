"""Every kind of picture on two rulers (gap to the request, far score), cut by the two bars into four cells; reads data/four_cells.json.
Usage: python four_cells_v4.py"""
import textwrap

from _panels import *

MID = "#A99E8C"
d = load_data("four_cells")
FID = d["faithful_bar_gap"]
LD, LS = d["far_levels_p99"]["dinov3"], d["far_levels_p99"]["siglip"]
PANELS = [  # key, name, colour
    ("p2", "real works before 1900, repainted", PALE),
    ("pos", "real works 1900 to 1969, repainted", MID),
    ("today1970", "real works 1970 on, repainted", INK),
    ("rand_cone", "random requests, inside the cone", MUTE),
    ("rand_sphere", "random requests, anywhere on the sphere", MUTE),
    ("noise", "requests read from noise images", MUTE),
    ("free", "the free search, 3 runs", MUTE),
    ("reanchor", "DuROOM search", ACCENT),
]
PTS, CELLS = {}, {}
for key, _, _ in PANELS:
    a = np.array(d["points"][key])
    x, y = a[:, 0], np.minimum(a[:, 1] / LD, a[:, 2] / LS)
    t = d["table"][key]
    far, fa = (a[:, 1] > LD) & (a[:, 2] > LS), x <= FID
    assert np.array_equal(far, y > 1)
    if not d["sampled"][key]:
        assert len(a) == t["n"] and [int(np.sum(far & fa)), int(np.sum(far & ~fa)), int(np.sum(~far & fa)),
                                     int(np.sum(~far & ~fa))] == t["FF_FN_NF_NN"], key
    else:
        assert key == "reanchor"
    PTS[key], CELLS[key] = (x, y), t["FF_FN_NF_NN"]
X0, X1, Y0, Y1 = 0.25, 1.52, 0.38, 1.37
DASH = (0, (4, 3))
CORNER = ["far, faithful", "far, not faithful", "near, faithful", "near, not faithful"]


def pct(c, n):
    """Share of n as text: 0 -> 0%, under 10 one decimal, 10 or more a whole number."""
    v = 100 * c / n
    if c == 0:
        return "0%"
    if v < 0.05:
        return "<0.1%"
    return f"{v:.1f}%" if round(v, 1) < 10 else f"{v:.0f}%"


def corner(ax, px, py, ha, va, head, c, n, size, colour):
    """A cell's share of the panel's n as the main label, its count small and muted next to it (toward the panel's middle)."""
    box = dict(boxstyle="square,pad=0.08", fc=PAPER, ec="none", alpha=0.85)
    ax.text(px, py, pct(c, n), transform=ax.transAxes, fontsize=size, ha=ha, va=va, zorder=4,
            color=colour, bbox=box)
    dy = size * 1.3
    ax.annotate(f"{head}, {c:,}" if head else f"{c:,}", (px, py), xycoords="axes fraction", xytext=(0, -dy if va == "top" else dy), textcoords="offset points",
                fontsize=size - 2.5, ha=ha, va=va, zorder=4, color=MUTE, bbox=box)


def panel(ax, key, name, col, fs, first, wrap):
    x, y = PTS[key]
    edge = dict(edgecolor=MUTE, linewidths=0.2) if col == PALE else dict(edgecolor="none")
    ax.scatter(x, y, s=2.2 if len(x) > 900 else 4, color=col, alpha=0.55 if len(x) > 900 else 0.8, rasterized=True, zorder=2, **edge)
    ax.axvline(FID, color=MUTE, lw=0.7, ls=DASH, zorder=1)
    ax.axhline(1.0, color=MUTE, lw=0.7, ls=DASH, zorder=1)
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
    clean_axes(ax)
    ax.tick_params(labelsize=fs - 1)
    ax.set_xticks([0.4, 0.8, 1.2]); ax.set_yticks([0.6, 0.8, 1.0, 1.2])
    n = d["table"][key]["n"]
    shown = f"{n:,}" if len(x) == n else f"{n:,}; {len(x):,} drawn"
    ax.set_title("\n".join(textwrap.wrap(f"{name} ({shown})", wrap)), loc="left", fontsize=fs, color=INK, pad=4, linespacing=1.1)
    pos = [(0.03, 0.97, "left", "top"), (0.97, 0.97, "right", "top"), (0.03, 0.03, "left", "bottom"), (0.97, 0.03, "right", "bottom")]
    for (px, py, ha, va), c, word in zip(pos, CELLS[key], CORNER):
        corner(ax, px, py, ha, va, word if first else "", c, n, fs - 1 if first else fs, ACCENT if (col == ACCENT and c) else INK)


XLAB = "gap from the request to the reading of the picture (left of the dashed line: faithful)"
YLAB = "distance from the reference, over the far bar (above 1: far in both observers)"


def desktop():
    W, H = 8.8, 6.0
    fig, axes = plt.subplots(2, 4, figsize=(W, H), dpi=200, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.82, bottom=0.11, wspace=0.08, hspace=0.42)
    fig.text(0.0, 0.995, "Far or near, faithful or not", fontsize=12, color=INK, ha="left", va="top")
    fig.text(0.0, 0.955, "Each point is 1 picture. The dashed lines are the faithful line and the far bar; each cell shows its percent of the "
             "group's pictures, with the count small.", fontsize=9.5, color=MUTE, ha="left", va="top")
    for ax, (key, name, col) in zip(axes.flat, PANELS):
        panel(ax, key, name, col, 8.5, key == "p2", 30)
    fig.text(0.53, 0.015, XLAB, fontsize=9, color=INK, ha="center", va="bottom")
    fig.text(0.0, 0.46, "\n".join(textwrap.wrap(YLAB, 46)), fontsize=9, color=INK, ha="left", va="center", rotation=90,
             linespacing=1.15)
    return fig


def mobile():
    W, H = 3.9, 10.4
    fig, axes = plt.subplots(4, 2, figsize=(W, H), dpi=200, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.835, bottom=0.07, wspace=0.1, hspace=0.42)
    fig.text(0.0, 0.998, "Far or near, faithful or not", fontsize=12, color=INK, ha="left", va="top")
    fig.text(0.0, 0.975, "\n".join(textwrap.wrap("Each point is 1 picture. The dashed lines are the faithful line and the far bar; each cell shows its percent "
             "of the group's pictures, with the count small, from top left: far and faithful, far and not, near and faithful, near and not.",
             44)), fontsize=10, color=MUTE, ha="left", va="top", linespacing=1.2)
    for ax, (key, name, col) in zip(axes.flat, PANELS):
        panel(ax, key, name, col, 9, False, 26)
    fig.text(0.55, 0.0, "\n".join(textwrap.wrap(XLAB, 44)), fontsize=9.5, color=INK, ha="center", va="bottom", linespacing=1.15)
    fig.text(0.0, 0.48, YLAB, fontsize=9.5, color=INK, ha="left", va="center", rotation=90)
    return fig


finish(desktop(), "four_cells_v4", False)
finish(mobile(), "four_cells_v4", True)
