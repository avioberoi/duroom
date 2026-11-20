"""Direction-response charts (how far the picture moved, how close to the request) from data/search_walk.json: colour is the direction's family, dashes tell members apart, each line is named at its end.
Usage: python search_response_v2.py"""
from search_figures import *

FAMILY = {"after": ACCENT, "past": "#2a6fc9", "random": "#178a5f"}
CURVES = [("shift", "after-1900 shift", "after", "-"), ("axis1", "new axis 1", "after", "--"),
          ("axis2", "new axis 2", "after", ":"), ("past_rank1", "basis direction 1", "past", "-"),
          ("past_rank50", "basis direction 50", "past", "--"), ("random1", "random direction", "random", "-")]
D = data("walk")
BYKEY = {c["key"]: c for c in D["curves"]}
assert all(k in BYKEY for k, *_ in CURVES)


def end_labels(ax, ends, gap, fs, x_end=50):
    """Name each line at its right end; labels too close are spread apart (by at least gap) with a thin leader."""
    ends = sorted(ends, key=lambda e: e[0])
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = (sum(e[0] for e in ends) - sum(ys)) / len(ys)          # recentre the spread block on the data
    ys = [y + shift for y in ys]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (y0, label, col), y in zip(ends, ys):
        ax.plot([x_end + 0.6, x_end + 2.4], [y0, y], color=col, lw=0.6, clip_on=False)
        ax.text(x_end + 2.9, y, label, fontsize=fs, color=INK, va="center", ha="left", clip_on=False)


def draw(ax, key, fs, gap):
    ends = []
    for k, label, fam, ls in CURVES:
        pts = BYKEY[k]["points"]
        x = [p["angle"] for p in pts]; y = [p[key] for p in pts]
        ax.plot(x, y, color=FAMILY[fam], ls=ls, lw=1.6 if k == "shift" else 1.3, zorder=3, dash_capstyle="butt")
        ends.append((y[-1], label, FAMILY[fam]))
    end_labels(ax, ends, gap, fs)
    ax.set_xlim(0, 50); ax.set_xticks([0, 10, 20, 30, 40, 50])


def family_key(fig, x, y, fs, widen=1.0):
    """One line naming the three colours: colour is the family, dashes tell members apart (widen: narrow figures)."""
    for name, fam in (("after 1900", "after"), ("the basis from before 1900", "past"), ("random", "random")):
        fig.add_artist(plt.Line2D([x, x + 0.03 * widen], [y, y], color=FAMILY[fam], lw=2, transform=fig.transFigure))
        fig.text(x + 0.037 * widen, y, name, fontsize=fs, color=INK, va="center", ha="left")
        x += (0.037 + 0.0105 * len(name) * fs / 9 + 0.03) * widen


def response(mobile, name):
    if mobile:
        fig, axs = plt.subplots(2, 1, figsize=(4.6, 7.0)); fig.subplots_adjust(left=0.15, right=0.6, top=0.855, bottom=0.08, hspace=0.45)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(8.8, 4.1)); fig.subplots_adjust(left=0.07, right=0.8, top=0.8, bottom=0.14, wspace=0.62)
    fs = 8.6
    for ax, key, heading in zip(axs, ["achieved", "chord"], ["How far the picture moved", "How close to the request"]):
        axes(ax, "Requested angle (°)", "Moved along request (°)" if key == "achieved" else "Chord to request")
        title(ax, heading)
        if key == "achieved":
            ax.plot([0, 50], [0, 50], ls="--", lw=0.7, color=HAIR, zorder=1); ax.set_ylim(-10, 55)
            ax.text(33, 40, "as asked", fontsize=7.5, color=MUTE, rotation=0)
            draw(ax, key, fs, 3.6)
        else:
            ax.set_ylim(0.4, 1.1)
            draw(ax, key, fs, 0.046)
    sub = "Colour is the direction's family; dashes tell its members apart. Medians over six starts."
    fig.text(0.0, 0.985, sub.replace("apart. ", "apart.\n") if mobile else sub, fontsize=8.6, color=MUTE, ha="left",
             va="top", linespacing=1.2)
    family_key(fig, 0.0, 0.925 if not mobile else 0.926, 8.6, 1.55 if mobile else 1.0)
    save(fig, name)


if __name__ == "__main__":
    for fn, nm in ((response, "search_response_v2"),):
        fn(False, nm); fn(True, nm + "-mobile")
