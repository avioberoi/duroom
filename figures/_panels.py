"""Small layout helpers; all type and colour tokens come from _style."""
import numpy as np
from _style import *


def clean_axes(ax, mobile=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIR)
    ax.tick_params(length=3, width=0.6, labelsize=11 if mobile else 10, colors=INK)
    ax.minorticks_off()


def panel_label(ax, text, mobile=False):
    ax.text(0, 1.10, text, transform=ax.transAxes, fontsize=12,
            ha="left", va="bottom", color=INK, clip_on=False)


def points(value):
    return f"{value:+.1f}".replace("-", "−")


def finish(fig, name, mobile):
    save(fig, name + ("-mobile" if mobile else ""))
