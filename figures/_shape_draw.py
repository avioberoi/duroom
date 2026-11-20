"""Role markers shared by the drawn local-subspace figures (shape_walk, shape_orient): anchor, fit and test dots, left-out crosses.
Everything drawn here is schematic."""
from _style import *
import numpy as np
from matplotlib.patches import Circle, Arc

DOT = 46          # marker area for fit / test dots on the wide canvas


def blank(ax, xlim, ylim):
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect('equal', adjustable='box', anchor='NW')
    ax.axis('off')


def anchor(ax, p, size=84):
    ax.scatter(*p, s=size, color=ACCENT, edgecolors='none', zorder=7)


def fit_dots(ax, p, size=DOT):
    ax.scatter(p[:, 0], p[:, 1], s=size, color=INK, edgecolors='none', zorder=5)


def test_dots(ax, p, size=DOT):
    ax.scatter(p[:, 0], p[:, 1], s=size, facecolors=PAPER, edgecolors=INK, linewidths=1.1, zorder=5)


def pool_dots(ax, p, size=DOT*.5, color=PALE):
    ax.scatter(p[:, 0], p[:, 1], s=size, color=color, edgecolors='none', zorder=4)


def left_out(ax, p, size=60):
    ax.scatter(p[:, 0], p[:, 1], s=size, marker='x', color=MUTE, linewidths=1.3, zorder=5)


def ring(ax, centre, r, color=MUTE, lw=1.1, dashed=True, z=3):
    ax.add_patch(Circle(centre, r, facecolor='none', edgecolor=color, lw=lw,
                        linestyle=(0, (3.5, 2.5)) if dashed else '-', zorder=z))


def sphere_arc(ax, centre=(0, -3.2), R=3.6, t0=62, t1=118):
    """One faint arc: a hint that the plane is a patch of a sphere."""
    ax.add_patch(Arc(centre, 2*R, 2*R, theta1=t0, theta2=t1, color=HAIR, lw=1.0, zorder=0))


def mean_tick(ax, p, size=13):
    ax.plot(*p, marker='+', color=INK, ms=size, mew=1.5, zorder=8)


def key_marker(fig, x, y, kind, color=INK, ms=6.5):
    """Marker in figure-fraction coordinates, for key rows."""
    kw = dict(linestyle='none', transform=fig.transFigure)
    if kind == 'dot':
        fig.add_artist(plt.Line2D([x], [y], marker='o', ms=ms, color=color, **kw))
    elif kind == 'hollow':
        fig.add_artist(plt.Line2D([x], [y], marker='o', ms=ms, mfc=PAPER, mec=color, mew=1.2, **kw))
    elif kind == 'cross':
        fig.add_artist(plt.Line2D([x], [y], marker='x', ms=ms, mew=1.4, color=color, **kw))
    elif kind == 'plus':
        fig.add_artist(plt.Line2D([x], [y], marker='+', ms=ms + 3, mew=1.6, color=color, **kw))


def key_line(fig, x0, x1, y, color=INK, lw=1.4, dashed=False):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=color, lw=lw,
                              linestyle=(0, (3.5, 2.5)) if dashed else '-', transform=fig.transFigure))


def head(fig, title, sub):
    """Title and the 'Drawn, not measured' line, positioned in inches from the top."""
    H = fig.get_size_inches()[1]
    fig.text(.02, 1 - .10/H, title, fontsize=17, va='top')
    fig.text(.02, 1 - .52/H, sub, fontsize=12, color=MUTE, va='top')
