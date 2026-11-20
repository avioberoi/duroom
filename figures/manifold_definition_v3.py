"""Drawn definition of support and neighbour spacing: a cone, a lumpy region and spacing rings. Positions are illustrative, not corpus statistics.
Usage: python manifold_definition_v3.py"""
from _style import *
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon


def project(xyz):
    x, y, z = np.asarray(xyz).T
    elevation = .43
    return np.column_stack((x, z*np.cos(elevation)-y*np.sin(elevation),
                            y*np.cos(elevation)+z*np.sin(elevation)))


def ring(theta):
    phi = np.linspace(0, 2*np.pi, 240)
    return np.column_stack((np.sin(theta)*np.cos(phi),
                            np.sin(theta)*np.sin(phi), np.full_like(phi, np.cos(theta))))


def sphere(ax):
    ax.set_aspect('equal')
    ax.set(xlim=(-1.35, 1.35), ylim=(-1.3, 1.62))
    ax.axis('off')
    ax.set_title('The unit sphere', loc='left', fontsize=13, pad=12)
    ax.add_patch(Circle((0, 0), 1, facecolor='none', edgecolor=HAIR, lw=1))
    phi = np.linspace(0, 2*np.pi, 240)
    for xyz in [ring(np.pi/2), np.column_stack((np.zeros_like(phi), np.sin(phi), np.cos(phi)))]:
        xy = project(xyz)
        ax.plot(*xy[:, :2].T, color=HAIR, lw=.7)

    # A cone and a lumpy band of drawn unit vectors, never a reconstruction.
    cap = project(ring(.99))
    ax.add_patch(Polygon([(0, 0), (-.84, .5), (.84, .5)],
                         facecolor=DEEP, alpha=.55, edgecolor='none', zorder=0))
    ax.plot(*cap[:, :2].T, color=PALE, lw=.8)
    for end in [(-.84, .5), (.84, .5)]:
        ax.plot([0, end[0]], [0, end[1]], color=PALE, lw=.8)
    rng = np.random.default_rng(27)
    points = []
    for azimuth, polar, width in zip(
        [.1, .85, 1.7, 2.6, 3.3, 4.2, 5.1, 5.8],
        [.69, .78, .58, .83, .7, .85, .63, .81],
        [.045, .09, .05, .08, .055, .07, .085, .04],
    ):
        a = rng.normal(azimuth, .20, 55)
        t = np.clip(rng.normal(polar, width, len(a)), .40, .94)
        points.extend(np.column_stack((np.sin(t)*np.cos(a), np.sin(t)*np.sin(a), np.cos(t))))
    xy = project(points)
    xy = xy[np.argsort(xy[:, 2])]
    ax.scatter(xy[:, 0], xy[:, 1], s=5, color=MUTE, alpha=.65, edgecolors='none')
    ax.scatter([0], [0], s=9, color=MUTE)
    ax.annotate('Works in one cone', xy=(.48, .79), xytext=(0, 1.36), ha='center',
                fontsize=11, arrowprops={'arrowstyle':'-', 'color':MUTE, 'lw':.7})
    # Locator: a magnification, not a second measured embedding.
    ax.add_patch(Rectangle((.20, .515), .46, .33, facecolor='none',
                           edgecolor=INK, lw=.9))


def local(ax, mobile):
    ax.set_aspect('equal')
    ax.set(xlim=(-.6, 9.5), ylim=(-1.2, 7.8))
    ax.axis('off')
    ax.set_title('Real works, up close', loc='left', fontsize=13, pad=12)
    rng = np.random.default_rng(64)
    centres = np.array([[1.0, 2.2], [4.2, 3.1], [7.6, 3.7]])
    points = []
    for i, centre in enumerate(centres):
        # Three distinct dense clumps with fuzzy, finite-thickness edges.
        cloud = rng.normal(size=(115, 2)) * [[.62, .41], [.67, .5], [.69, .45]][i]
        cloud[:, 1] += cloud[:, 0] * [.25, -.28, .25][i]
        cloud += centre
        points.extend(cloud)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=36, c=PALE, alpha=.15, edgecolors='none', zorder=1)
    # Sparse narrow stretches connect the clumps, without a smooth boundary.
    for a, b in [(np.array([1.7, 2.35]), np.array([3.5, 3.0])),
                 (np.array([4.9, 3.0]), np.array([6.9, 3.55]))]:
        t = rng.uniform(0, 1, (30, 1))
        bridge = a + t*(b-a) + rng.normal(size=(30, 2))*[.12, .12]
        points.extend(bridge)
    points = np.asarray(points)
    ax.scatter(points[:, 0], points[:, 1], s=7, c=MUTE, alpha=.75, edgecolors='none', zorder=2)
    radius = .7
    on = np.array([3.6, 2.6])
    off = np.array([6.2, 6.15])
    on_nearest = points[np.linalg.norm(points-on, axis=1).argmin()]
    off_nearest = points[np.linalg.norm(points-off, axis=1).argmin()]
    assert np.linalg.norm(on_nearest-on) < radius
    assert np.linalg.norm(off_nearest-off) > radius
    for query, nearest, marker in [(on, on_nearest, 'o'), (off, off_nearest, 'x')]:
        ax.add_patch(Circle(query, radius, facecolor='none', edgecolor=ACCENT,
                            lw=1, linestyle=(0, (3, 2)), zorder=4))
        ax.plot([query[0], nearest[0]], [query[1], nearest[1]], color=ACCENT, lw=.9, zorder=3)
        ax.scatter(*query, s=40, marker=marker, color=ACCENT, zorder=5)
        ax.scatter(*nearest, s=20, facecolors=PAPER, edgecolors=INK, linewidths=.7, zorder=4)
    # Labels live outside the cloud; leaders terminate at the spacing rings.
    ax.annotate('A work within one spacing', xy=(on[0], on[1]-radius), xytext=(on[0], .1),
                ha='center', va='top', fontsize=11, color=ACCENT,
                arrowprops={'arrowstyle':'-', 'color':ACCENT, 'lw':.7})
    ax.text(off[0]+1.0, off[1], 'No work within\none spacing', color=ACCENT, fontsize=11, va='center', linespacing=1.25)
    ax.text(0, 7.4, 'The region occupied by real works', fontsize=10, color=MUTE)
    ax.text(0, -1.03, 'Dots stand for real works.\nOther directions are omitted.' if mobile else
            'Dots stand for real works; other directions are omitted.',
            fontsize=10, color=MUTE, va='top')


def draw(mobile=False):
    fig = plt.figure(figsize=(4.3, 9.0) if mobile else (9.2, 5.6))
    if mobile:
        left = fig.add_axes([.08, .53, .84, .30])
        right = fig.add_axes([.07, .14, .89, .35])
    else:
        left = fig.add_axes([.015, .22, .32, .62])
        right = fig.add_axes([.39, .14, .59, .70])
    sphere(left)
    local(right, mobile)
    fig.text(.025, .98, 'The manifold: the art we have', fontsize=16, va='top')
    fig.text(.025, .92 if not mobile else .935,
             'Drawn explanation, not measured geometry.' if not mobile else
             'Drawn explanation, not measured.', fontsize=11, color=ACCENT, va='top')
    fig.text(.025, .07 if not mobile else .08,
             'Dashed rings: one spacing, drawn to illustrate the idea.' if not mobile else
             'Dashed rings: one spacing, drawn\nto illustrate the idea.', fontsize=10, color=MUTE, va='top')
    if not mobile:
        fig.text(.345, .52, 'Closer\nview', fontsize=10, color=MUTE, ha='center')
    save(fig, 'manifold_definition_v3' + ('-mobile' if mobile else ''))


if __name__ == '__main__':
    draw()
    draw(mobile=True)
