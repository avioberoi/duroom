"""Drawn, not measured: comparing local directions at two works, 58 degrees apart on a sphere, with a flat patch at each and the angle between the patches.
Usage: python shape_orient.py"""
from _shape_draw import *
from matplotlib.patches import Polygon, FancyArrowPatch

OUT = 'shape_orient_s'
ALPHA_A, ALPHA_B = 18., 62.         # patch long-axis angle from the great-circle direction (degrees), drawn
SEP = 58.                            # degrees between the two works (the measured median)


def vec(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.array([np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)])


def view(P, tilt=-17., yaw=0.):
    """Orthographic view: rotate about x by tilt, then drop z. Returns (x, y, depth)."""
    t = np.deg2rad(tilt)
    R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    P = np.atleast_2d(P) @ R.T
    return P[:, 0], P[:, 1], P[:, 2]


def step(P, toward, deg):
    """Point `deg` degrees from P along the great circle towards `toward`."""
    t = toward - (toward @ P)*P
    t /= np.linalg.norm(t)
    d = np.deg2rad(deg)
    return np.cos(d)*P + np.sin(d)*t, -np.sin(d)*P + np.cos(d)*t     # new point, tangent there


def build():
    A = vec(-34, 14)
    B_target = vec(36, -14)
    # scale B to be exactly SEP degrees from A along the same great circle
    B, tB = step(A, B_target, SEP)
    t = B_target - (B_target @ A)*A
    tA = t/np.linalg.norm(t)
    assert abs(np.degrees(np.arccos(np.clip(A @ B, -1, 1))) - SEP) < 1e-6
    return A, B, tA, tB


def tangent_patch(P, t, alpha, a=.27, b=.15):
    """Corners of a small flat patch in P's tangent plane, long axis at `alpha` degrees from the geodesic direction t."""
    n = np.cross(P, t)
    al = np.deg2rad(alpha)
    d1 = np.cos(al)*t + np.sin(al)*n
    d2 = -np.sin(al)*t + np.cos(al)*n
    corners = [P + s1*a*d1 + s2*b*d2 for s1, s2 in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    return np.array(corners), P - a*d1, P + a*d1


def poly3(ax, corners3, color=INK, lw=1.3, dashed=False, z=6):
    x, y, _ = view(corners3)
    xy = np.column_stack((x, y))
    ax.add_patch(Polygon(xy, closed=True, facecolor='none', edgecolor=color, lw=lw,
                         linestyle=(0, (3, 2)) if dashed else '-', zorder=z))


def axis3(ax, p0, p1, color=INK, lw=2.6, dashed=False, z=7):
    x, y, _ = view(np.array([p0, p1]))
    ax.plot(x, y, color=color, lw=lw, linestyle=(0, (3, 2)) if dashed else '-', zorder=z, solid_capstyle='butt')


def rect2(c, a, b, ang):
    r = np.deg2rad(ang)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return np.array([[-a, -b], [a, -b], [a, b], [-a, b]]) @ R.T + c, R @ np.array([a, 0.])


def patch2(ax, c, a, b, ang, color=INK, lw=1.3, dashed=False, axis_lw=2.8, z=5):
    corners, half = rect2(np.asarray(c, float), a, b, ang)
    ax.add_patch(Polygon(corners, closed=True, facecolor='none', edgecolor=color, lw=lw,
                         linestyle=(0, (3.5, 2.5)) if dashed else '-', zorder=z))
    ax.plot([c[0] - half[0], c[0] + half[0]], [c[1] - half[1], c[1] + half[1]], color=color, lw=axis_lw,
            linestyle=(0, (3.5, 2.5)) if dashed else '-', zorder=z + 1, solid_capstyle='butt')


def sphere_panel(ax, A, B, tA, tB, fs):
    blank(ax, (-1.35, 1.35), (-1.12, 1.12))
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=MUTE, lw=1.0, zorder=1)
    # faint graticule: the equator, front half only
    x, y, z = view(np.column_stack((np.sin(th), np.zeros_like(th), np.cos(th))))
    m = z > 0
    ax.plot(np.where(m, x, np.nan), np.where(m, y, np.nan), color=HAIR, lw=.8, zorder=0)
    # great circle from A to B, with an arrowhead
    s = np.linspace(0, SEP, 80)
    path = np.array([step(A, B, d)[0] for d in s])
    x, y, _ = view(path)
    ax.plot(x[:-6], y[:-6], color=INK, lw=1.5, zorder=4)
    ax.add_patch(FancyArrowPatch((x[-8], y[-8]), (x[-1], y[-1]), arrowstyle='-|>', mutation_scale=13, color=INK,
                                 lw=1.5, zorder=4, shrinkA=0, shrinkB=0))
    # the two patches, one flat patch of local directions at each work
    cA, a0, a1 = tangent_patch(A, tA, ALPHA_A)
    cB, b0, b1 = tangent_patch(B, tB, ALPHA_B)
    poly3(ax, cA); axis3(ax, a0, a1)
    poly3(ax, cB); axis3(ax, b0, b1)
    xa, ya, _ = view(A); xb, yb, _ = view(B)
    ax.scatter(xa, ya, s=70, color=ACCENT, edgecolors='none', zorder=9)
    ax.scatter(xb, yb, s=40, color=INK, edgecolors='none', zorder=9)
    ax.annotate('the anchor', xy=(xa[0], ya[0]), xytext=(xa[0] - .02, ya[0] - .56), ha='center', va='center',
                fontsize=fs, color=ACCENT, arrowprops={'arrowstyle': '-', 'color': MUTE, 'lw': .8, 'shrinkA': 2, 'shrinkB': 6})
    ax.annotate('another work', xy=(xb[0], yb[0]), xytext=(xb[0] + .02, yb[0] - .56), ha='center', va='center',
                fontsize=fs, color=INK, arrowprops={'arrowstyle': '-', 'color': MUTE, 'lw': .8, 'shrinkA': 2, 'shrinkB': 6})
    # the "58 degrees" label goes to the free spot nearest the middle of the arc: wholly inside the sphere, and clear of
    # both patches, the arc and arrow, the equator and the two work labels (all checked, by box, below)
    mid = path[len(path)//2]
    xm, ym, _ = view(mid)
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    box = lambda t: inv.transform(np.array(t.get_window_extent(rend)).reshape(2, 2)) if False else \
        (lambda bb: (bb.x0, bb.y0, bb.x1, bb.y1))(t.get_window_extent(rend).transformed(inv))
    patch_boxes = []
    for corners in (cA, cB):
        px, py, _ = view(corners)
        patch_boxes.append((px.min() - .05, py.min() - .05, px.max() + .05, py.max() + .05))
    others = [box(t) for t in ax.texts] + [box(t) for t in ax.texts if False]
    arc = np.column_stack((x, y))
    eq_x, eq_y, eq_z = view(np.column_stack((np.sin(th), np.zeros_like(th), np.cos(th))))
    equator = np.column_stack((eq_x[eq_z > 0], eq_y[eq_z > 0]))

    def free(bx):
        x0, y0, x1, y1 = bx
        if max(np.hypot(cx, cy) for cx in (x0, x1) for cy in (y0, y1)) > .93:                     # wholly inside the sphere
            return False
        if any(x0 < q1 and x1 > q0 and y0 < q3 and y1 > q2 for q0, q2, q1, q3 in patch_boxes):   # clear of patches
            return False
        if any(x0 - .04 < u and u < x1 + .04 and y0 - .04 < v and v < y1 + .04 for u, v in np.vstack((arc, equator))):
            return False                                                                         # clear of arc, arrow, equator
        if any(x0 < q2 + .03 and x1 > q0 - .03 and y0 < q3 + .03 and y1 > q1 - .03 for q0, q1, q2, q3 in others):
            return False                                                                         # clear of the work labels
        return True

    best = None
    for text in ('about 58 degrees apart', 'about 58 degrees\napart', 'about 58\ndegrees apart'):
        for cx in np.arange(-.7, .75, .05):
            for cy in np.arange(-.9, .9, .04):
                t = ax.text(cx, cy, text, fontsize=fs, ha='center', va='center', color=INK, linespacing=1.15)
                bx = box(t)
                t.remove()
                if free(bx):
                    cost = np.hypot(cx - xm[0], cy - ym[0] - .3) + (0 if '\n' not in text else .08)
                    if best is None or cost < best[0]:
                        best = (cost, text, cx, cy)
    assert best is not None, 'no free spot for the 58-degree label'
    _, text, cx, cy = best
    lab = ax.text(cx, cy, text, fontsize=fs, ha='center', va='center', color=INK, linespacing=1.15)
    assert free(box(lab))


def zoom_panel(ax, fs):
    blank(ax, (-1.35, 1.35), (-1.1, 1.1))
    a, b = .78, .4
    patch2(ax, (0, 0), a, b, ALPHA_A, color=MUTE, dashed=True, lw=1.4, axis_lw=2.6)          # the anchor's, carried
    patch2(ax, (0, 0), a, b, ALPHA_B, color=INK, lw=1.4, axis_lw=3.0, z=7)                     # fitted at this work
    # angle between the two long axes
    r = .98
    t = np.deg2rad(np.linspace(ALPHA_A, ALPHA_B, 40))
    ax.plot(r*np.cos(t), r*np.sin(t), color=INK, lw=1.1, zorder=8)
    for ang in (ALPHA_A, ALPHA_B):
        ax.plot([.8*np.cos(np.deg2rad(ang)), 1.05*np.cos(np.deg2rad(ang))], [.8*np.sin(np.deg2rad(ang)), 1.05*np.sin(np.deg2rad(ang))],
                color=INK, lw=1.0, zorder=8)
    mid = np.deg2rad((ALPHA_A + ALPHA_B)/2)
    ax.text(1.12*np.cos(mid), 1.12*np.sin(mid) + .02, 'angle\nbetween', fontsize=fs, ha='left', va='center', linespacing=1.15)
    ax.scatter([0], [0], s=40, color=INK, zorder=9)


def key_panel(ax, fs):
    """Two tiny cases: patches that coincide (overlap 1) and patches that cross (about 0.05, unscaled)."""
    blank(ax, (-.2, 3.0), (-2.3, .5))
    a, b = .5, .26
    patch2(ax, (.55, -.3), a, b, 30, color=MUTE, dashed=True, lw=1.3, axis_lw=2.2)
    patch2(ax, (.55, -.3), a, b, 30, color=INK, lw=1.3, axis_lw=2.2, z=7)
    ax.text(1.35, -.3, 'same directions:\noverlap 1', fontsize=fs, va='center', linespacing=1.25)
    patch2(ax, (.55, -1.6), a, b, 20, color=INK, lw=1.3, axis_lw=2.2)
    patch2(ax, (.55, -1.6), a, b, 110, color=MUTE, dashed=True, lw=1.3, axis_lw=2.2, z=7)
    ax.text(1.35, -1.6, 'unrelated directions:\noverlap about 0.05,\nunscaled', fontsize=fs, va='center', linespacing=1.25)


def draw(mobile=False):
    fs = 11.5
    A, B, tA, tB = build()
    if mobile:
        W, H = 4.3, 12.2
        fig = plt.figure(figsize=(W, H))
        boxes = [(.2, 1.35, 3.9, 2.9), (.3, 5.0, 3.6, 2.7), (.2, 8.6, 3.9, 2.0)]
        titles = [(1.2, 'Two works'), (4.85, 'Carried to the other work'), (8.45, 'Two limiting cases')]
        title_x = [.2, .3, .2]
    else:
        W, H = 11.6, 4.95
        fig = plt.figure(figsize=(W, H))
        boxes = [(.2, 1.2, 3.5, 3.0), (4.15, 1.2, 3.3, 2.7), (8.0, 1.2, 3.4, 2.7)]
        titles = [(1.05, 'Two works'), (1.05, 'Carried to the other work'), (1.05, 'Two limiting cases')]
        title_x = [.2, 4.15, 8.0]
    axs = [fig.add_axes([x/W, 1 - (y + h)/H, w/W, h/H]) for x, y, w, h in boxes]
    sphere_panel(axs[0], A, B, tA, tB, fs)
    zoom_panel(axs[1], fs)
    key_panel(axs[2], fs)
    for (ty, t), x in zip(titles, title_x):
        fig.text(x/W, 1 - ty/H, t, fontsize=13, va='bottom')
    # keys under the zoom panel
    zx, zy, zw, zh = boxes[1]
    ky = zy + zh + .15
    for k, (dashed, color, txt) in enumerate([(False, INK, 'subspace fitted at this work'),
                                              (True, MUTE, 'the anchor\'s subspace, carried here')]):
        yy = ky + k*.21
        fig.add_artist(plt.Line2D([zx/W, (zx + .3)/W], [1 - yy/H]*2, color=color, lw=2.2,
                                  linestyle=(0, (3.5, 2.5)) if dashed else '-', transform=fig.transFigure))
        fig.text((zx + .42)/W, 1 - yy/H, txt, fontsize=fs, va='center')
    head(fig, 'Comparing local directions at two works', 'Drawn, not measured.')
    foot = ('The real subspace has 39 directions in 768 dimensions; one flat patch stands in for it. The overlap is how much of one\n'
            'lies inside the other. On the scaled overlap, 0 is what two unrelated subspaces give and 1 is how well the two halves agree at one work.' if not mobile else
            'The real subspace has 39 directions in 768\ndimensions; one flat patch stands in for it. The overlap\nis how much of one lies inside the other. On the scaled\n'
            'overlap, 0 is what two unrelated subspaces give and\n1 is how well the two halves agree at one work.')
    fig.text(.2/W, 1 - (4.45 if not mobile else 11.0)/H, foot, fontsize=11, color=MUTE, va='top', linespacing=1.4)
    save(fig, OUT + ('-mobile' if mobile else ''))


if __name__ == '__main__':
    draw()
    draw(mobile=True)
