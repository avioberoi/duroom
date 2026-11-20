"""Drawn, not measured: the cross-artist test at one anchor, 12 dots per half standing in for 100 and one line for the 39-direction local subspace.
Usage: python shape_walk.py"""
from _shape_draw import *

N_PER_HALF = 12          # stands in for 100
# each half takes its own nearest works to the anchor; the fitted line passes through the fit mean, not the anchor
OUT = 'shape_walk_s'
XL, YL = (-1.45, 1.45), (-1.66, .88)


def build(seed=131, alpha=0., beta=-.5, L=1.3, sb=.2, n=22, sep=.15, ang=24):
    """A wide field of works by other artists, assigned to halves A and B, then each half's own nearest."""
    rng = np.random.default_rng(seed)
    t = np.deg2rad(ang)
    u = np.array([np.cos(t), np.sin(t)])
    v = np.array([-u[1], u[0]])
    anchor_p = alpha*u + beta*v
    P, lab = [], []
    while len(P) < 2*n:
        q = rng.uniform(-L, L)*u + rng.normal(0, sb)*v
        if np.linalg.norm(q - anchor_p) < .3 or any(np.linalg.norm(q - p) <= sep for p in P):
            continue
        P.append(q); lab.append(len(P) % 2)
    P, lab = np.array(P), np.array(lab)
    A, B = P[lab == 0], P[lab == 1]
    nearest = lambda Q: np.argsort(np.linalg.norm(Q - anchor_p, axis=1))[:N_PER_HALF]
    ia, ib = nearest(A), nearest(B)       # taken within each half separately
    fit, test = A[ia], B[ib]
    r_A = np.linalg.norm(fit - anchor_p, axis=1).max() + .05
    r_B = np.linalg.norm(test - anchor_p, axis=1).max() + .05
    left = np.array([[-.32, .14], [.34, -.18], [.04, .32]]) + anchor_p   # same-artist works, close to the anchor
    return dict(anchor=anchor_p, fit=fit, test=test, rest_A=np.delete(A, ia, axis=0),
                rest_B=np.delete(B, ib, axis=0), r_A=r_A, r_B=r_B, left=left)


def checks(d):
    a, fit, test = d['anchor'], d['fit'], d['test']
    assert len(fit) == len(test) == N_PER_HALF
    mean = fit.mean(0)
    _, _, vt = np.linalg.svd(fit - mean, full_matrices=False)
    u = vt[0] if vt[0][0] > 0 else -vt[0]
    med = np.median(np.linalg.norm(np.vstack((fit, test)) - a, axis=1))
    ratio = np.linalg.norm(mean - a)/med
    off_line = abs(u[0]*(a - mean)[1] - u[1]*(a - mean)[0])/med
    spread = np.ptp((fit - mean) @ u)
    print(f'mean offset / median offset = {ratio:.2f}; anchor-to-line / median offset = {off_line:.2f}; '
          f'r_A={d["r_A"]:.2f} r_B={d["r_B"]:.2f}; fit spread along line = {spread:.2f}')
    assert .5 <= ratio <= .72, ratio
    assert off_line >= .3, off_line
    # the two halves reach almost the same distance in the data: rings differ by 8-14%, solid/dashed does the rest
    gap = abs(d['r_A'] - d['r_B'])/max(d['r_A'], d['r_B'])
    assert .08 <= gap <= .14, gap
    assert spread > 1.5
    # no dot sits on a same-artist cross
    allp = np.vstack((fit, test, d['rest_A'], d['rest_B']))
    assert np.min(np.linalg.norm(allp[:, None] - d['left'][None], axis=2)) > .12
    return mean, u


def layout(mobile):
    """Boxes in inches from the top-left, plus per-panel y ranges (mobile crops the empty bands)."""
    if mobile:
        W = 4.3
        ax_w = 3.5
        scale = ax_w/(XL[1] - XL[0])
        ylims = [YL, YL, (-1.25, .88), (-1.55, .45)]
        hs = [(y1 - y0)*scale for y0, y1 in ylims]
        boxes, top = [], 1.25
        for h in hs:
            boxes.append((.3, top, ax_w, h))
            top += h + 1.24
        H = top - 1.24 + .64 + .3 + .95
        return W, H, boxes, -.14, .14, H - .95 - .02, ylims
    W, H = 11.6, 4.7
    boxes = [(.2 + i*2.86, 1.28, 2.74, 2.4) for i in range(4)]
    return W, H, boxes, -.14, .2, 4.28, [YL]*4


class Page:
    def __init__(self, mobile):
        self.mobile = mobile
        self.W, self.H, self.boxes, self.title_dy, self.key_dy, self.foot_y, ylims = layout(mobile)
        self.fig = plt.figure(figsize=(self.W, self.H))
        self.axs = [self.fig.add_axes([x/self.W, 1 - (y + h)/self.H, w/self.W, h/self.H]) for x, y, w, h in self.boxes]
        for ax, yl in zip(self.axs, ylims):
            blank(ax, XL, yl)
        self.fs = 11.5

    def fx(self, x_in): return x_in/self.W
    def fy(self, y_in): return 1 - y_in/self.H

    def title(self, i, text):
        x, y, w, h = self.boxes[i]
        self.fig.text(self.fx(x), self.fy(y + self.title_dy), text, fontsize=13, va='bottom')

    def row(self, i, k, text, marker=None, line=None, color=INK, tcolor=INK, dashed=False, lw=1.4):
        """Key row k (0-based) under panel i: a marker or a short line sample, then the words."""
        x, y, w, h = self.boxes[i]
        yy = y + h + self.key_dy + k*.2
        if marker:
            key_marker(self.fig, self.fx(x + .12), self.fy(yy), marker, color=color)
        if line:
            key_line(self.fig, self.fx(x), self.fx(x + .26), self.fy(yy), color=color, lw=lw, dashed=dashed)
        tx = x + (.36 if (marker or line) else 0.0)
        self.fig.text(self.fx(tx), self.fy(yy), text, fontsize=self.fs, color=tcolor, va='center')

    def foot(self, text):
        self.fig.text(self.fx(.2), self.fy(self.foot_y), text, fontsize=11, color=MUTE, va='top', linespacing=1.4)


def label(ax, xy, text, fs, color=INK, ha='center', va='center'):
    return ax.text(xy[0], xy[1], text, fontsize=fs, color=color, ha=ha, va=va, zorder=10)


def data_box(ax, t):
    """A text's bounding box in the axes' data coordinates (x0, y0, x1, y1)."""
    ax.figure.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transData.inverted())
    return bb.x0, bb.y0, bb.x1, bb.y1


def draw(mobile=False):
    d = build()
    mean, u = checks(d)
    a, fit, test = d['anchor'], d['fit'], d['test']
    P = Page(mobile)
    ax0, ax1, ax2, ax3 = P.axs
    fs = P.fs
    rr = max(d['r_A'], d['r_B']) + .04
    ext = np.array([-1.55, 1.55])

    # i. One work and its neighbours: the other artists' works near the anchor, no halves yet
    sphere_arc(ax0, (0, -3.9), 4.6, 63, 117)
    allp = np.vstack((fit, test, d['rest_A'], d['rest_B']))
    inside = np.linalg.norm(allp - a, axis=1) <= rr
    pool_dots(ax0, allp[~inside])
    fit_dots(ax0, allp[inside])
    ring(ax0, a, rr)
    left_out(ax0, d['left'])
    anchor(ax0, a)
    P.title(0, 'One work and its neighbours')
    P.row(0, 0, 'anchor', marker='dot', color=ACCENT)
    P.row(0, 1, 'same artist, left out', marker='cross', color=MUTE)
    ring_label = label(ax0, a + np.array([0, -.6]), 'how far the\nneighbours reach', fs, color=MUTE)
    bx0, by0, bx1, by1 = data_box(ax0, ring_label)
    for cx, cy in d['left']:                                   # no cross touches the ring label
        assert not (bx0 - .08 < cx < bx1 + .08 and by0 - .08 < cy < by1 + .08), ('cross on the label', cx, cy)
    assert all(np.hypot(*(c - a)) < rr for c in d['left'])      # all three crosses sit inside the ring

    # ii. Two halves of the artists: each half keeps its own nearest works and so has its own ring
    sphere_arc(ax1, (0, -3.9), 4.6, 63, 117)
    pool_dots(ax1, d['rest_A']); pool_dots(ax1, d['rest_B'])
    ring(ax1, a, d['r_A'], color=INK, lw=1.3, dashed=False)
    ring(ax1, a, d['r_B'], color=INK, lw=1.3, dashed=True)
    fit_dots(ax1, fit); test_dots(ax1, test)
    anchor(ax1, a)
    P.title(1, 'Two halves of the artists')
    P.row(1, 0, 'fit: artists in half A, solid ring', marker='dot')
    P.row(1, 1, 'test: artists in half B, dashed ring', marker='hollow')

    # iii. Fit: only the fit half, spread along the line; their mean; the line through the mean
    sphere_arc(ax2, (0, -3.9), 4.6, 63, 117)
    ends = mean + ext[:, None]*u
    ax2.plot(ends[:, 0], ends[:, 1], color=INK, lw=1.5, zorder=3, solid_capstyle='round')
    fit_dots(ax2, fit)
    ax2.plot([a[0], mean[0]], [a[1], mean[1]], color=MUTE, lw=1.1, linestyle=(0, (1.5, 2.5)), zorder=2)
    mean_tick(ax2, mean)
    # place 'mean' where no dot or the line sits under the word: try the four corners around the +
    def clear(box):
        x0, x1, y0, y1 = box
        inbox = ((fit[:, 0] > x0 - .1) & (fit[:, 0] < x1 + .1) & (fit[:, 1] > y0 - .1) & (fit[:, 1] < y1 + .1)).any()
        t = np.linspace(x0, x1, 12)
        online = np.any(np.abs(mean[1] + (t - mean[0])*u[1]/u[0] - (y0 + y1)/2) < .14)
        return not inbox and not online
    for dx, dy, ha in [(.14, -.30, 'left'), (-.14, -.30, 'right'), (.14, .38, 'left'), (-.14, .38, 'right')]:
        x0, x1 = (mean[0] + dx, mean[0] + dx + .5) if ha == 'left' else (mean[0] + dx - .5, mean[0] + dx)
        y0, y1 = mean[1] + dy - .1, mean[1] + dy + .1
        if clear((x0, x1, y0, y1)):
            ax2.annotate('mean', xy=mean, xytext=mean + np.array([dx, dy]), ha=ha, va='center', fontsize=fs, color=INK,
                         arrowprops={'arrowstyle': '-', 'color': MUTE, 'lw': .8, 'shrinkA': 2, 'shrinkB': 7}, zorder=10)
            break
    else:
        raise RuntimeError('no clear spot for the mean label')
    anchor(ax2, a)
    P.title(2, 'Fit')
    P.row(2, 0, 'mean of the fit dots', marker='plus')
    P.row(2, 1, 'local subspace, through the mean', line=True, color=INK, lw=1.5)
    label(ax2, a + np.array([0, -.62]), 'The anchor is\nnot the mean.', fs + .5, color=INK)

    # iv. Test: the other half, centred on its own mean; the fitted directions carried over.
    # Offsets are magnified (M) so that the split into held / not held is easy to read; the layout is schematic.
    M = 1.4
    O = np.array([0., -.45])
    c = (test - test.mean(0))*M
    ax3.plot(*(ext[:, None]*u + O).T, color=INK, lw=1.5, zorder=3, solid_capstyle='round')
    perp = np.array([u[1], -u[0]])
    along, off = c @ u, c @ perp
    i_pos = max((i for i in range(len(c)) if along[i] > .35), key=lambda i: abs(off[i]))
    i_neg = max((i for i in range(len(c)) if along[i] < -.35), key=lambda i: abs(off[i]))
    test_dots(ax3, c + O)
    for i in (i_pos, i_neg):
        q = c[i]; foot = along[i]*u
        ax3.plot(*np.array([O, q + O]).T, color=MUTE, lw=1.0, linestyle=(0, (1.5, 2.5)), zorder=4)
        ax3.plot(*np.array([O, foot + O]).T, color=INK, lw=3.6, zorder=7, solid_capstyle='butt')
        ax3.plot(*np.array([foot + O, q + O]).T, color=MUTE, lw=1.9, zorder=6, solid_capstyle='butt')
        e1 = -np.sign(along[i])*u*.08; e2 = np.sign(off[i])*perp*.08
        ax3.plot(*np.array([foot + O + e1, foot + O + e1 + e2, foot + O + e2]).T, color=MUTE, lw=.9, zorder=6)
    # direct labels, once each, on the right-hand dot: no leader lines
    i = i_pos; foot = along[i]*u; s_off = np.sign(off[i])
    label(ax3, foot*.72 + O + s_off*perp*.2, 'held', fs, color=INK)
    label(ax3, c[i] + O + np.array([0, -.22]), 'not held', fs, color=MUTE)
    mean_tick(ax3, O)
    P.title(3, 'Test')
    P.row(3, 0, 'share held = spread along the', tcolor=INK)
    P.row(3, 1, 'fitted directions / all spread', tcolor=INK)
    ax3.plot(*np.array([O + [0, -.14], O + [0, -.34]]).T, color=MUTE, lw=.8, zorder=4)
    label(ax3, O + np.array([0, -.5]), 'own mean', fs, color=INK)

    head(P.fig, 'The test at one work', 'Drawn, not measured.')
    P.foot('One line stands in for the 39 directions of the real subspace, which lies in 768 dimensions; 12 dots stand in for 100 per half.\n'
           'Each half is centred on its own mean, so the fitted line need not pass through the anchor.' if not mobile else
           'One line stands in for the 39 directions of the real\nsubspace, which lies in 768 dimensions; 12 dots stand\n'
           'in for 100 per half. Each half is centred on its own\nmean, so the fitted line need not pass through the anchor.')
    save(P.fig, OUT + ('-mobile' if mobile else ''))


if __name__ == '__main__':
    draw()
    draw(mobile=True)
