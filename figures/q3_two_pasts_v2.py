"""Median leftover share by decade under the whole-corpus basis and under the basis from the works before 1900 (CLIP, 32 directions), and four works whose share moves between the two.
Usage: python q3_two_pasts_v2.py"""
import textwrap
from _panels import *

d = load_data('q3_two_pasts')
dec = d['decades']
works = {w['short']: w for w in d['works']}
by = {r['decade']: r for r in dec}
XCUT = d['cutoff_year']
NUM = {'Calder': '1', 'Hamilton': '2', 'Kelly': '3', 'Erkmen': '4'}     # mobile order = year order


def segments(key, x0):
    """Runs of consecutive decades from x0 on (the 1680s and 1690s hold too few works to plot)."""
    runs, run = [], []
    for r in dec:
        if r['decade'] < x0:
            continue
        if run and r['decade'] - run[-1]['decade'] > 10:
            runs.append(run); run = []
        run.append(r)
    runs.append(run)
    return [([r['decade'] + 5 for r in s], [r[key] for r in s]) for s in runs]


def pct(x):
    return f'{x:.3f}'.lstrip('0')


def shares(w):
    return f"{pct(w['share'])} to {pct(w['share_past'])}"


def plot_core(ax, x0, mobile):
    ax.axvspan(x0, XCUT, color=PALE, alpha=.35, lw=0, zorder=0)
    ax.axvline(XCUT, color=HAIR, lw=.8, ls=(0, (3, 3)), zorder=1)
    for xs, ys in segments('corpus', x0):
        ax.plot(xs, ys, color=INK, lw=1.1, zorder=3, solid_capstyle='round')
    for xs, ys in segments('past', x0):
        ax.plot(xs, ys, color=ACCENT, lw=1.3, zorder=4, solid_capstyle='round')
    for w in works.values():
        x = w['year']
        ax.plot([x, x], [w['share'], w['share_past']], color=INK, lw=.9, zorder=5)
        ax.scatter([x], [w['share']], s=26 if mobile else 24, facecolor=PAPER, edgecolor=INK, linewidth=.9, zorder=6)
        ax.scatter([x], [w['share_past']], s=26 if mobile else 24, color=ACCENT, zorder=6, linewidth=0)
    ax.set_xlim(x0, 2030); ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(['0', '.25', '.5', '.75', '1'])
    ax.set_ylabel('Leftover share', fontsize=11 if mobile else 10.5, labelpad=4)
    clean_axes(ax, mobile)


def fig_inches(fig, ax):
    inv = fig.dpi_scale_trans.inverted()
    return lambda x, y: tuple(inv.transform(ax.transData.transform((x, y))))


def overlay(fig):
    W, H = fig.get_size_inches()
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis('off'); ov.set_zorder(10)
    return ov


def thumb(fig, key, cx, ytop, wmax, hmax):
    """Artwork with its top centre at (cx, ytop) inches, shape kept, inside wmax x hmax. Returns width, height."""
    img = plt.imread(HERE / works[key]['asset'])
    h, w = img.shape[:2]
    s = min(wmax / w, hmax / h)
    iw, ih = w * s, h * s
    W, H = fig.get_size_inches()
    a = fig.add_axes([(cx - iw / 2) / W, (ytop - ih) / H, iw / W, ih / H])
    a.imshow(img, aspect='auto'); a.axis('off')
    return iw, ih


def wrapped(text, n):
    return '\n'.join(textwrap.wrap(text, n))


def curve_labels(ax, mobile, xr, y_past, y_corpus, fs):
    c, p = by[1960]['corpus'], by[1960]['past']
    ax.text(xr, y_past, f'{"Basis from before 1900" if mobile else "The basis from before 1900"}\n1960s median {pct(p)}',
            color=ACCENT, fontsize=fs, ha='right', va='center', linespacing=1.25, zorder=7)
    ax.plot([xr + 3, 1960], [y_past - .03, p + .012], color=HAIR, lw=.8, zorder=2)
    ax.text(xr, y_corpus, f'{"Whole corpus" if mobile else "Whole-corpus basis"}\n1960s median {pct(c)}',
            color=INK, fontsize=fs, ha='right', va='center', linespacing=1.25, zorder=7)
    ax.plot([xr + 3, 1960], [y_corpus + .035, c - .012], color=HAIR, lw=.8, zorder=2)


def desktop():
    W, H = 8.8, 5.1
    fig = plt.figure(figsize=(W, H))
    L, R, B, T = .075, .985, .08, .625
    ax = fig.add_axes([L, B, R - L, T - B])
    x0 = 1700
    plot_core(ax, x0, False)
    ax.set_xticks([1700, 1750, 1800, 1850, 1900, 1950, 2000])
    fig.canvas.draw()
    P = fig_inches(fig, ax)
    curve_labels(ax, False, 1938, .765, .345, 10)
    ax.text(1704, .975, 'The basis from before 1900\nis fit on these works', color=MUTE, fontsize=10, ha='left', va='top',
            linespacing=1.25)
    ov = overlay(fig)
    top, tw, th = H - .06, 1.05, .9
    for k, cx in {'Calder': 3.2, 'Kelly': 4.85, 'Erkmen': 6.5}.items():
        w = works[k]
        thumb(fig, k, cx, top, tw, th)
        body = f"{w['artist']}\n{wrapped(w['title'] + ', ' + str(w['year']), 30)}\n{shares(w)}"
        ov.text(cx, top - th - .07, body, color=INK, fontsize=10, ha='center', va='top', linespacing=1.22)
        sx, sy = P(w['year'], w['share_past'] + .02)
        ov.plot([cx, sx], [top - th - .07 - .17 * (body.count('\n') + 1) - .06, sy], color=HAIR, lw=.8)
    # legend for the two ends of a stem, top left
    lx, ly = .85, H - .2
    ov.scatter([lx], [ly], s=24, facecolor=PAPER, edgecolor=INK, linewidth=.9)
    ov.text(lx + .14, ly, 'share under the', fontsize=10, va='center', ha='left', color=INK)
    ov.text(lx + .14, ly - .19, 'whole-corpus basis', fontsize=10, va='center', ha='left', color=INK)
    ov.scatter([lx], [ly - .5], s=24, color=ACCENT, linewidth=0)
    ov.text(lx + .14, ly - .5, 'share under the basis', fontsize=10, va='center', ha='left', color=INK)
    ov.text(lx + .14, ly - .69, 'from before 1900', fontsize=10, va='center', ha='left', color=INK)
    # Hamilton: inside the plot, below the curves
    w = works['Hamilton']
    cx, ytop = P(1735, .31)
    iw, ih = thumb(fig, 'Hamilton', cx, ytop, .7, .6)
    tx = cx + iw / 2 + .1
    ov.text(tx, ytop - .01, f"{w['artist']}\n{wrapped(w['title'] + ', ' + str(w['year']), 24)}\n{shares(w)}", color=INK,
            fontsize=10, ha='left', va='top', linespacing=1.22)
    sx, sy = P(w['year'], w['share'] - .02)
    ov.plot([tx + 1.58, sx], [ytop - .2, sy], color=HAIR, lw=.8)
    finish(fig, 'q3_two_pasts_v2', False)


def mobile():
    W, H = 3.9, 8.2
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.15, .535, .83, .435])
    x0 = 1800
    plot_core(ax, x0, True)
    ax.set_xticks([1800, 1850, 1900, 1950, 2000])
    fig.canvas.draw()
    curve_labels(ax, True, 1948, .73, .27, 11)
    ax.text(1804, .985, 'The basis from before\n1900 is fit on these works', color=MUTE, fontsize=11, ha='left',
            va='top', linespacing=1.25)
    for k, w in works.items():
        y = w['share_past'] + .035 if k in ('Calder', 'Kelly', 'Erkmen') else w['share'] - .035
        va = 'bottom' if k in ('Calder', 'Kelly', 'Erkmen') else 'top'
        ha = {'Kelly': 'right', 'Erkmen': 'left'}.get(k, 'center')
        dx = {'Kelly': 1.5, 'Erkmen': -1.5}.get(k, 0)
        ax.text(w['year'] + dx, y, NUM[k], fontsize=11, color=INK, ha=ha, va=va, zorder=8)
    ov = overlay(fig)
    ly = H * .535 - .55
    ov.scatter([.45], [ly], s=26, facecolor=PAPER, edgecolor=INK, linewidth=.9)
    ov.text(.6, ly, 'share under the whole-corpus basis', fontsize=11, va='center', ha='left', color=INK)
    ov.scatter([.45], [ly - .26], s=26, color=ACCENT, linewidth=0)
    ov.text(.6, ly - .26, 'share under the basis from before 1900', fontsize=11, va='center', ha='left', color=INK)
    top0 = ly - .55
    cell_w, cell_h = 1.85, 2.08
    for i, k in enumerate(['Calder', 'Hamilton', 'Kelly', 'Erkmen']):
        w = works[k]
        cx = .15 + cell_w / 2 + (i % 2) * cell_w + (i % 2) * .05
        ytop = top0 - (i // 2) * cell_h
        thumb(fig, k, cx, ytop, 1.45, 1.0)
        ov.text(cx, ytop - 1.06, f"{NUM[k]}  {w['artist']}\n{wrapped(w['title'] + ', ' + str(w['year']), 22)}\n{shares(w)}",
                color=INK, fontsize=11, ha='center', va='top', linespacing=1.2)
    finish(fig, 'q3_two_pasts_v2', True)


desktop()
mobile()
