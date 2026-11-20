"""Each work at its measured angle on a quarter circle (leftover share = sin^2 of the angle), for the whole-corpus basis and the basis from the works before 1900.
Usage: python q3_fan.py"""
import textwrap
from matplotlib.patches import Wedge, Arc, Rectangle
from PIL import Image
from _panels import *

d = load_data('q3_fan')
W = {w['short']: w for w in d['works']}
ref = d['reference']
CALDER = W['Calder']
rad = np.radians
pct = lambda x: f'{100 * x:.1f}%'
deg = lambda x: f'{x:.1f}°'
IMG = {}


def thumb(w, px=120):
    if w['asset'] not in IMG:
        im = Image.open(HERE / w['asset']).convert('RGB')
        im.thumbnail((px, px), Image.Resampling.LANCZOS)
        IMG[w['asset']] = np.asarray(im)
    return IMG[w['asset']]


def pt(ox, oy, r, a):
    return ox + r * np.cos(rad(a)), oy + r * np.sin(rad(a))


def panel(ax, ox, oy, past, m):
    R, Rt, s, fs = m['R'], m['R'] + m['gap'], m['s'], m['fs']
    key, rkey = ('angle_past', 'ring_past') if past else ('angle', 'ring')
    skey = 'share_past' if past else 'share'
    ax.add_patch(Wedge((ox, oy), R, ref['cut']['angle'], 90, fc=DEEP, ec='none', zorder=0))
    ax.plot([ox, ox + R + .12], [oy, oy], color=INK, lw=.6, zorder=1)
    ax.plot([ox, ox], [oy, oy + R + .12], color=INK, lw=.6, zorder=1)
    ax.add_patch(Arc((ox, oy), 2 * R, 2 * R, theta1=0, theta2=90, ec=INK, lw=1.1, zorder=2))
    # whole-corpus references: median (dashed, as in the distribution plate) and the top-1% cut
    x, y = pt(ox, oy, R, ref['median']['angle'])
    ax.plot([ox, x], [oy, y], color=INK, lw=.8, ls=(0, (3, 3)), zorder=1)
    x, y = pt(ox, oy, R, ref['cut']['angle'])
    ax.plot([ox, x], [oy, y], color=INK, lw=1.1, zorder=1)
    # legend for the two reference rays, under the axis to the right of the labels
    lx, ly = ox + m['leg_x'], oy - m['leg_y']
    ax.plot([lx, lx + .28], [ly, ly], color=INK, lw=.8, ls=(0, (3, 3)), zorder=5, clip_on=False)
    ax.text(lx + .36, ly, f'median, {deg(ref["median"]["angle"])}', fontsize=fs, color=INK, ha='left', va='center')
    ax.plot([lx, lx + .28], [ly - m['row'], ly - m['row']], color=INK, lw=1.1, zorder=5)
    ax.text(lx + .36, ly - m['row'], f'top-1% cut, {deg(ref["cut"]["angle"])}', fontsize=fs, color=INK, ha='left',
            va='center')
    ax.add_patch(Rectangle((lx, ly - 2 * m['row'] - .06), .28, .12, fc=DEEP, ec='none', zorder=5, clip_on=False))
    ax.text(lx + .36, ly - 2 * m['row'], 'top 1%', fontsize=fs, color=INK, ha='left', va='center')
    if past:
        ax.text(lx, ly - 3 * m['row'], 'whole-corpus values, as before', fontsize=fs, color=MUTE, ha='left',
                va='center', style='italic')
    # one worked triangle (Calder): the part along the directions and the part off them
    th = CALDER[key]
    cx, cy = pt(ox, oy, R, th)
    ax.plot([ox, cx], [oy, cy], color=INK, lw=.6, zorder=3)
    ax.plot([ox, cx], [oy, oy], color=INK, lw=2.4, solid_capstyle='butt', zorder=3)
    ax.plot([cx, cx], [oy, cy], color=ACCENT, lw=1.4, ls=(0, (3.5, 2.5)), zorder=3)
    q = .09
    ax.plot([cx - q, cx - q, cx], [oy, oy + q, oy + q], color=INK, lw=.5, zorder=3)
    ax.add_patch(Arc((ox, oy), .5 * R, .5 * R, theta1=0, theta2=th, ec=INK, lw=.6, zorder=3))
    x, y = pt(ox, oy, .27 * R, th / 2)
    ax.text(x + .04, y, 'θ', fontsize=fs, color=INK, ha='left', va='center', style='italic', zorder=5)
    ax.text(cx + .07, oy + m['lo_y'], f'left over\n{pct(CALDER[skey])}', fontsize=fs, color=ACCENT,
            ha='left', va='center', zorder=5, linespacing=1.12)
    ax.text(ox + (cx - ox) / 2, oy - .08, f'described {pct(1 - CALDER[skey])}', fontsize=fs, color=INK,
            ha='center', va='top', zorder=5)
    ax.text(ox + R / 2, oy - .08 - m['note_dy'], 'along the 32 directions', fontsize=fs, color=MUTE, ha='center',
            va='top', style='italic')
    ax.text(ox - .1, oy + R * .5, 'off the directions', fontsize=fs, color=MUTE, ha='right', va='center',
            style='italic', rotation=90)
    # marks: dot on the arc, leader to the thumbnail on the ring
    for w in d['works']:
        a, ra = w[key], w[rkey]
        dx, dy = pt(ox, oy, R, a)
        tx, ty = pt(ox, oy, Rt, ra)
        ex, ey = tx, ty
        ax.plot([dx, ex], [dy, ey], color=HAIR, lw=.8, zorder=2)
        ax.plot([dx], [dy], 'o', color=INK, ms=m['dot'], mec=PAPER, mew=.6, zorder=6)
        im = thumb(w)
        h, wd = im.shape[:2]
        k = s / max(h, wd)
        ax.imshow(im, extent=(tx - wd * k / 2, tx + wd * k / 2, ty - h * k / 2, ty + h * k / 2), zorder=4,
                  interpolation='lanczos')
        ax.add_patch(Rectangle((tx - wd * k / 2, ty - h * k / 2), wd * k, h * k, fc='none', ec=INK, lw=.5, zorder=5))
        txt = f'{w["short"]}\n{deg(a)}'
        if past and not m['mobile'] and abs(w['angle_past'] - w['angle']) >= 1:
            txt += f'\nwas {deg(w["angle"])}'
        right = ra < m['right_below']
        if right:
            ax.text(tx + s / 2 + .06, ty + (.05 if m['mobile'] else 0), txt, fontsize=fs, color=INK, ha='left', va='center', linespacing=1.12,
                    zorder=6)
        else:
            lift = m['lift'].get(w['short'], 0)
            edge = m['mobile'] and ra > 80          # leftmost slot: label grows leftwards, clear of its neighbour
            ax.text(tx + (s / 2 if edge else 0), ty + s / 2 + .05 + lift, txt, fontsize=fs, color=INK,
                    ha='right' if edge else 'center', va='bottom', linespacing=1.12, zorder=6)


def credit_text(width):
    return '\n'.join(textwrap.wrap('Artworks (WikiArt reproductions): ' + '; '.join(
        f'{w["artist"]}, {w["title"]}, {w["year"]}' for w in d['works']) + '. The basis from the works before 1900 '
        'includes those works and takes its own average.', width))


for mobile in (False, True):
    if mobile:
        m = dict(R=1.74, gap=.38, s=.38, fs=11, dot=4.5, mobile=True, right_below=60, leg_x=1.78, leg_y=.2, row=.27, lo_y=.45,
                 note_dy=.30, lift={})
        Wd, left = 3.9, .42
        up = m['R'] + m['gap'] + m['s'] / 2 + .62          # axis note below, thumbnails and labels above
        base = 1.2
        ph = base + up + .34                                # one panel with its title
        credits = credit_text(46)
        ncred = credits.count('\n') + 1
        cred_h = .08 + ncred * .205
        Hd = 2 * ph + cred_h + .1
        origins = [(left, Hd - ph + base), (left, Hd - 2 * ph + base)]
        titles = [Hd - .02, Hd - ph - .02]
        credit_top = cred_h
        credit_x = left - .2
    else:
        m = dict(R=2.3, gap=.52, s=.50, fs=10, dot=5, mobile=False, right_below=52, leg_x=1.95, leg_y=.16, row=.2, lo_y=.62,
                 note_dy=.19, lift={})
        Wd, left = 8.8, .42
        up = m['R'] + m['gap'] + m['s'] / 2 + .62
        base = .98
        credits = credit_text(125)
        ncred = credits.count('\n') + 1
        cred_h = .08 + ncred * .17
        Hd = base + up + .34 + cred_h + .12
        origins = [(left, cred_h + .12 + base), (left + 4.5, cred_h + .12 + base)]
        titles = [Hd - .02, Hd - .02]
        credit_top = cred_h
        credit_x = left - .2
    fig = plt.figure(figsize=(Wd, Hd))
    ax = fig.add_axes([0, 0, 1, 1])
    for i, past in enumerate((False, True)):
        ox, oy = origins[i]
        panel(ax, ox, oy, past, m)
        ax.text(ox - .2, titles[i], 'Basis: the works before 1900 only' if past else 'Basis: the whole corpus',
                fontsize=12, color=INK, ha='left', va='top')
    ax.text(credit_x, credit_top, credits, fontsize=m['fs'], color=MUTE, ha='left', va='top', linespacing=1.3)
    ax.set_xlim(0, Wd)
    ax.set_ylim(0, Hd)
    ax.set_aspect('equal')
    ax.axis('off')
    finish(fig, 'q3_fan', mobile)
