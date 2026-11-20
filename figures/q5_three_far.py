"""Three ways to be far, one CLIP work per distance, all panels to one scale; also writes q5_three_far_local-small (the local panel alone).
Usage: python q5_three_far.py"""
import textwrap

from _panels import *
from matplotlib.patches import Arc, Circle

d = load_data('q5_three_far')
ABS, LOC, CEN = d['panels']
CORPUS = d['corpus']
N = d['n']
DASH = (0, (4, 3))
THUMB = {'absolute': .64, 'local': 1.12, 'centre': .48}   # thumbnail size in chord units (height, width, height)


def thumb(ax, work, xy, width=None, height=None):
    """The artwork is the plot mark, centred on its position; equal axes keep its aspect."""
    img = plt.imread(HERE / work['asset'])
    h, w = img.shape[:2]
    width = width if width is not None else height * w / h
    height = height if height is not None else width * h / w
    x, y = xy
    ax.imshow(img, extent=[x - width / 2, x + width / 2, y - height / 2, y + height / 2],
              zorder=6, interpolation='antialiased')


def ring(ax, r, xy=(0, 0), color=INK, lw=.8, ls='-', z=2):
    ax.add_patch(Circle(xy, r, fill=False, edgecolor=color, lw=lw, ls=ls, zorder=z))


def polar(r, degrees):
    t = np.radians(degrees)
    return r * np.cos(t), r * np.sin(t)


def draw_absolute(ax, fs):
    a = CORPUS['absolute']
    ring(ax, a['median'], color=MUTE)
    ring(ax, a['p99'], color=MUTE, ls=DASH)
    ring(ax, ABS['ring'], color=ACCENT, lw=1.3, z=3)
    nb = np.array(ABS['layout']['neighbours'])
    ax.scatter(nb[:, 0], nb[:, 1], s=10, color=INK, linewidth=0, zorder=4)
    thumb(ax, ABS['work'], (0, 0), height=THUMB['absolute'])
    x, y = polar(ABS['ring'], -50)                  # a radius with no neighbour drawn on it
    ax.text(x + .05, y - .03, f"{ABS['ring']:.3f}", fontsize=fs, color=ACCENT, ha='left', va='top')


def draw_local(ax, fs):
    nb = np.array(LOC['layout']['neighbours'])
    for (x, y), r in zip(nb, LOC['neighbour_rings']):
        ring(ax, r, (x, y), color=MUTE, lw=.6)
    ax.scatter(nb[:, 0], nb[:, 1], s=10, color=INK, linewidth=0, zorder=4)
    ring(ax, LOC['ring'], color=ACCENT, lw=1.3, z=3)
    thumb(ax, LOC['work'], (0, 0), width=THUMB['local'])
    x, y = polar(LOC['ring'], 64)
    ax.text(x + .05, y + .03, f"{LOC['ring']:.3f}", fontsize=fs, color=ACCENT, ha='left', va='bottom', zorder=7)


def draw_centre(ax, fs):
    c = CORPUS['centre_angle']
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor=HAIR, lw=.9, zorder=1))
    # Latitude rings seen edge-on: every work within that angle of the pole lies above the line.
    for deg, ls in [(c['median'], '-'), (c['p99'], DASH)]:
        t = np.radians(deg)
        ax.plot([-np.sin(t), np.sin(t)], [np.cos(t)] * 2, color=MUTE, lw=.8, ls=ls, zorder=2)
        ax.text(-np.sin(t) - .05, np.cos(t), f'{deg:.1f}°', fontsize=fs, color=MUTE, ha='right', va='center')
    t = np.radians(CEN['angle'])
    work = (np.sin(t), np.cos(t))                   # the right-hand side is drawn; only the angle is measured
    ax.plot([0, 0], [0, 1], color=HAIR, lw=.7, zorder=1)
    ax.plot([0, work[0]], [0, work[1]], color=HAIR, lw=.7, zorder=1)
    ax.add_patch(Arc((0, 0), .56, .56, theta1=90 - CEN['angle'], theta2=90, color=INK, lw=.8, zorder=3))
    lx, ly = polar(.36, 90 - CEN['angle'] / 2)
    ax.text(lx, ly, f"{CEN['angle']:.1f}°", fontsize=fs, color=INK, ha='left', va='bottom')
    # The chord from the pole is the stored centre distance; its label sits just outside the circle, beside it.
    ax.plot([0, work[0]], [1, work[1]], color=ACCENT, lw=1.3, zorder=4)
    ax.text(.68, .80, f"{CEN['distance']:.3f}", fontsize=fs, color=ACCENT, ha='left', va='bottom')
    ax.scatter([0], [1], s=16, color=INK, linewidth=0, zorder=5)
    ax.scatter([0], [0], s=8, color=MUTE, linewidth=0, zorder=5)
    ax.text(0, 1.07, 'Average direction', fontsize=fs, color=MUTE, ha='center', va='bottom')
    thumb(ax, CEN['work'], work, height=THUMB['centre'])


def local_extent():
    nb = np.array(LOC['layout']['neighbours']); r = np.array(LOC['neighbour_rings'])
    return (min((nb[:, 0] - r).min(), -LOC['ring']) - .03, max((nb[:, 0] + r).max(), LOC['ring']) + .03,
            min((nb[:, 1] - r).min(), -LOC['ring']) - .03, max((nb[:, 1] + r).max(), LOC['ring']) + .03)


# Content boxes in chord units (x0, x1, y0, y1): every panel is cropped to its drawing, never rescaled.
EXTENT = {'absolute': (-1.13, 1.22, -1.13, 1.13), 'local': local_extent(), 'centre': (-1.34, 1.25, -1.03, 1.25)}

PANELS = [
    ('absolute', 'Absolute', 'distance to the 10th nearest work', ABS, draw_absolute),
    ('local', 'Local', 'that distance, over its neighbours’ own', LOC, draw_local),
    ('centre', 'Centre', 'angle from the average direction', CEN, draw_centre),
]


def metric(panel):
    if panel is ABS:
        return f"{ABS['ring']:.3f}, rank {ABS['rank']:,} of {N:,}"
    if panel is LOC:
        return (f"{LOC['ring']:.3f} / {LOC['neighbour_median']:.3f} = {LOC['ratio']:.2f}, "
                f"rank {LOC['rank']:,} of {N:,}")
    return f"{CEN['distance']:.3f}, or {CEN['angle']:.1f}°, rank {CEN['rank']:,} of {N:,}"


def note(panel):
    if panel is ABS:
        a = CORPUS['absolute']
        return f"Half of all works’ rings are within {a['median']:.3f} (solid), 99% within {a['p99']:.3f} (dashed)"
    if panel is LOC:
        rings = LOC['neighbour_rings']
        return (f"Its ten neighbours’ own rings are {min(rings):.2f} to {max(rings):.2f}, "
                f"median {LOC['neighbour_median']:.3f}")
    c = CORPUS['centre_angle']
    return f"Half of all works are within {c['median']:.1f}° (solid), 99% within {c['p99']:.1f}° (dashed)"


def credit(work, width):
    year = work['year'] if work['year'] is not None else 'date unknown'
    return '\n'.join([work['artist']] + textwrap.wrap(f"{work['title']}, {year}", width))


def text_block(fig, x, top, panel, fs, wrap, figw, figh):
    """Credit (ink), the measured value (accent), the corpus reference (mute); returns the lowest y used."""
    line = fs * 1.2 / 72                            # inches per text line
    cr = credit(panel['work'], wrap)
    fig.text(x / figw, top / figh, cr, fontsize=fs, color=INK, ha='left', va='top', linespacing=1.2)
    y = top - (cr.count('\n') + 1) * line - .08
    fig.text(x / figw, y / figh, metric(panel), fontsize=fs, color=ACCENT, ha='left', va='top')
    y -= line + .08
    nt = '\n'.join(textwrap.wrap(note(panel), wrap + 4))
    fig.text(x / figw, y / figh, nt, fontsize=fs, color=MUTE, ha='left', va='top', linespacing=1.2)
    return y - (nt.count('\n') + 1) * line


def draw_box(fig, key, draw, x0, y0, scale, figw, figh, fs, height=None):
    """One geometry box at the shared scale; `height` (chord units) centres the drawing in a taller row."""
    ex0, ex1, ey0, ey1 = EXTENT[key]
    if height is not None:
        mid = (ey0 + ey1) / 2; ey0, ey1 = mid - height / 2, mid + height / 2
    w, h = (ex1 - ex0) * scale, (ey1 - ey0) * scale
    ax = fig.add_axes([x0 / figw, y0 / figh, w / figw, h / figh])
    draw(ax, fs)
    ax.set_xlim(ex0, ex1); ax.set_ylim(ey0, ey1); ax.set_aspect('equal'); ax.axis('off')
    return w, h


def desktop(panels, scale=1.0, figw=8.8):
    fs, wrap = 10, 36
    title_h, text_h = .58, 1.30
    widths = [(EXTENT[k][1] - EXTENT[k][0]) * scale for k, *_ in panels]
    rows = max(EXTENT[k][3] - EXTENT[k][2] for k, *_ in panels)
    box_h = rows * scale
    gap = (figw - sum(widths)) / (len(panels) - 1)
    figh = title_h + box_h + text_h
    fig = plt.figure(figsize=(figw, figh), dpi=200)
    x = 0
    for (key, title, sub, panel, draw), w in zip(panels, widths):
        draw_box(fig, key, draw, x, text_h, scale, figw, figh, fs, height=rows)
        fig.text(x / figw, (figh - .02) / figh, title, fontsize=12, color=INK, ha='left', va='top')
        fig.text(x / figw, (figh - .26) / figh, sub, fontsize=fs, color=MUTE, ha='left', va='top')
        text_block(fig, x, text_h - .06, panel, fs, wrap, figw, figh)
        x += w + gap
    return fig


def text_height(panel, fs, wrap):
    lines = credit(panel['work'], wrap).count('\n') + 1 + 1 + len(textwrap.wrap(note(panel), wrap + 4))
    return lines * fs * 1.2 / 72 + .16 + .08


def stacked(panels, scale, figw=3.9):
    fs, wrap = 11, 40
    title_h, pad = .60, .42
    heights = [(EXTENT[k][3] - EXTENT[k][2]) * scale for k, *_ in panels]
    texts = [text_height(p, fs, wrap) for _, _, _, p, _ in panels]
    figh = sum(title_h + h + t + pad for h, t in zip(heights, texts)) - pad
    fig = plt.figure(figsize=(figw, figh), dpi=200)
    top = figh
    for (key, title, sub, panel, draw), h, t in zip(panels, heights, texts):
        w = (EXTENT[key][1] - EXTENT[key][0]) * scale
        fig.text(0, (top - .02) / figh, title, fontsize=12, color=INK, ha='left', va='top')
        fig.text(0, (top - .27) / figh, sub, fontsize=fs, color=MUTE, ha='left', va='top')
        y0 = top - title_h - h
        draw_box(fig, key, draw, .5 * (figw - w), y0, scale, figw, figh, fs)
        text_block(fig, 0, y0 - .08, panel, fs, wrap, figw, figh)
        top = y0 - t - pad
    return fig


finish(desktop(PANELS), 'q5_three_far', False)
finish(stacked(PANELS, scale=1.06), 'q5_three_far', True)

# The local panel alone.
SMALL = [('local', 'The Forest Fire among its ten neighbours', 'its 10th-neighbour ring against theirs, CLIP',
          LOC, draw_local)]
save(stacked(SMALL, scale=.98, figw=3.4), 'q5_three_far_local-small')
save(stacked(SMALL, scale=1.06), 'q5_three_far_local-small-mobile')
