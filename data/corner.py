"""Is the corner of the bend a void? Straight line vs the record's decade-centre path from before the cutoff to a later window: works within R, d10, d100 and artists at each point, and a three-part verdict per setting and seed.
Usage: corner.py OUT_DIR [CUT=1900] [WINDOW=1960,1979] [STEPS=12] [SEEDS=5] [CAPS=1,20,0] [STARTS=full,equal]"""
import json
import os
import sys
import zlib

import numpy as np

from basis import capped
from data import CLIP, card, labels, lens, year

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
CUT = int(sys.argv[2]) if len(sys.argv) > 2 else 1900
W0, W1 = (int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else '1960,1979').split(','))
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 12
SEEDS = int(sys.argv[5]) if len(sys.argv) > 5 else 5
CAPS = [int(c) for c in (sys.argv[6] if len(sys.argv) > 6 else '1,20,0').split(',')]
STARTS = (sys.argv[7] if len(sys.argv) > 7 else 'full,equal').split(',')
assert SEEDS >= 2
X = lens(CLIP).astype('float64')
X /= np.linalg.norm(X, axis=1, keepdims=True)
artist = labels['artist']
usable = (artist != '') & (year >= 1000) & (year <= 2026)
unit = lambda p: p / np.linalg.norm(p)    # noqa: E731


def centre(rows, cap, rng):
    """Mean of a capped sample (cap 0 = no cap, bootstrapped), back on the sphere, and how many works stand behind it."""
    c = capped(rows, cap, rng) if cap else rng.choice(rows, len(rows), replace=True)
    return unit(X[c].mean(0)), int(len(np.unique(c)))


def look(p, R, cards):
    """What is near a point on the sphere."""
    d = np.sqrt(np.maximum(2 - 2 * (X @ p), 0))    # Euclidean distance between unit vectors
    o = np.argsort(d)
    near = d < R
    out = {'d10': float(d[o[9]]), 'd100': float(d[o[99]]), 'within_R': int(near.sum()), 'artists_within_R': int(len(np.unique(artist[near & (artist != '')])))}
    if cards:
        out['nearest'] = card(o[0], OUT, float(d[o[0]]))
        out['nearest_years'] = [None if np.isnan(year[i]) else int(year[i]) for i in o[:10]]
    return out


def references(rng):
    """R from 2,000 real works, each against works by other artists; and random midpoints of the record's span, as context."""
    sample = rng.choice(np.flatnonzero(usable), 2000, replace=False)
    D = np.sqrt(np.maximum(2 - 2 * (X[sample] @ X.T), 0))
    for i, s in enumerate(sample):
        D[i, artist == artist[s]] = np.inf    # not itself, not its own artist
    Ds = np.sort(D, axis=1)
    R = float(np.median(Ds[:, 9]))
    pairs = rng.choice(np.flatnonzero(usable), (500, 2))
    mids = X[pairs].mean(1)
    mids /= np.linalg.norm(mids, axis=1, keepdims=True)
    Dm = np.sort(np.sqrt(np.maximum(2 - 2 * (mids @ X.T), 0)), axis=1)
    return R, {'d10': np.percentile(Ds[:, 9], [10, 50, 90]).tolist(), 'd100': np.percentile(Ds[:, 99], [10, 50, 90]).tolist()}, {'d10': np.percentile(Dm[:, 9], [10, 50, 90]).tolist(), 'within_R': np.percentile((Dm < R).sum(1), [10, 50, 90]).tolist()}


def one(cap, start_kind, seed):
    """One setting and seed: both routes, the chords, the hub numbers, and the verdict."""
    rng = np.random.default_rng([CUT, cap, zlib.crc32(start_kind.encode()), seed])    # the same draws whatever the argument order
    cards = seed == 0
    R, works_ref, random_ref = references(rng)
    start_rows = np.flatnonzero(usable & (year < CUT) & ((year >= CUT - 20) if start_kind == 'equal' else True))
    start, n_start = centre(start_rows, cap, rng)
    end, n_end = centre(np.flatnonzero(usable & (year >= W0) & (year <= W1)), cap, rng)
    decades = list(range(CUT, W0, 10))
    centres = [centre(np.flatnonzero(usable & (year >= dc) & (year < dc + 10)), cap, rng) for dc in decades]
    pts = [(start, 'start', n_start)] + [(c, f'{dc}s', n) for (c, n), dc in zip(centres, decades)] + [(end, 'end', n_end)]
    axis = end - start
    progress = lambda p: float((p - start) @ axis / (axis @ axis))    # noqa: E731
    straight = [{'t': float(t), 'progress': progress(unit((1 - t) * start + t * end)), **look(unit((1 - t) * start + t * end), R, cards)} for t in np.linspace(0, 1, STEPS + 1)]
    path = []
    for i, (p, label, n) in enumerate(pts):
        path.append({'label': label, 'n_behind': n, 'progress': progress(p), **look(p, R, cards)})
        if i + 1 < len(pts):
            q = unit((p + pts[i + 1][0]) / 2)
            path.append({'label': 'between', 'progress': progress(q), **look(q, R, cards)})
    chords = [{'from': pts[i][1], 'to': pts[j][1], 'length': float(np.linalg.norm(pts[i][0] - pts[j][0])), **look(unit((pts[i][0] + pts[j][0]) / 2), R, False)}
              for i in range(len(pts)) for j in range(i + 2, len(pts))]
    glob, _ = centre(np.flatnonzero(usable), cap, rng)
    hub = {'global': look(glob, R, False), 'centre_to_global': {label: float(np.linalg.norm(p - glob)) for p, label, _ in pts}}
    seg = [pts[i + 1][0] - pts[i][0] for i in range(len(pts) - 1)]
    turn = [float(np.degrees(np.arccos(np.clip(seg[i] @ seg[i + 1] / np.linalg.norm(seg[i]) / np.linalg.norm(seg[i + 1]), -1, 1)))) for i in range(len(seg) - 1)]
    first = 2 if start_kind == 'full' else 1    # with the whole record as the start, the first segment spans centuries: not a comparable turn
    e = first + int(np.argmax(turn[first - 1:]))    # the elbow: the centre where the path turns most
    elbow = pts[e][1]
    across = next(c for c in chords if c['from'] == pts[e - 1][1] and c['to'] == pts[e + 1][1])    # the chord that bypasses it
    pp = np.array([s['progress'] for s in path]); order = np.argsort(pp)
    ratio = [s['within_R'] / max(1.0, np.interp(s['progress'], pp[order], np.array([x['within_R'] for x in path])[order])) for s in straight if s['progress'] >= 0.5]
    mid = min(straight, key=lambda s: abs(s['t'] - 0.5))
    L = float(np.linalg.norm(axis))
    like = [c['d10'] for c in chords if abs(c['length'] - L) <= 0.15 * L and not (c['from'] == 'start' and c['to'] == 'end')]
    a_ratio = across['within_R'] / max(1, path[2 * e]['within_R'])
    verdict = {'elbow': elbow, 'turn_degrees': turn, 'a_chord_across_elbow_not_thinner': bool(a_ratio >= 0.9), 'a_ratio': float(a_ratio), 'a_within_R_chord_vs_elbow': [across['within_R'], path[2 * e]['within_R']],
               'b_second_half_not_thinner': bool(min(ratio) >= 0.9), 'b_min_ratio': float(min(ratio)),
               'c_midpoint_on_chord_curve': (bool(mid['d10'] <= max(like)) if len(like) >= 2 else None), 'c_n_like': len(like), 'c_midpoint_d10_vs_like_chords': [mid['d10'], [round(v, 3) for v in like]]}
    return {'seed': seed, 'R': R, 'works_ref': works_ref, 'random_ref': random_ref, 'start_to_end': L, 'straight': straight, 'path': path, 'chords': chords, 'hub': hub, 'verdict': verdict}


def summarise(runs):
    """Mean and sd over seeds for every number; cards from seed 0."""
    def agg(key_fn):
        v = np.array([key_fn(r) for r in runs], dtype=float)
        return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1))}
    out = {'R': agg(lambda r: r['R']), 'start_to_end': agg(lambda r: r['start_to_end']), 'works_ref_d10_median': agg(lambda r: r['works_ref']['d10'][1]), 'random_ref_within_R_median': agg(lambda r: r['random_ref']['within_R'][1])}
    for route in ('straight', 'path', 'chords'):
        out[route] = []
        for i, s0 in enumerate(runs[0][route]):
            item = {k: s0[k] for k in s0 if k in ('t', 'label', 'from', 'to', 'nearest', 'nearest_years')}
            for k in ('progress', 'n_behind', 'length', 'd10', 'd100', 'within_R', 'artists_within_R'):
                if k in s0:
                    item[k] = agg(lambda r: r[route][i][k])
            out[route].append(item)
    out['hub'] = {'global_within_R': agg(lambda r: r['hub']['global']['within_R']), 'global_artists_within_R': agg(lambda r: r['hub']['global']['artists_within_R']), 'global_d10': agg(lambda r: r['hub']['global']['d10']),
                  'centre_to_global': {lab: agg(lambda r: r['hub']['centre_to_global'][lab]) for lab in runs[0]['hub']['centre_to_global']}}
    out['verdict'] = {'elbow': [r['verdict']['elbow'] for r in runs], 'a': [r['verdict']['a_chord_across_elbow_not_thinner'] for r in runs], 'b': [r['verdict']['b_second_half_not_thinner'] for r in runs], 'c': [r['verdict']['c_midpoint_on_chord_curve'] for r in runs],
                      'a_ratio': [r['verdict']['a_ratio'] for r in runs], 'b_min_ratio': [r['verdict']['b_min_ratio'] for r in runs], 'a_counts': [r['verdict']['a_within_R_chord_vs_elbow'] for r in runs],
                      'c_n_like': [r['verdict']['c_n_like'] for r in runs], 'c_numbers': [r['verdict']['c_midpoint_d10_vs_like_chords'] for r in runs]}
    return out


result = {'cut': CUT, 'window': [W0, W1], 'steps': STEPS, 'seeds': SEEDS, 'caps': CAPS, 'starts': STARTS, 'settings': {}}
for cap in CAPS:
    for start_kind in STARTS:
        runs = [one(cap, start_kind, seed) for seed in range(SEEDS)]
        s = summarise(runs)
        result['settings'][f'cap{cap}_{start_kind}'] = s
        json.dump(result, open(f'{OUT}/corner.json', 'w'), indent=1)    # after each setting
        v = s['verdict']
        print(f"cap {cap or 'none'}, start {start_kind}: elbow {v['elbow'][0]}; (a) chord across elbow not thinner {sum(v['a'])}/{SEEDS} (ratio {min(v['a_ratio']):.2f}); (b) second half not thinner {sum(v['b'])}/{SEEDS} (min ratio {min(v['b_min_ratio']):.2f}); "
              f"(c) midpoint on chord curve {sum(1 for x in v['c'] if x)}/{SEEDS}{' untestable' if v['c'][0] is None else ''}; R {s['R']['mean']:.3f}±{s['R']['sd']:.3f}; global centre within R {s['hub']['global_within_R']['mean']:.0f}; "
              f"path within R: {' '.join('%s %.0f' % (p['label'], p['within_R']['mean']) for p in s['path'] if p['label'] != 'between')}", flush=True)
print('done', flush=True)
