"""Fact lines under each painted picture of the galleries: the round, the nearest real work per observer with its distance against the manifold line, and whether a work dated before 1900 (on the manifold) or any work (off it) lies within the line. Reads data/picture_facts_v1.json.
Usage: imported by the gallery scripts"""
import json
import re

import search_finds_v1 as sf
from _style import *

FACTS = json.load(open(HERE / 'data' / 'picture_facts_v1.json', encoding='utf-8'))['pictures']


def pid_of(found_path):
    m = re.search(r'found_(.+)\.jpg$', found_path)
    assert m, found_path
    return m.group(1)


def obs_line(lab, x):
    w = lambda within: 'within the line' if within else 'beyond the line'  # noqa: E731
    if x['nearest_is_shown']:
        return f"{lab}: the work shown is the nearest real work, {x['shown_chord']:.3f} apart, {w(x['shown_within'])} ({x['line']:.3f})"
    n = x['nearest']
    return (f"{lab}: nearest real work {n['artist']}, {n['title']}, {n['year']}, {n['chord']:.3f} apart, {w(n['chord'] <= x['line'])} "
            f"({x['line']:.3f}); the work shown {x['shown_chord']:.3f}, {w(x['shown_within']).replace(' the line', '')}")


def fact_blocks(pid, large_leftover=False):
    f = FACTS[pid]
    order = [f['key_observer']] + [o for o in ('DINOv3', 'SigLIP') if o != f['key_observer']]
    blocks = [[(f"painted in round {f['round']}", False)]] + [[(obs_line(o, f['obs'][o]), False)] for o in order]
    if f['label'] == 'SINCE':
        assert not any(f['obs'][o]['pre_within'] for o in order), pid
        last = 'no work dated before 1900 within the line in either observer'
    else:
        assert f['label'] == 'NONE', pid
        last = 'no real work within the line in either observer'
    if large_leftover:
        last += '; large leftover share'
    return blocks + [[(last, False)]]


_orig = sf.credit_runs


def credit_runs(r, role):
    if role == 'found':
        return fact_blocks(pid_of(r['found']))
    return _orig(r, role)


sf.credit_runs = credit_runs   # Plate.credits looks this up at call time, for every gallery module
