"""Why meetreal_v1 finds no places: ball sizes, same-artist share of neighbours, a ladder of rules and a positive control, on the eligible set stored in the meetreal_v1 output.
Usage: python -B meetreal_diag_v1.py MEETREAL_NPZ OUTPREFIX"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meetreal_v1 as mr  # noqa: E402



def main(npz, prefix):
    t0 = time.time()
    for ext in ('.txt',):
        assert not os.path.exists(prefix + ext)
    mr.FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    say = mr.say
    t3 = mr.load_t3()
    say('meetreal_diag_v1.py')
    C = t3.load_corpus('real')
    z = np.load(npz, allow_pickle=True)
    test, el = z['test_rows'], z['el_rows']
    assert np.array_equal(test, np.flatnonzero(C['dated'] & (C['year'] >= mr.CUT)))
    OBS = t3.OBS
    X = {n: C['E'][n][el] for n in OBS}
    art = C['artist'][el]
    g = mr.artist_groups(C['artist'][test], mr.SEED)
    grp = np.array([g[x] for x in art], int)
    low = lambda s: s.strip().lower()  # noqa: E731
    unk = lambda arr: int(sum(1 for s in arr if low(s) in ('', 'unknown', 'anonymous', 'unknown artist', 'nan', 'none') or 'anonym' in low(s) or 'unknown' in low(s)))  # noqa: E731
    say(f'labels: blank / unknown / anonymous artist among test works {unk(C["artist"][test])} of {len(test)}; among eligible {unk(art)} of {len(el)}')
    vc = {}
    for s in art:
        vc[s] = vc.get(s, 0) + 1
    top = sorted(vc.items(), key=lambda kv: -kv[1])[:8]
    say(f'eligible works per artist: median {np.median(list(vc.values())):.0f}, max {top[0][1]}; top artists {top}')
    cut = np.asarray(z['theta']) * np.asarray(z['med'])
    say(f'E cut-off (theta x med) per observer {np.round(cut, 6)}; radii record {np.round(z["r_rec"], 6)}, originals {np.round(z["r_orig"], 6)}')
    # positive control: the same real_places call, grouping 0, on a seeded size-matched sample of the 1970-on works that fail E
    fail = test[~np.all(z['test_e'] > cut, axis=1)]
    assert len(fail) == len(test) - len(el)
    ctl = np.sort(np.random.default_rng(mr.SEED + 1000).choice(fail, len(el), replace=False))
    Xc = {n: C['E'][n][ctl] for n in OBS}
    gc = np.array([g[x] for x in C['artist'][ctl]], int)
    for rname, r in (('record', z['r_rec']), ('originals', z['r_orig'])):
        plc = mr.real_places(t3, Xc, gc, C['artist'][ctl], ctl, r, t3.Q, t3.MIN_ROUNDS, mr.M)
        plc = [p for p in plc if p['support'] >= mr.M]
        say(f'POSITIVE CONTROL {rname}: {len(ctl)} of the {len(fail)} 1970-on works that fail E (rng {mr.SEED + 1000}), grouping 0, the run\'s rule: '
            f'places {len(plc)}' + (f', largest support {max(p["support"] for p in plc)}, members of the first {len(plc[0]["members"])}' if plc else ''))
    for rname, r in (('record', z['r_rec']), ('originals', z['r_orig'])):
        say()
        say(f'RADIUS {rname} {np.round(r, 4)}')
        Tm = t3.together(X, X, r).tocsr()
        deg = np.diff(Tm.indptr) - 1
        say(f'  eligible works within r of an eligible work (itself excluded): median {np.median(deg):.0f}, p90 {np.percentile(deg, 90):.0f}, max {deg.max()}; '
            f'with >= 14 others {int((deg >= 14).sum())}')
        same, nart, ngrp, g5, g5r = [], [], [], [], []
        for i in range(len(el)):
            nb = Tm.indices[Tm.indptr[i]:Tm.indptr[i + 1]]
            o = nb[nb != i]
            if len(o):
                same.append(np.mean(art[o] == art[i]))
            nart.append(len(np.unique(art[nb])))
            gs = grp[nb]
            ngrp.append(len(np.unique(gs)))
            g5.append(sum(1 for k in range(5) if (gs == k).sum() >= 5))
            g5r.append(sum(1 for k in range(5) if (gs == k).sum() >= 5 and len(np.unique(art[nb][gs == k])) >= 2))
        nart, ngrp, g5, g5r = map(np.array, (nart, ngrp, g5, g5r))
        say(f'  same-artist share of ball neighbours: mean {np.mean(same):.3f}, median {np.median(same):.3f}')
        say(f'  distinct artists in a ball: median {np.median(nart):.0f}, max {nart.max()}; distinct groups: median {np.median(ngrp):.0f}, max {ngrp.max()}')
        say(f'  most groups with >= 5 works in one ball: {g5.max()} (balls with >= 3 such groups: {int((g5 >= 3).sum())}); '
            f'with >= 5 works from >= 2 artists: {g5r.max()} (balls with >= 3: {int((g5r >= 3).sum())})')
        rnd_art = np.unique(art, return_inverse=True)[1]
        rows_ = np.asarray(el, int)
        ladder = [('as run: groups, q 5, rounds (artists) >= 2', grp, 5, rnd_art, 2),
                  ('rounds rule off: groups, q 5', grp, 5, np.zeros(len(el), int), 1),
                  ('groups, q 1, no rounds', grp, 1, np.zeros(len(el), int), 1),
                  ('units = artists, q 1, no rounds', rnd_art, 1, np.zeros(len(el), int), 1)]
        for name, unit_, q, rnd, mrnd in ladder:
            pl = [p for p in t3.greedy(Tm, np.asarray(unit_, int), np.asarray(rnd, int), rows_, q, mrnd, mr.M) if p['support'] >= mr.M]
            say(f'  places (support >= {mr.M}) under "{name}": {len(pl)}' + (f'; largest support {max(p["support"] for p in pl)}' if pl else ''))
    say(f'done {time.time() - t0:.0f} s')
    mr.FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
