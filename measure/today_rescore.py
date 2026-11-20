"""Re-scores existing pictures against a reference of every date ("today", push_today_emb.npz) beside the 1900 reference (push_emb.npz): far, coverage and density losses, matched thresholds, chains. CPU.
Usage: python today_rescore.py OUT_ROOT [LABEL=RUN_DIR ...]"""
import os
import re
import sys

import numpy as np

from local_rescore import FID, FID_LOOP, K, OBS, chord, unit

pid = lambda paths: np.array([int(re.search(r'push_(\d+)\.png$', str(p)).group(1)) for p in paths])  # noqa: E731


def reference(z, year=None):
    art = np.array([str(a) for a in z['artist']])
    ref = {'artist': art, 'year': year, 'faithful': np.asarray(z['fidelity_clip'], float) <= FID}
    over = np.ones(len(art), bool)
    for o in OBS:
        E = unit(z[f'emb_{o}'])
        D = chord(E @ E.T)
        D[art[:, None] == art[None, :]] = np.inf          # leave-one-out, same artist excluded (the diagonal is same-artist)
        own = np.sort(D, axis=1)[:, :K].mean(1)
        ref[o] = dict(E=E, own=own, bar=float(np.percentile(own, 99)))
        over &= own > ref[o]['bar']
    ref['over'] = over
    return ref


def score(ref, emb, artist=None):
    """per observer: (d = mean chord to the K nearest reference pictures, their rows); same-artist reference rows excluded if artist given."""
    out = {}
    for o in OBS:
        D = chord(unit(emb[f'emb_{o}']) @ ref[o]['E'].T)
        if artist is not None:
            D[np.array([str(a) for a in artist])[:, None] == ref['artist'][None, :]] = np.inf
        nn = np.argsort(D, axis=1)[:, :K]
        out[o] = (np.take_along_axis(D, nn, 1).mean(1), nn)
    return out


far = lambda per, ref, key='bar': np.all([per[o][0] > ref[o][key] for o in OBS], axis=0)  # noqa: E731


def match(T, N):
    """The highest common percentile at which at least N today-reference pictures are over it in both observers; stores T[o]['mbar'], returns the percentile and its count M."""
    joint = lambda p: int(np.all([T[o]['own'] > np.percentile(T[o]['own'], p) for o in OBS], axis=0).sum())  # noqa: E731
    pm = next(p for p in np.arange(99.95, 50, -0.05) if joint(p) >= N)
    for o in OBS:
        T[o]['mbar'] = float(np.percentile(T[o]['own'], pm))
    return pm, joint(pm)


def split(name, sP, sT, P, T, ok=None, tag=''):
    n = len(sP[OBS[0]][0])
    if n == 0:
        print(f'   {name:30s} n 0')
        return
    f19, ftd, fold = far(sP, P), far(sT, T), far(sT, P)   # fold: far from the today reference at the 1900 thresholds
    fm = far(sT, T, 'mbar')                               # far from the today reference at its matched thresholds
    lost, gain = f19 & ~ftd, ~f19 & ftd
    cov, den = lost & ~fold, lost & fold
    later = np.mean([(T['year'][sT[o][1]] >= 1900).mean(1) for o in OBS], axis=0)
    c = lambda m: int(np.sum(m))  # noqa: E731
    s = (f'   {name:30s} n {n:4d} | far-1900 {c(f19):4d} | of those far-today {c(f19 & ftd):4d} ({100 * c(f19 & ftd) / c(f19) if c(f19) else float("nan"):5.1f}%) | '
         f'lost {c(lost):4d} = coverage {c(cov):4d} (more than half of today-neighbours from 1900 on: {c(cov & (later > 0.5)):4d}; median share {np.median(later[cov]) if c(cov) else float("nan"):.2f}) '
         f'+ density {c(den):4d} | gained {c(gain):4d} | far-today in all {c(ftd):4d} '
         f'| THREE RULES, of the far-1900: far from today at today\'s 99th {c(f19 & ftd):4d}, at the matched thresholds {c(f19 & fm):4d}, at the 1900 thresholds {c(f19 & fold):4d}; '
         f'in all at matched {c(fm):4d}')
    if ok is not None:
        s += (f' | faithful{tag} {c(ok):4d}: far-1900 {c(f19 & ok):4d}, of those far-today {c(f19 & ftd & ok):4d}, far-today {c(ftd & ok):4d}; of the far-1900 and faithful, '
              f'far from today at matched {c(f19 & fm & ok):4d}, at the 1900 thresholds {c(f19 & fold & ok):4d}')
    print(s, flush=True)


def run(P, T, name, d, exact_emb=None):
    f = f'{d}/all_candidates.npz' if os.path.exists(f'{d}/all_candidates.npz') else f'{d}/all_draws.npz'
    z = np.load(f, allow_pickle=True)
    m = z['iteration'] >= 0
    emb = {f'emb_{o}': z[f'emb_{o}'][m] for o in OBS}
    sP, sT = score(P, emb), score(T, emb)
    if exact_emb is not None:
        e = np.load(exact_emb, allow_pickle=True)
        assert len(z['paths']) == len(e['paths']) and all(str(x) == str(y) for x, y in zip(z['paths'], e['paths'])), name
        ok, tag = chord((unit(z['style_embs']) * unit(e['emb_clip-vit-l-14'])).sum(1))[m] <= FID, ''
    else:
        ok, tag = z['fidelity_clip'][m] <= FID_LOOP, ' (LOOP)'
    adm = z['admitted'][m].astype(bool)
    inc = '' if (int(z['iteration'].max()) + 1 == 60 and m.sum() == 1920) else ' INCOMPLETE'
    sub = lambda s, k: {o: (s[o][0][k], s[o][1][k]) for o in OBS}  # noqa: E731
    split(f'{name} admitted{inc}', sub(sP, adm), sub(sT, adm), P, T, ok[adm], tag)
    split(f'{name} all{inc}', sP, sT, P, T, ok, tag)


if __name__ == '__main__':
    O = sys.argv[1]
    Pz, Tz = np.load(f'{O}/push_emb.npz', allow_pickle=True), np.load(f'{O}/push_today_emb.npz', allow_pickle=True)
    pool = np.load(f'{O}/pools/today_cap3.npz', allow_pickle=True)
    yr = dict(zip(pool['ids'].astype(int).tolist(), pool['year'].astype(float).tolist()))
    tid = pid(Tz['paths'])
    assert len(tid) == 2048 and len(set(tid.tolist())) == 2048 and all(i in yr for i in tid.tolist())
    P, T = reference(Pz), reference(Tz, np.array([yr[i] for i in tid.tolist()]))
    for o in OBS:
        print(f'thresholds {o}: 1900 reference {P[o]["bar"]:.4f} (median {np.median(P[o]["own"]):.3f}, spread MAD {np.median(np.abs(P[o]["own"] - np.median(P[o]["own"]))):.4f}); '
              f'today reference {T[o]["bar"]:.4f} (median {np.median(T[o]["own"]):.3f}, spread MAD {np.median(np.abs(T[o]["own"] - np.median(T[o]["own"]))):.4f})')
    assert not np.isnan(T['year']).any()
    pm, M = match(T, int(P['over'].sum()))
    for o in OBS:
        P[o]['mbar'] = P[o]['bar']
    print(f'matched thresholds for the today reference: percentile {pm:.2f}, {M} today-reference pictures over both (the 1900 reference has {int(P["over"].sum())} '
          f'over both of its 99th percentiles): ' + ', '.join(f'{o} {T[o]["mbar"]:.4f}' for o in OBS))
    old, new = T['year'] < 1900, T['year'] >= 1900
    dec, cnt = np.unique((T['year'] // 10 * 10).astype(int), return_counts=True)
    print('today reference by decade: ' + ', '.join(f'{d}s {n}' for d, n in zip(dec.tolist(), cnt.tolist())))
    print(f'today reference: {int(old.sum())} rows before 1900, {int(new.sum())} from 1900 on; {int(np.isin(tid, pid(Pz["paths"])).sum())} works are also in P; faithful rows {int(T["faithful"].sum())}')
    print(f'reference pictures over both of their own thresholds (leave-one-out): 1900 reference {int(P["over"].sum())} of {len(P["over"])}; today reference {int(T["over"].sum())} of 2048 = '
          f'before 1900 {int((T["over"] & old).sum())} of {int(old.sum())} (faithful {int((T["over"] & old & T["faithful"]).sum())}), from 1900 on {int((T["over"] & new).sum())} of {int(new.sum())} '
          f'(faithful {int((T["over"] & new & T["faithful"]).sum())})')

    for rule, use in (('no artist removal', False), ('same-artist reference rows excluded', True)):
        print(f'\nreal works, {rule}')
        for name, f in (('later works 1900 to 1969', 'pos_emb.npz'), ('earlier works (calibration)', 'push2_emb.npz')):
            z = np.load(f'{O}/{f}', allow_pickle=True)
            a = z['artist'] if use else None
            ok = chord((unit(z['vec']) * unit(z['emb_clip-vit-l-14'])).sum(1)) <= FID
            split(name, score(P, z, a), score(T, z, a), P, T, ok)

    print('\nextra runs (LOOP fidelity)')
    for arg in sys.argv[2:]:
        lab, d = arg.split('=', 1)
        run(P, T, lab, d)

    m, e = np.load(f'{O}/baselines/chains/manifest.npz', allow_pickle=True), np.load(f'{O}/baselines/chains/emb.npz', allow_pickle=True)
    assert len(m['paths']) == len(e['paths']) and all(str(x) == str(y) for x, y in zip(m['paths'], e['paths']))
    ok = chord((unit(m['v']) * unit(e['emb_clip-vit-l-14'])).sum(1)) <= FID
    steps, rows, Pv = np.array([int(x) for x in m['step']]), np.array([str(t) for t in m['row']]), unit(Pz['vec'])
    src = {}
    for t in sorted(set(rows)):                            # the chain's source = the P work whose request its step-0 request is
        k = (rows == t) & (steps == 0)
        assert k.sum() == 1
        j = int(np.argmax(Pv @ unit(m['v'][k][0])))
        assert Pv[j] @ unit(m['v'][k][0]) > 0.999
        src[t] = P['artist'][j]
    for what, art in (('source work left in the 1900 reference', None), ("source work's artist excluded from both references", np.array([src[t] for t in rows]))):
        print(f'\nchains, no search, by step; {what}')
        sP, sT = score(P, e, art), score(T, e, art)
        for s in sorted(set(steps.tolist())):
            k = steps == s
            split(f'chain step {s}', {o: (sP[o][0][k], sP[o][1][k]) for o in OBS}, {o: (sT[o][0][k], sT[o][1][k]) for o in OBS}, P, T, ok[k])
