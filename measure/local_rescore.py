"""Locally scaled threshold: re-scores existing pictures by r(x) = d(x) / s(x), x's mean chord to its K nearest reference pictures over the median own-distance of those neighbours; far = over the threshold in both observers. CPU.
Usage: python local_rescore.py OUT_ROOT [LABEL=RUN_DIR ...]"""
import os
import sys

import numpy as np

OBS, K = ('dinov3-l', 'siglip-so400m'), 15
BARS = (0.9614356994985623, 0.9000611238334555)
FID, FID_LOOP = 0.778528646826744, 0.8735
unit = lambda X: np.asarray(X, np.float64) / np.linalg.norm(np.asarray(X, np.float64), axis=-1, keepdims=True)  # noqa: E731
chord = lambda c: np.sqrt(np.clip(2 - 2 * c, 0, None))  # noqa: E731


def reference(Pz):
    art = np.array([str(a) for a in Pz['artist']])
    ref = {'artist': art}
    for o, bar in zip(OBS, BARS):
        E = unit(Pz[f'emb_{o}'])
        D = chord(E @ E.T)
        D[art[:, None] == art[None, :]] = np.inf          # leave-one-out and same artist excluded (the diagonal is same-artist)
        order = np.argsort(D, axis=1)[:, :K + 1]
        near = np.take_along_axis(D, order, 1)             # each row's K + 1 nearest distances
        nn, own = order[:, :K], near[:, :K].mean(1)
        s = np.empty(len(own))
        for i in range(len(own)):                          # strict: own of each neighbour j recomputed without i
            js = nn[i]
            hit = (nn[js] == i).any(1)
            s[i] = np.median(np.where(hit, (near[js].sum(1) - D[js, i]) / K, own[js]))
        ref[o] = dict(E=E, own=own, s=s, r=own / s, r_loose=own / np.median(own[nn], axis=1), bar=bar)
    N = int(np.all([ref[o]['own'] > ref[o]['bar'] for o in OBS], axis=0).sum())
    joint = lambda p: int(np.all([ref[o]['r'] > np.percentile(ref[o]['r'], p) for o in OBS], axis=0).sum())  # noqa: E731
    pm = next(p for p in np.arange(99.95, 50, -0.05) if joint(p) >= N)
    for o in OBS:
        ref[o]['L99'], ref[o]['matched'] = float(np.percentile(ref[o]['r'], 99)), float(np.percentile(ref[o]['r'], pm))
        print(f'reference {o}: global threshold {ref[o]["bar"]:.4f}; strict r median {np.median(ref[o]["r"]):.3f}, 99th percentile {ref[o]["L99"]:.4f} '
              f'(loose version: median {np.median(ref[o]["r_loose"]):.3f}, 99th percentile {np.percentile(ref[o]["r_loose"], 99):.4f}); matched threshold {ref[o]["matched"]:.4f}; '
              f's quartiles {np.percentile(ref[o]["s"], 25):.3f} / {np.percentile(ref[o]["s"], 50):.3f} / {np.percentile(ref[o]["s"], 75):.3f}')
    print(f'reference pictures over both thresholds: global N = {N}; local at the 99th percentile {joint(99)}; local at the matched percentile {pm:.2f}: M = {joint(pm)}')
    return ref


def score(ref, emb, artist=None):
    """emb: mapping with emb_<observer>; artist: if given, same-artist reference rows are excluded. Returns far, inh (rules G, L99, matched -> bool) and per observer (d, r, s)."""
    far, inh, per = {k: True for k in ('G', 'L99', 'matched')}, {k: False for k in ('G', 'L99', 'matched')}, {}
    for o in OBS:
        D = chord(unit(emb[f'emb_{o}']) @ ref[o]['E'].T)
        if artist is not None:
            D[np.array([str(a) for a in artist])[:, None] == ref['artist'][None, :]] = np.inf
        nn = np.argsort(D, axis=1)[:, :K]
        d = np.take_along_axis(D, nn, 1).mean(1)
        s = np.median(ref[o]['own'][nn], axis=1)
        r, first = d / s, nn[:, 0]
        per[o] = (d, r, s)
        far['G'], inh['G'] = far['G'] & (d > ref[o]['bar']), inh['G'] | (ref[o]['own'][first] > ref[o]['bar'])
        for k in ('L99', 'matched'):
            far[k], inh[k] = far[k] & (r > ref[o][k]), inh[k] | (ref[o]['r'][first] > ref[o][k])
    return far, inh, per


def line(name, far, inh, ok=None, tag=''):
    n = len(far['G'])
    if n == 0:
        print(f'   {name:28s} n 0')
        return
    c = lambda m: int(np.sum(m))  # noqa: E731
    s = f'   {name:28s} n {n:4d}'
    for k, lab in (('G', 'far-global'), ('L99', 'far-local 99th'), ('matched', 'far-local matched')):
        s += f' | {lab} {c(far[k]):4d} ({100 * c(far[k]) / n:5.1f}%), not inherited {c(far[k] & ~inh[k]):4d}'
    g = c(far['G'])
    s += f' | global and matched {c(far["G"] & far["matched"]):4d} = {100 * c(far["G"] & far["matched"]) / g if g else float("nan"):5.1f}% of far-global; matched only {c(far["matched"] & ~far["G"]):4d}'
    if ok is not None:
        s += (f' | faithful{tag} {c(ok):4d}: with far-global {c(far["G"] & ok):4d}, with far-local 99th {c(far["L99"] & ok):4d}, with far-local matched {c(far["matched"] & ok):4d} '
              f'(not inherited {c(far["matched"] & ok & ~inh["matched"]):4d}), with global and matched {c(far["G"] & far["matched"] & ok):4d}')
    print(s, flush=True)


def auc(a, b):
    """P(a random member of a scores above a random member of b), ties counted half."""
    b = np.sort(b)
    return float((np.searchsorted(b, a, 'left') + np.searchsorted(b, a, 'right')).sum() / (2 * len(a) * len(b)))


def pct_min(ref, per, key, idx):
    """minimum over the two observers of the picture's percentile in the reference's own distribution of that score (own for d, strict r for r)."""
    return np.min([np.searchsorted(np.sort(ref[o][key]), per[o][idx]) / len(ref[o][key]) for o in OBS], axis=0)


def run(ref, name, d, exact_emb=None):
    f = f'{d}/all_candidates.npz' if os.path.exists(f'{d}/all_candidates.npz') else f'{d}/all_draws.npz'
    z = np.load(f, allow_pickle=True)
    m = z['iteration'] >= 0
    far, inh, _ = score(ref, {f'emb_{o}': z[f'emb_{o}'][m] for o in OBS})
    if exact_emb is not None:
        e = np.load(exact_emb, allow_pickle=True)
        assert len(z['paths']) == len(e['paths']) and all(str(x) == str(y) for x, y in zip(z['paths'], e['paths'])), name
        ok, tag = chord((unit(z['style_embs']) * unit(e['emb_clip-vit-l-14'])).sum(1))[m] <= FID, ''
    else:
        ok, tag = z['fidelity_clip'][m] <= FID_LOOP, ' (LOOP)'
    adm = z['admitted'][m].astype(bool)
    inc = '' if (int(z['iteration'].max()) + 1 == 60 and m.sum() == 1920) else ' INCOMPLETE'
    line(f'{name} all{inc}', far, inh, ok, tag)
    sub = lambda x: {k: v[adm] for k, v in x.items()}  # noqa: E731
    line(f'{name} admitted{inc}', sub(far), sub(inh), ok[adm], tag)


if __name__ == '__main__':
    O = sys.argv[1]
    Pz = np.load(f'{O}/push_emb.npz', allow_pickle=True)
    ref = reference(Pz)
    Pv = unit(Pz['vec'])

    for rule, use_artist in (('no artist removal', False), ('same-artist reference rows excluded', True)):
        print(f'\nreal works, {rule}: positive control = 512 works of 1900 to 1969; calibration = 1,000 other pre-1900 works')
        got = {}
        for name, f in (('positive control', 'pos_emb.npz'), ('calibration', 'push2_emb.npz')):
            z = np.load(f'{O}/{f}', allow_pickle=True)
            far, inh, per = score(ref, z, z['artist'] if use_artist else None)
            ok = chord((unit(z['vec']) * unit(z['emb_clip-vit-l-14'])).sum(1)) <= FID
            line(name, far, inh, ok)
            got[name] = (far, per, ok)
        (fp, pp, okp), (_, pc, _) = got['positive control'], got['calibration']
        print('   AUC, later works above calibration works, on the minimum over observers of the percentile in the reference: '
              f'under d {auc(pct_min(ref, pp, "own", 0), pct_min(ref, pc, "own", 0)):.3f}; under r {auc(pct_min(ref, pp, "r", 1), pct_min(ref, pc, "r", 1)):.3f}')
        sel = fp['G'] & okp
        for o in OBS:
            ps = lambda v: 100 * float((ref[o]['s'] < np.median(v)).mean())  # noqa: E731
            print(f'   s (local spacing) {o}: median of the {int(sel.sum())} far-global faithful later works {np.median(pp[o][2][sel]):.3f} = reference percentile {ps(pp[o][2][sel]):.0f}; '
                  f'the other {int((~sel).sum())} later works {np.median(pp[o][2][~sel]):.3f} = {ps(pp[o][2][~sel]):.0f}; calibration {np.median(pc[o][2]):.3f} = {ps(pc[o][2]):.0f}; '
                  f'reference 75th percentile {np.percentile(ref[o]["s"], 75):.3f}')

    print('\nextra runs (LOOP fidelity)')
    for arg in sys.argv[2:]:
        lab, d = arg.split('=', 1)
        run(ref, lab, d)

    m, e = np.load(f'{O}/baselines/chains/manifest.npz', allow_pickle=True), np.load(f'{O}/baselines/chains/emb.npz', allow_pickle=True)
    assert len(m['paths']) == len(e['paths']) and all(str(x) == str(y) for x, y in zip(m['paths'], e['paths']))
    ok = chord((unit(m['v']) * unit(e['emb_clip-vit-l-14'])).sum(1)) <= FID
    steps, tags = np.array([int(x) for x in m['step']]), np.array([str(t) for t in m['row']])
    assert all(((tags == t) & (steps == 0)).sum() == 1 for t in set(tags)), 'expected one step-0 frame per chain'
    src_art = {}
    for t in sorted(set(tags)):                            # the chain's source = the reference work whose request its step-0 request is
        v0 = unit(m['v'][(tags == t) & (steps == 0)][0])
        j = int(np.argmax(Pv @ v0))
        assert Pv[j] @ v0 > 0.999, (t, float(Pv[j] @ v0))
        src_art[t] = ref['artist'][j]
    for what, art in (('source work left in the reference', None), ("source work's artist excluded", np.array([src_art[t] for t in tags]))):
        print(f'\nchains, no search (48 works by step; exact fidelity to the step\'s own request); {what}')
        far, inh, _ = score(ref, e, art)
        for s in sorted(set(steps)):
            k = steps == s
            line(f'chain step {s}', {a: b[k] for a, b in far.items()}, {a: b[k] for a, b in inh.items()}, ok[k])
