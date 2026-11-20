"""Far in AIMv2, the encoder that never entered the loop, on the stored pictures of the 15 main runs: of the pictures the two observers call far, how many are also far in AIMv2, and does the search still lead its controls.
Usage (cwd = the search folder): python -B aimv2_far_v1.py OUTPREFIX -> OUTPREFIX.txt + OUTPREFIX.npz (refused if either exists)."""
import importlib.util
import os
import sys
import time

import numpy as np

B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
R, L, E = f'{O}/runs', f'{B}/logs', f'{O}/eval'
EV_FILE = f'{C_}/controls_evaluate.py'
STARTS = (6, 7, 8, 9, 10)
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
ENC = ('dinov3-l', 'siglip-so400m', 'aimv2-huge')
K, Q = 15, 99
FH, OUT = [None], []


def say(s=''):
    OUT.append(str(s))
    print(s, flush=True)
    if FH[0] is not None:
        FH[0].write(str(s) + '\n')
        FH[0].flush()


def unit(X):
    X = np.asarray(X, np.float64)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def chord(c):
    return np.sqrt(np.clip(2 - 2 * c, 0, None))


def ref_bar(Pe, artist, k=K, q=Q):
    """baselines_score.py: P against itself, diagonal and same-artist rows excluded, mean of the k smallest chords; the q-th percentile"""
    D = chord(Pe @ Pe.T)
    np.fill_diagonal(D, np.inf)
    D[artist[:, None] == artist[None, :]] = np.inf
    return float(np.percentile(np.sort(D, axis=1)[:, :k].mean(1), q))


def score(X, Pe, k=K, chunk=1024):
    """baselines_score.py: an item's mean chord to its k nearest P rows (nothing masked)"""
    out = np.empty(len(X))
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = np.sort(chord(unit(X[i:i + chunk]) @ Pe.T), axis=1)[:, :k].mean(1)
    return out


def ahead(v, c):
    """more at every start and at least a quarter more in sum"""
    v, c = np.asarray(v), np.asarray(c)
    return bool((v > c).all() and v.sum() >= 1.25 * c.sum())


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(prefix):
    t0 = time.time()
    for ext in ('.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    ev = load_mod('controls_evaluate', EV_FILE)
    say(f'numpy {np.__version__}')
    P, T, Pz = ev.references(O)
    ids, q = ev.startids(O), ev.qfun(O)
    Pd = np.load(f'{O}/push_emb.npz', allow_pickle=True)
    art = np.asarray(Pd['artist']).astype(str)
    Pe = {n: unit(Pd[f'emb_{n}']) for n in ENC}
    bars = {n: ref_bar(Pe[n], art) for n in ENC}
    for n in ENC:
        say(f'   bar {n} (k {K}, q {Q}, same-artist excluded): {bars[n]:.6f}')
    pos, p2 = np.load(f'{O}/pos_emb.npz', allow_pickle=True), np.load(f'{O}/push2_emb.npz', allow_pickle=True)
    far_of = lambda z: {n: score(z[f'emb_{n}'], Pe[n]) > bars[n] for n in ENC}  # noqa: E731
    fp, f2 = far_of(pos), far_of(p2)
    both = fp['dinov3-l'] & fp['siglip-so400m']
    say(f'scale: AIMv2-far on the 512 repainted 1900-1969 {100 * fp["aimv2-huge"].mean():.1f}% ({int(fp["aimv2-huge"].sum())}), on the 1,000 held out before 1900 '
        f'{100 * f2["aimv2-huge"].mean():.1f}% ({int(f2["aimv2-huge"].sum())}); on the 512, AIMv2-far among both-observer-far {int((fp["aimv2-huge"] & both).sum())} of {int(both.sum())}, '
        f'among the rest {int((fp["aimv2-huge"] & ~both).sum())} of {int((~both).sum())}')
    cells = {}
    for arm, tg, _ in ARMS:
        for n in STARTS:
            tag = f'qgc_{tg}_s{n}'
            d = f'{R}/{tag}_committee_euclidean_min_seedrow'
            r = ev.load(f'{tag}={d},{E}/{tag}_emb.npz,{L}/{tag}.out', P, T, ids, q, conf=True)
            z = r['z']
            m = z['iteration'] >= 0
            F, ok = np.asarray(r['F'], bool), np.asarray(r['ok_exact'], bool)
            ex = np.load(f'{E}/{tag}_emb.npz', allow_pickle=True)
            paths = np.array([str(p) for p in np.asarray(z['paths'])[m]])
            assert np.array_equal(np.array([str(p) for p in np.asarray(ex['paths'])[m]]), paths), f'{tag}: eval rows are not the run rows'
            sa = score(np.asarray(ex['emb_aimv2-huge'])[m], Pe['aimv2-huge'])
            cells[tag] = dict(FF=F & ok, ok=ok, F=F, A=sa > bars['aimv2-huge'], sa=sa)
    say(f'AIMv2 bar {bars["aimv2-huge"]:.6f}')
    tab, sv = {}, {}
    for arm, tg, w in ARMS:
        rows = []
        for n in STARTS:
            c = cells[f'qgc_{tg}_s{n}']
            rows.append(dict(n=1920, FF=int(c['FF'].sum()), FFA=int((c['FF'] & c['A']).sum()), okA=int((c['ok'] & c['A']).sum()),
                             A=int(c['A'].sum()), ok=int(c['ok'].sum()), F=int(c['F'].sum()), FA=int((c['F'] & c['A']).sum()),
                             nFA=int((~c['F'] & c['A']).sum()), nF=int((~c['F']).sum())))
            sv[f'qgc_{tg}_s{n}__aim_score'] = c['sa']
        tab[arm] = rows
    say()
    say('PER START (counts of 1,920; FF = far in both observers and faithful; AIM = far in AIMv2)')
    for arm, tg, w in ARMS:
        for n, r in zip(STARTS, tab[arm]):
            say(f'   {arm} {w:18s} s{n}: FF {r["FF"]:5d}, FF&AIM {r["FFA"]:5d} (A {100 * r["FFA"] / max(r["FF"], 1):5.1f}%); faithful {r["ok"]:5d}, faithful&AIM {r["okA"]:5d}; '
                f'AIM {r["A"]:5d}; observers-far {r["F"]}, of them AIM {r["FA"]} ({100 * r["FA"] / max(r["F"], 1):.1f}%); observers-near AIM {r["nFA"]} of {r["nF"]}')
    say()
    say('POOLED (of 9,600 per loop)')
    S = {arm: {k: np.array([r[k] for r in tab[arm]]) for k in tab[arm][0]} for arm in tab}
    for arm, tg, w in ARMS:
        s = S[arm]
        say(f'   {arm} {w:18s}: A = {s["FFA"].sum()} of {s["FF"].sum()} FF also AIMv2-far = {100 * s["FFA"].sum() / s["FF"].sum():.1f}% '
            f'(by start {", ".join(f"{100 * a / max(b, 1):.0f}" for a, b in zip(s["FFA"], s["FF"]))}); '
            f'B = faithful & AIM {s["okA"].sum()} = {100 * s["okA"].sum() / 9600:.2f}%; C = FF & AIM {s["FFA"].sum()} = {100 * s["FFA"].sum() / 9600:.2f}%; '
            f'agreement: of observers-far, AIM {100 * s["FA"].sum() / s["F"].sum():.1f}%; of observers-near, AIM {100 * s["nFA"].sum() / s["nF"].sum():.1f}%')
    say()
    say('SEARCH AGAINST EACH CONTROL')
    for c_arm in ('C1', 'C2'):
        for key, nm in (('okA', 'B faithful & AIMv2-far'), ('FFA', 'C faithful & far in all 3'), ('FF', 'FF (observers), for reference')):
            v, c = S['V'][key], S[c_arm][key]
            say(f'   V vs {c_arm} {nm:34s}: {v.sum()} vs {c.sum()} = {v.sum() / max(c.sum(), 1):.2f}x; per start V {v.tolist()} vs {c.tolist()}; '
                f'more at {int((v > c).sum())} of 5 starts; ahead (more at every start and >= 1.25x in sum): {ahead(v, c)}')
    for arm in S:
        for k, v in S[arm].items():
            sv[f'{arm}__{k}'] = v
    sv.update({f'bar__{n}': np.array(bars[n]) for n in ENC})
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'saved {prefix}.npz; done {time.time() - t0:.0f} s')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
