"""Nearest-work split (FAR / NEAR / one reader) and large leftover on the far-and-faithful (FF) population, from onoff_v3's npz.
Usage (cwd = the search folder): python -B onoff_ff_v1.py OUTPREFIX -> OUTPREFIX.txt + OUTPREFIX.npz (refused if either exists)."""
import importlib.util
import os
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
R, L, E = f'{O}/runs', f'{B}/logs', f'{O}/eval'
R2 = f'{B}/tests12/t2/renders_v1'
OO = f'{B}/tests12/onoff/out'
EV_FILE = f'{C_}/controls_evaluate.py'
TR_FILE = f'{C_}/tr_readout_v2.py'
AB_NPZ = f'{OO}/onoff_v3_AB.npz'
CC_NPZ = f'{OO}/onoff_v3_C.npz'
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
STARTS = (6, 7, 8, 9, 10)


pid = lambda paths: np.array([int(re.search(r'push_(\d+)\.png$', str(p)).group(1)) for p in paths])
FH = [None]
c_ = lambda x: int(np.sum(x))


def say(s=''):
    print(s, flush=True)
    FH[0].write(s + '\n')
    FH[0].flush()


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(prefix):
    assert 'qgt' not in prefix
    for ext in ('.txt', '_STOP.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x')
    ev = load_mod('controls_evaluate', EV_FILE)
    tr = load_mod('tr_readout_v2', TR_FILE)
    A, Cz = np.load(AB_NPZ, allow_pickle=True), np.load(CC_NPZ, allow_pickle=True)
    say('onoff_ff_v1.py')
    say('FAR / NEAR / one = onoff_v3\'s nearest-work flags (bars D 0.762848 / S 0.830485); LL = onoff_v3\'s over the past\'s line in both readers.')
    say('FF (loop cells) = evaluator load F & ok_exact; FF (known answers) = grid_v1\'s far99,FID recipe. Rows aligned with onoff_v3 by path (asserted).')
    P, T, Pz = ev.references(O)
    OBS, K, unit, chord, FID = ev.OBS, ev.K, ev.unit, ev.chord, float(ev.FID)
    Pid, Part = P['id'], P['artist']
    lev = {o: float(np.percentile(P[o]['own'], 99)) for o in OBS}
    say(f'P bars (own 99th): ' + ', '.join(f'{o} {lev[o]:.6f}' for o in OBS) + f'; FID {FID:.6f}')

    def far_scores(emb, masks=None):   # grid_v1.far_scores
        d = {}
        for o in OBS:
            D = chord(unit(emb[o]) @ P[o]['E'].T)
            if masks is not None:
                for i, ks in enumerate(masks):
                    if len(ks):
                        D[i, ks] = np.inf
            d[o] = np.sort(D, axis=1)[:, :K].mean(1)
        return d

    far2 = lambda d: (d[OBS[0]] > lev[OBS[0]]) & (d[OBS[1]] > lev[OBS[1]])
    sv = {}
    G = {}
    row_in_P = {int(i): k for k, i in enumerate(Pid.tolist())}
    pools = {n: np.load(f'{O}/pools/{n}.npz', allow_pickle=True) for n in ('today_cap3',)}
    yr = dict(zip(pools['today_cap3']['ids'].astype(int).tolist(), pools['today_cap3']['year'].astype(float).tolist()))
    for name, f, key in (('P2', 'push2_emb.npz', 'P2'), ('pos', 'pos_emb.npz', 'pos'), ('today', 'push_today_emb.npz', 'today70')):
        z = np.load(f'{O}/{f}', allow_pickle=True)
        ids, art = pid(z['paths']), np.array([str(a) for a in z['artist']])
        inP = np.array([int(i) in row_in_P for i in ids])
        masks = [np.flatnonzero(Part == art[i]) if inP[i] else np.zeros(0, int) for i in range(len(ids))]
        ff = far2(far_scores({o: z[f'emb_{o}'] for o in OBS}, masks)) & (chord((unit(z['vec']) * unit(z['emb_clip-vit-l-14'])).sum(1)) <= FID)
        paths = np.array([str(p) for p in z['paths']])
        if name == 'today':
            s = np.array([yr[int(i)] for i in ids]) >= 1970
            ff, paths = ff[s], paths[s]
        assert np.array_equal(paths, A[f'{key}__paths']), f'{key}: rows differ from onoff_v3'
        G[key] = dict(FF=ff, FAR=A[f'{key}__FAR'], NEAR=A[f'{key}__NEAR'], LL=A[f'{key}__LL'])
    for sub, mf, kk, key in (('prior256_v1', 'prior/manifest.npz', 'kind', 'prior'), ('noise160_v1', 'noise/manifest.npz', 'tag', 'noise')):
        em = np.load(f'{R2}/{sub}/emb.npz', allow_pickle=True)
        kat = [ln.strip() for ln in open(f'{R2}/{sub}/inputs/expected_kat.txt') if ln.strip()]
        assert [str(p) for p in em['paths'][:len(kat)]] == kat
        k = len(kat)
        man = np.load(f'{R2}/{sub}/{mf}', allow_pickle=True)
        assert [str(p) for p in em['paths'][k:]] == [str(p) for p in man['paths']]
        ff = far2(far_scores({o: em[f'emb_{o}'][k:] for o in OBS})) & (chord((unit(man['v']) * unit(em['emb_clip-vit-l-14'][k:])).sum(1)) <= FID)
        assert np.array_equal(np.array([str(p) for p in em['paths'][k:]]), A[f'{key}__paths'])
        kind = np.array([str(x) for x in man[kk]])
        assert np.array_equal(kind, A[f'{key}__kind'])
        G[key] = dict(FF=ff, FAR=A[f'{key}__FAR'], NEAR=A[f'{key}__NEAR'], LL=A[f'{key}__LL'], kind=kind)
    ids, q = ev.startids(O), ev.qfun(O)
    cv = {}
    for arm, tg, _ in ARMS:
        for n in STARTS:
            tag = f'qgc_{tg}_s{n}'
            d = f'{R}/{tag}_committee_euclidean_min_seedrow'
            assert 'qgt' not in d
            r = ev.load(f'{tag}={d},{E}/{tag}_emb.npz,{L}/{tag}.out', P, T, ids, q, conf=True)
            st = ev.stats(r, 'exact')
            m = r['z']['iteration'] >= 0
            assert np.array_equal(np.array([str(p) for p in np.asarray(r['z']['paths'])[m]]), Cz[f'{tag}__paths']), f'{tag}: rows differ from onoff_v3'
            kept, hits = np.asarray(r['ev'], bool), np.asarray(st['DH'], bool)
            assert np.array_equal(kept, Cz[f'{tag}__kept']) and np.array_equal(hits, Cz[f'{tag}__hits'])
            cv[tag] = dict(FF=np.asarray(r['F'], bool) & np.asarray(r['ok_exact'], bool), FAR=Cz[f'{tag}__FAR'], NEAR=Cz[f'{tag}__NEAR'], LL=Cz[f'{tag}__LL'], kept=kept, hits=hits)
            sv[f'{tag}__FF'] = cv[tag]['FF']
    for key, g in G.items():
        sv[f'{key}__FF'] = g['FF']
    say()
    say('   FF per start: ' + '; '.join(f'{a} ' + str([c_(cv[f"qgc_{tg}_s{n}"]["FF"]) for n in STARTS]) for a, tg, _ in ARMS))
    say(f'   hits inside FF: ' + ', '.join(f'{a} {sum(c_(cv[f"qgc_{tg}_s{n}"]["hits"] & cv[f"qgc_{tg}_s{n}"]["FF"]) for n in STARTS)}' for a, tg, _ in ARMS)
        + '; kept inside FF: ' + ', '.join(f'{a} {sum(c_(cv[f"qgc_{tg}_s{n}"]["kept"] & cv[f"qgc_{tg}_s{n}"]["FF"]) for n in STARTS)}' for a, tg, _ in ARMS))

    def split(g, sel):
        F, N = g['FAR'] & sel, g['NEAR'] & sel
        one = sel & ~g['FAR'] & ~g['NEAR']
        return c_(N), c_(one), c_(F)

    def row(label, g, sel, rates=False):
        ff = g['FF'] & sel
        nb, no, fb = split(g, ff)
        lnb, lno, lfb = split(g, ff & g['LL'])
        unb, uno, ufb = split(g, ff & ~g['LL'])
        k = c_(ff)
        rt = f' | rates near both {nb / k:.3f}, near one {no / k:.3f}, far both {fb / k:.3f}' if rates and k else ''
        say(f'   {label:40s} n {c_(sel):5d} FF {k:5d} | near both / near one / far both {nb} / {no} / {fb} | LL {lnb} / {lno} / {lfb} | not LL {unb} / {uno} / {ufb}{rt}')

    cat = lambda cs, key: np.concatenate([c[key] for c in cs])
    say('PER LOOP, start and pooled: n | FF | among FF: near in both / near in one / far in both | the same among FF and LL | among FF and not LL '
        '(rates of FF, pooled rows of candidates)')
    for subset in ('candidates', 'kept', 'hits'):
        say(f' {subset}:')
        for arm, tg, w in ARMS:
            for n in STARTS + ('pooled',):
                cs = [cv[f'qgc_{tg}_s{k}'] for k in (STARTS if n == 'pooled' else (n,))]
                g = {k: cat(cs, k) for k in ('FF', 'FAR', 'NEAR', 'LL', 'kept', 'hits')}
                sel = np.ones(len(g['FF']), bool) if subset == 'candidates' else g[subset]
                row(f'{arm} {w}, {"s" + str(n) if n != "pooled" else "pooled"}, {subset}', g, sel, rates=(subset == 'candidates' and n == 'pooled'))
    say()
    say('KNOWN ANSWERS (source work masked as in onoff_v3): n | FF | among FF: near in both / near in one / far in both | among FF and LL | among FF and not LL')
    for key, lab, sel in (('P2', 'repaints, before 1900 held out (P2)', None), ('pos', 'repaints, 1900 to 1969 (pos)', None), ('today70', 'repaints, 1970 on (T rows)', None),
                          ('noise', 'noise requests, all', None)) + tuple(('noise', f'noise requests, kind {x}', ('kind', x)) for x in 'abcde') + (
                          ('prior', 'random requests, cone', ('kind', 'cone')), ('prior', 'random requests, sphere', ('kind', 'sphere')), ('prior', 'random requests, all', None)):
        g = G[key]
        s = np.ones(len(g['FF']), bool) if sel is None else (g[sel[0]] == sel[1])
        row(lab, g, s)
    say()
    say('RULE (tr_readout_v2.rule): "ahead" = more at all five starts and 4 x sum_search >= 5 x sum_control and, when the control\'s sum is 0, sum_search >= 10; '
        '"behind" the same way round; otherwise "cannot tell"')
    counts = {'FF and far in both': lambda c: c['FF'] & c['FAR'], 'FF and not far in both': lambda c: c['FF'] & ~c['FAR'],
              'FF and near in both': lambda c: c['FF'] & c['NEAR'], 'FF, LL and not far in both': lambda c: c['FF'] & c['LL'] & ~c['FAR']}
    for lab, fn in counts.items():
        v = [c_(fn(cv[f'qgc_ra_s{n}'])) for n in STARTS]
        for x, tg in (('C1', 'pin'), ('C2', 'c2')):
            xx = [c_(fn(cv[f'qgc_{tg}_s{n}'])) for n in STARTS]
            say(f'   {lab}: search vs {x}: search {v} = {sum(v)}, {x} {xx} = {sum(xx)} -> {tr.rule(v, xx).upper()}')
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'wrote {prefix}.txt and {prefix}.npz (per picture FF, aligned with onoff_v3 npz rows)')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
