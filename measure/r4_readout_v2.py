"""Counts of the CLIP-scored search and its random-score loop (qgcclip_ra, qgcclip_pin, starts 6 to 10) beside the main search and random-score loop (qgc_ra, qgc_pin): far in both observers and faithful, hits, far in CLIP and faithful, faithful, admitted, lean median, ahead.
Usage (cwd = the search folder): python -B r4_readout_v2.py OUTPREFIX [RUNS] -> OUTPREFIX.txt; RUNS = comma list like ra6,ra7,pin6 of the CLIP-scored runs to read (default all 10)."""
import importlib.util
import os
import re
import sys

import numpy as np

B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
EV_FILE, R3 = f'{C_}/controls_evaluate.py', f'{C_}/r3_clip_far_v1.py'
STARTS = (6, 7, 8, 9, 10)
X = 'clip-vit-l-14'
FH, OUT, CHK = [None], [], []


def say(s=''):
    OUT.append(str(s))
    print(s, flush=True)
    FH[0].write(str(s) + '\n')
    FH[0].flush()


def want(label, ok, got, exp):
    CHK.append(bool(ok))
    say(f'   CHECK {label}: got {got} | expected {exp} | {"ok" if ok else "FAILED"}')


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(prefix, runs=None):
    toks = None if runs is None else [t.strip() for t in runs.split(',')]
    assert toks is None or all(t in {f'{a}{n}' for a in ('ra', 'pin') for n in STARTS} for t in toks), f'bad RUNS {runs!r}'
    want_run = (lambda tg, n: True) if toks is None else (lambda tg, n: f'{tg}{n}' in toks)
    for ext in ('.txt', '_STOP.txt'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    ev = load_mod('controls_evaluate', EV_FILE)
    r3 = load_mod('r3', R3)
    say(f'numpy {np.__version__}')
    orig = ev.tag_role

    def tag_role(fl, d):
        t, role, n = orig(fl, d)
        mm = re.fullmatch(r'qgcclip_(ra|pin)_s([0-9]+)', t or '')
        return (t, ev.ROLE_OF[mm.group(1)], int(mm.group(2))) if mm else (t, role, n)
    ev.tag_role = tag_role   # the evaluator's tag rule knows only qgc_*; the CLIP-scored runs get the roles V and C1 by the same pattern
    P, T, Pz = ev.references(O)
    ids, q = ev.startids(O), ev.qfun(O)
    Pd = np.load(f'{O}/push_emb.npz', allow_pickle=True)
    Pe = r3.unit(Pd[f'emb_{X}'])
    bar = r3.ref_bar(Pe, np.asarray(Pd['artist']).astype(str))
    cells = {}

    def read(tag, d, exact, log):
        r = ev.load(f'{tag}={d},{exact},{log}', P, T, ids, q, conf=True)
        z = r['z']
        m = np.asarray(z['iteration']) >= 0
        F, ok = np.asarray(r['F'], bool), np.asarray(r['ok_exact'], bool)
        ex = np.load(exact, allow_pickle=True)
        sa = r3.score(np.asarray(ex[f'emb_{X}'])[m], Pe)
        DH = np.asarray(r['ev'], bool) & ok & np.asarray(r['C'], bool)
        cells[tag] = dict(FF=int((F & ok).sum()), DH=int(DH.sum()), CF=int(((sa > bar) & ok).sum()), ok=int(ok.sum()), lean=float(np.median(r['q_exact'])),
                          adm=int(np.asarray(z['admitted'])[m].sum()), card=r['card'][0], z=z)
    for tg in ('ra', 'pin'):
        for n in STARTS:
            tag = f'qgc_{tg}_s{n}'
            read(tag, f'{O}/runs/{tag}_committee_euclidean_min_seedrow', f'{O}/eval/{tag}_emb.npz', f'{B}/logs/{tag}.out')
    for tg in ('ra', 'pin'):
        for n in STARTS:
            tag = f'qgcclip_{tg}_s{n}'
            if not want_run(tg, n):
                continue
            d = f'{O}/runs/{tag}_committee_euclidean_min_seedrow'
            read(tag, d, f'{O}/eval/{tag}_emb.npz', f'{B}/logs/{tag}.out')
            z = cells[tag]['z']
            fl, it = str(z['flags']), np.asarray(z['iteration'])
            adm, feas = np.asarray(z['admitted']).astype(bool) & (it >= 0), np.asarray(z['feasible']).astype(bool)   # row 0 is the start: admitted, iteration -1, no raw_clip
            rawc = np.asarray(z[f'raw_{X}'], np.float64)
            ok3 = ("'score_observers': 'clip-vit-l-14'" in fl and len(it) == 1921 and bool(np.all(feas[adm] & (it[adm] >= 1) & (rawc[adm] > bar))))
            if tg == 'pin':   # the steps it took: --sigma-path is the candidate file of its own CLIP search
                sp = re.search(r"'sigma_path': '([^']*)'", fl)
                ok3 &= (sp is not None and '/'.join(sp.group(1).split('/')[-2:]) == f'qgcclip_ra_s{n}_committee_euclidean_min_seedrow/all_candidates.npz'
                        and "'random_fitness': True" in fl)
            want(f'{tag}: flags, rows, admitted rows over the CLIP bar' + (', steps = its search' if tg == 'pin' else '') + f' (card {cells[tag]["card"]})',
                 ok3, int(adm.sum()), 'all over the bar')
    if not all(CHK):
        say('STOP: a check failed.')
        FH[0].close()
        open(prefix + '_STOP.txt', 'x', encoding='utf-8').write('\n'.join(OUT) + '\n')
        sys.exit(3)
    say(f'   ALL {len(CHK)} CHECKS PASSED')
    say()
    say('per start s6..s10 | sum: FF (far in both observers, faithful), hits, CF (CLIP-far, faithful), faithful, admitted, lean median')
    rows = (('stored search', 'qgc_ra'), ('stored random-score', 'qgc_pin'), ('CLIP-scored search', 'qgcclip_ra'), ('its random-score loop', 'qgcclip_pin'))
    for lab, pre in rows:
        tags = [f'{pre}_s{n}' for n in STARTS]
        line = ' ; '.join(f'{k} ' + '/'.join(str(cells[t][k]) if t in cells else '-' for t in tags) + f' = {sum(cells[t][k] for t in tags if t in cells)}'
                          for k in ('FF', 'DH', 'CF', 'ok', 'adm'))
        say(f'   {lab:22s} {line} ; lean ' + '/'.join(f'{cells[t]["lean"]:+.3f}' if t in cells else '-' for t in tags))
    say()
    for k, nm in (('FF', 'far in both observers & faithful'), ('DH', 'hits'), ('CF', 'CLIP-far & faithful')):
        both = [n for n in STARTS if f'qgcclip_ra_s{n}' in cells and f'qgcclip_pin_s{n}' in cells]
        own = [n for n in STARTS if f'qgcclip_ra_s{n}' in cells]
        v = lambda pre, ns: [cells[f'{pre}_s{n}'][k] for n in ns]  # noqa: E731
        a, c, sa, sc = v('qgcclip_ra', both), v('qgcclip_pin', both), v('qgc_ra', STARTS), v('qgc_pin', STARTS)
        say(f'   {nm:34s} starts {own}: CLIP search {v("qgcclip_ra", own)} = {sum(v("qgcclip_ra", own))} | stored search same starts {v("qgc_ra", own)} = '
            f'{sum(v("qgc_ra", own))} | ratio {sum(v("qgcclip_ra", own)) / max(sum(v("qgc_ra", own)), 1):.2f}x')
        say(f'   {nm:34s} CLIP search / its random-score loop = {sum(a)}/{sum(c)} = {sum(a) / max(sum(c), 1):.2f}x, ahead {r3.ahead(a, c) if both else "n/a (no loop yet)"} | '
            f'stored search / stored random-score, all 5 starts = {sum(sa)}/{sum(sc)} = {sum(sa) / max(sum(sc), 1):.2f}x, ahead {r3.ahead(sa, sc)} | '
            f'(CLIP pair on starts {both})')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
