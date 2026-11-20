"""Lean baselines: the lean (controls_evaluate.qfun) of the exact CLIP reading for the reference sets, real repainted works, noise requests, the no-search pictures and the three loops (all, round 0, round 59), with the loops' request lean beside it.
Usage (cwd = the search folder): python -B r2_lean_v2.py OUTPREFIX -> OUTPREFIX.txt + OUTPREFIX.npz (refused if either exists)"""
import csv
import importlib.util
import os
import sys
import time

import numpy as np

B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
EV_FILE = f'{C_}/controls_evaluate.py'
ROWS_CSV = f'{B}/tests12/nosearch/nosearch_v5_rows.csv'
NOISE = f'{B}/tests12/t2/renders_v1/noise160_v1'
STARTS = (6, 7, 8, 9, 10)
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
KINDS = (('(480)', '35 deg steps past a work'), ('(36)', 'line between 2 regions\' centres'), ('(450)', 'lines past a real work'),
         ('(48)', 'midpoints of 2 works'), ('(64)', 'random, whole sphere'), ('(192)', 'random, inside the pre-1900 region'))
base = lambda p: os.path.basename(str(p))  # noqa: E731
FH, OUT = [None], []


def say(s=''):
    OUT.append(str(s))
    print(s, flush=True)
    if FH[0] is not None:
        FH[0].write(str(s) + '\n')
        FH[0].flush()



def stats(x):
    x = np.asarray(x, np.float64)
    return f'median {np.median(x):+.4f} p10 {np.percentile(x, 10):+.4f} p90 {np.percentile(x, 90):+.4f} n {len(x)}'


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shard_index(folder):
    """basename -> (request v, exact CLIP reading), over every shard of a stage"""
    out = {}
    for s in range(64):
        d = f'{folder}/shard{s}'
        if not os.path.isdir(d):
            break
        m, e = np.load(f'{d}/manifest.npz', allow_pickle=True), np.load(f'{d}/emb.npz', allow_pickle=True)
        assert [base(p) for p in m['paths']] == [base(p) for p in e['paths']], d
        for p, v, c in zip(m['paths'], m['v'], e['emb_clip-vit-l-14']):
            out[base(p)] = (v, c)
    return out


def main(prefix):
    t0 = time.time()
    for ext in ('.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    ev = load_mod('controls_evaluate', EV_FILE)
    say(f'numpy {np.__version__}')
    unit, chord = ev.unit, ev.chord
    q = ev.qfun(O)
    P, T, Pz = ev.references(O)
    sv = {}
    pos = np.load(f'{O}/pos_emb.npz', allow_pickle=True)
    ids, qq = ev.startids(O), q
    loops = {}
    for arm, tg, w in ARMS:
        allq, r0, r59, st, aq, q0, q59 = [], [], [], [], [], [], []
        for n in STARTS:
            tag = f'qgc_{tg}_s{n}'
            z = np.load(f'{O}/runs/{tag}_committee_euclidean_min_seedrow/all_candidates.npz', allow_pickle=True)
            ex = np.load(f'{O}/eval/{tag}_emb.npz', allow_pickle=True)
            assert [str(a) for a in z['paths']] == [str(b) for b in ex['paths']], tag
            it = np.asarray(z['iteration'])
            m = it >= 0
            lv = q(ex['emb_clip-vit-l-14'])
            allq.append(lv[m]); r0.append(lv[it == 0]); r59.append(lv[it == 59])
            lq = q(z['style_embs'])
            aq.append(lq[m]); q0.append(lq[it == 0]); q59.append(lq[it == 59])
            if tg == 'ra':
                st.append(float(lv[0]))
        loops[arm] = dict(all=np.concatenate(allq), r0=np.concatenate(r0), r59=np.concatenate(r59), starts=st,
                          qall=np.concatenate(aq), q0=np.concatenate(q0), q59=np.concatenate(q59))
        assert len(loops[arm]['r0']) == 160 and len(loops[arm]['r59']) == 160 and len(loops[arm]['qall']) == 9600, arm
    rows = list(csv.DictReader(open(ROWS_CSV, encoding='utf-8')))
    idx = {}
    idx.update(shard_index(f'{O}/eval/regions/stage1'))
    idx.update(shard_index(f'{O}/eval/probe/stage1'))
    pm, pe = np.load(f'{O}/baselines/prior/manifest.npz', allow_pickle=True), np.load(f'{O}/baselines/prior/emb.npz', allow_pickle=True)
    ep = {base(p): c for p, c in zip(pe['paths'], pe['emb_clip-vit-l-14'])}
    for p, v in zip(pm['paths'], pm['v']):
        if base(p) in ep:
            idx[base(p)] = (v, ep[base(p)])
    nos, gapbad, missing = {}, 0, 0
    for key, name in KINDS:
        rr = [r for r in rows if r['group'].endswith(key) and not r['group'].startswith('round trips')]
        vs, cs = [], []
        for r in rr:
            if r['file'] not in idx:
                missing += 1
                continue
            v, c = idx[r['file']]
            if abs(float(chord(float(unit(v) @ unit(c)))) - float(r['gap'])) > 1e-5:
                gapbad += 1
            vs.append(v); cs.append(c)
        nos[name] = dict(n=len(rr), reading=q(np.asarray(cs)) if cs else np.zeros(0), request=q(np.asarray(vs)) if vs else np.zeros(0))
    assert missing == 0 and gapbad == 0, f'no-search pictures: {missing} without a reading, {gapbad} readings whose chord differs from the rows file'
    say()
    say('LEAN on the exact CLIP reading (qfun). Request leans in brackets where a request exists.')
    p2 = np.load(f'{O}/push2_emb.npz', allow_pickle=True)
    yr = T['year']
    Tz = np.load(f'{O}/push_today_emb.npz', allow_pickle=True)
    TL = q(Tz['emb_clip-vit-l-14'])
    sets = [('reference P, 2,048 repaints before 1900', q(Pz['emb_clip-vit-l-14']), q(Pz['vec'])),
            ('held out before 1900 (push2), 1,000', q(p2['emb_clip-vit-l-14']), q(p2['vec'])),
            ('repainted 1900-1969 (pos), 512', q(pos['emb_clip-vit-l-14']), q(pos['vec'])),
            ("today's reference T, before 1900", TL[yr < 1900], None),
            ("today's reference T, 1900-1969", TL[(yr >= 1900) & (yr < 1970)], None),
            ("today's reference T, 1970 on", TL[yr >= 1970], None)]
    assert int((yr >= 1970).sum()) == 663, int((yr >= 1970).sum())
    for si, (name, rd, rq) in enumerate(sets):
        say(f'   {name:42s} reading {stats(rd)}' + ('' if rq is None else f' || request median {np.median(rq):+.4f}'))
        sv[f'set{si}__reading'] = rd
    say(f'   the 5 start repaints (row 0 of qgc_ra_s6..10, exact reading): ' + ', '.join(f's{n} {v:+.4f}' for n, v in zip(STARTS, loops['V']['starts'])))
    em = np.load(f'{NOISE}/emb.npz', allow_pickle=True)
    k = len([ln for ln in open(f'{NOISE}/inputs/expected_kat.txt') if ln.strip()])
    nz = q(em['emb_clip-vit-l-14'][k:])
    assert len(nz) == 160, len(nz)
    say(f'   {"the 160 noise-request pictures":42s} reading {stats(nz)}')
    say('   the 1,270 no-search pictures, by kind:')
    allr, allq = [], []
    for _, name in KINDS:
        say(f'      {name:40s} reading {stats(nos[name]["reading"])} || request median {np.median(nos[name]["request"]):+.4f}')
        allr.append(nos[name]['reading']); allq.append(nos[name]['request'])
    say(f'      {"pooled 1,270":40s} reading {stats(np.concatenate(allr))} || request median {np.median(np.concatenate(allq)):+.4f}')
    say()
    say('LOOPS: all 9,600; round 0 against round 59 (32 pictures x 5 starts = 160 each)')
    for arm, tg, w in ARMS:
        L = loops[arm]
        say(f'   {arm} {w:18s} all {stats(L["all"])} | round 0 {stats(L["r0"])} | round 59 {stats(L["r59"])}')
        say(f'   {arm} {"  its requests":18s} all {stats(L["qall"])} | round 0 {stats(L["q0"])} | round 59 {stats(L["q59"])}')
        for k2 in ('all', 'r0', 'r59', 'qall', 'q0', 'q59'):
            sv[f'{arm}__{k2}'] = L[k2]
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'saved {prefix}.npz; done {time.time() - t0:.0f} s')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
