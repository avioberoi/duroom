"""For every kind of picture on disk, its far score and rank against P (push_emb.npz) and its gap, with counts beyond the 95th / 99th far levels and the FID / T99 gap bars.
Usage (CPU): python -B grid_v1.py OUTPREFIX -> OUTPREFIX.txt, OUTPREFIX.npz, OUTPREFIX_rows.csv (refused if any exists)"""
import csv
import importlib.util
import io
import os
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
R2 = f'{B}/tests12/t2/renders_v1'
EV_FILE = f'{C}/controls_evaluate.py'
ARMS_NPZ = f'{B}/tests12/arms/arms_barfree_v1.npz'
CELLS = {arm: [f'qgc_{tg}_s{n}' for n in (6, 7, 8, 9, 10)] for arm, tg in (('V', 'ra'), ('C1', 'pin'), ('C2', 'c2'))}
FREE = ['main_s1', 'main_s2', 'main_s3', 'rf_s1', 'rf_s2', 'rf_s3', 'b0_s1', 'b0_s2', 'b0_s3', 'main_fid_s1']
QS = (10, 50, 90)
SEED, NSAMPLE = 20261115, 2000
pid = lambda paths: np.array([int(re.search(r'push_(\d+)\.png$', str(p)).group(1)) for p in paths])
OUT = []


def say(s=''):
    OUT.append(s)


def load_ev():
    spec = importlib.util.spec_from_file_location('controls_evaluate', EV_FILE)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def main(prefix):
    for ext in ('.txt', '.npz', '_rows.csv'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    ev = load_ev()
    OBS, K, unit, chord, FID = ev.OBS, ev.K, ev.unit, ev.chord, float(ev.FID)
    P, T, Pz = ev.references(O)                       # checks P's bars and FID
    Pid, Part = P['id'], P['artist']
    own = {o: P[o]['own'] for o in OBS}
    lev = {o: {q: float(np.percentile(own[o], q)) for q in (95, 99)} for o in OBS}
    Tz = np.load(f'{O}/push_today_emb.npz', allow_pickle=True)
    T99 = float(np.percentile(chord((unit(Tz['vec']) * unit(Tz['emb_clip-vit-l-14'])).sum(1)), 99))

    def far_scores(emb, masks=None):
        """emb: {observer: (n, dim)}; masks: per picture an array of P row indices to leave out (None = nothing). Mean chord to the K nearest P rows."""
        d = {}
        for o in OBS:
            D = chord(unit(emb[o]) @ P[o]['E'].T)
            if masks is not None:
                for i, ks in enumerate(masks):
                    if len(ks):
                        D[i, ks] = np.inf
            d[o] = np.sort(D, axis=1)[:, :K].mean(1)
        return d

    def rank(v, o):
        return 100.0 * np.searchsorted(np.sort(own[o]), v, 'left') / len(own[o])

    G = {}   # group name -> dict(d_dino, d_siglip, gap, [hit])

    def add(name, d, gap, **extra):
        rd, rs = rank(d[OBS[0]], OBS[0]), rank(d[OBS[1]], OBS[1])
        G[name] = dict(d_dino=np.asarray(d[OBS[0]], float), d_siglip=np.asarray(d[OBS[1]], float), rank_dino=rd, rank_siglip=rs,
                       minrank=np.minimum(rd, rs), gap=np.asarray(gap, float), **extra)

    # check: the score code reproduces P's own score when each P row and its artist's rows are left out (leave-one-out, artist out)
    selfmask = [np.flatnonzero(Part == a) for a in Part]
    dP = far_scores({o: Pz[f'emb_{o}'] for o in OBS}, selfmask)
    for o in OBS:
        assert np.allclose(dP[o], own[o], rtol=0, atol=1e-12), o
    add('0 P reference (own score, leave-one-out, artist out)', dP, chord((unit(Pz['vec']) * unit(Pz['emb_clip-vit-l-14'])).sum(1)))
    say('check: the far-score code with each P row and its artist\'s P rows left out reproduces P\'s own scores (1e-12)')

    pools = {n: np.load(f'{O}/pools/{n}.npz', allow_pickle=True) for n in ('pre1900_cap3', 'pos_1900_1969_cap3', 'today_cap3')}
    yr = {n: dict(zip(p['ids'].astype(int).tolist(), p['year'].astype(float).tolist())) for n, p in pools.items()}
    row_in_P = {int(i): k for k, i in enumerate(Pid.tolist())}
    for name, f, pool in (('P2', 'push2_emb.npz', 'pre1900_cap3'), ('pos', 'pos_emb.npz', 'pos_1900_1969_cap3'), ('today', 'push_today_emb.npz', 'today_cap3')):
        z = np.load(f'{O}/{f}', allow_pickle=True)
        ids, art = pid(z['paths']), np.array([str(a) for a in z['artist']])
        year = np.array([yr[pool][int(i)] for i in ids])
        inP = np.array([int(i) in row_in_P for i in ids])
        for i in np.flatnonzero(inP):
            assert Part[row_in_P[int(ids[i])]] == art[i], (name, ids[i])
        src_masks = [np.flatnonzero(Part == art[i]) if inP[i] else np.zeros(0, int) for i in range(len(ids))]   # the P row itself is among its artist's rows
        art_masks = [np.flatnonzero(Part == a) for a in art]
        emb = {o: z[f'emb_{o}'] for o in OBS}
        d1, d2 = far_scores(emb, src_masks), far_scores(emb, art_masks)
        gap = chord((unit(z['vec']) * unit(z['emb_clip-vit-l-14'])).sum(1))
        say(f'{name} ({f}): {len(ids)} rows, {int(inP.sum())} whose source work is a row of P (that row and its artist\'s P rows left out); '
            f'{int(sum(len(m) > 0 for m in art_masks))} whose artist has rows in P')
        sel = {'P2': [('1 P2 (before 1900)', np.ones(len(ids), bool))], 'pos': [('1 pos (1900-1969)', np.ones(len(ids), bool))],
               'today': [('1 today, before 1900', year < 1900), ('1 today, 1900-1969', (year >= 1900) & (year < 1970)), ('1 today, 1970 on', year >= 1970),
                         ('1 today, all dates', np.ones(len(ids), bool))]}[name]
        for lab, s in sel:
            add(lab, {o: d1[o][s] for o in OBS}, gap[s], ids=ids[s])
            add(lab.replace('1 ', '1b ', 1) + ' [variant: source artist\'s P rows left out for every row]', {o: d2[o][s] for o in OBS}, gap[s], ids=ids[s])

    # the first rows of renders_v1/prior256_v1 are known-answer pictures, dropped
    em = np.load(f'{R2}/prior256_v1/emb.npz', allow_pickle=True)
    kat = [ln.strip() for ln in open(f'{R2}/prior256_v1/inputs/expected_kat.txt') if ln.strip()]
    assert [str(p) for p in em['paths'][:len(kat)]] == kat
    man = np.load(f'{R2}/prior256_v1/prior/manifest.npz', allow_pickle=True)
    k = len(kat)
    assert [str(p) for p in em['paths'][k:]] == [str(p) for p in man['paths']]
    kind = np.array([str(x) for x in man['kind']])
    dR = far_scores({o: em[f'emb_{o}'][k:] for o in OBS})
    gR = chord((unit(man['v']) * unit(em['emb_clip-vit-l-14'][k:])).sum(1))
    for kd, n in (('cone', 192), ('sphere', 64)):
        s = kind == kd
        assert s.sum() == n
        add(f'2 random requests, {kd}', {o: dR[o][s] for o in OBS}, gR[s])
    say(f'random requests: {R2}/prior256_v1/emb.npz; {k} known-answer rows dropped')

    # noise requests: the request is the noise image's CLIP reading
    em4 = np.load(f'{R2}/noise160_v1/emb.npz', allow_pickle=True)
    kat4 = [ln.strip() for ln in open(f'{R2}/noise160_v1/inputs/expected_kat.txt') if ln.strip()]
    assert [str(p) for p in em4['paths'][:len(kat4)]] == kat4
    man4 = np.load(f'{R2}/noise160_v1/noise/manifest.npz', allow_pickle=True)
    k4 = len(kat4)
    assert [str(p) for p in em4['paths'][k4:]] == [str(p) for p in man4['paths']]
    nk = np.array([str(t) for t in man4['tag']])
    dN = far_scores({o: em4[f'emb_{o}'][k4:] for o in OBS})
    gN = chord((unit(man4['v']) * unit(em4['emb_clip-vit-l-14'][k4:])).sum(1))
    assert len(nk) == 160
    add('3 noise requests, all kinds', dN, gN, kind=nk)
    for kd in 'abcde':
        s = nk == kd
        assert s.sum() == 32, (kd, int(s.sum()))
        add(f'3 noise requests, kind {kd}', {o: dN[o][s] for o in OBS}, gN[s])
    say(f'noise requests: {R2}/noise160_v1/emb.npz; {k4} known-answer rows dropped; '
        'kinds a uniform pixel noise, b gaussian pixel noise, c solid colour, d smooth low-frequency, e 8x8 colour blocks')

    # the 15 qgc cells: far from the run file's DINOv3 / SigLIP (the evaluator's F), gap from the exact CLIP file; hits = DH from the arms npz
    A = np.load(ARMS_NPZ)
    for arm, cells in CELLS.items():
        parts = []
        for cell in cells:
            z = np.load(f'{O}/runs/{cell}_committee_euclidean_min_seedrow/all_candidates.npz', allow_pickle=True)
            ex = np.load(f'{O}/eval/{cell}_emb.npz', allow_pickle=True)
            m = z['iteration'] >= 0
            assert m.sum() == 1920 and all(str(a) == str(b) for a, b in zip(z['paths'], ex['paths']))
            d = far_scores({o: z[f'emb_{o}'][m] for o in OBS})
            gap = chord((unit(z['style_embs']) * unit(ex['emb_clip-vit-l-14'])).sum(1))[m]
            assert np.allclose(d[OBS[0]], A[f'{cell}__d_dino'], rtol=0, atol=1e-12) and np.allclose(d[OBS[1]], A[f'{cell}__d_siglip'], rtol=0, atol=1e-12)
            assert np.allclose(gap, A[f'{cell}__gap'], rtol=0, atol=1e-12)
            parts.append((d, gap, A[f'{cell}__DH'].astype(bool)))
        d = {o: np.concatenate([p[0][o] for p in parts]) for o in OBS}
        gap, hit = np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
        add(f'4 {arm}, all candidates (5 starts)', d, gap, hit=hit)
        add(f'4 {arm}, hits DH', {o: d[o][hit] for o in OBS}, gap[hit])
    say(f'qgc cells: far and gap recomputed and asserted equal (1e-12) to the arms file ({ARMS_NPZ}); hits = its DH arrays')

    # free runs: every reading from the exact file eval/<tag>_emb.npz
    for tag in FREE:
        dd = f'{O}/runs/{tag}_committee_euclidean_min_seedrow'
        f = f'{dd}/all_candidates.npz' if os.path.exists(f'{dd}/all_candidates.npz') else f'{dd}/all_draws.npz'
        exf = f'{O}/eval/{tag}_emb.npz'
        if not (os.path.exists(f) and os.path.exists(exf)):
            say(f'5 {tag}: skipped, no run file or exact file')
            continue
        z, ex = np.load(f, allow_pickle=True), np.load(exf, allow_pickle=True)
        m = z['iteration'] >= 0
        same = len(z['paths']) == len(ex['paths']) and all(str(a) == str(b) for a, b in zip(z['paths'], ex['paths']))
        if not same or m.sum() != 1920 or int(z['iteration'].max()) != 59:
            say(f'5 {tag}: LEFT OUT (rows after start {int(m.sum())}, last round {int(z["iteration"].max())}, paths equal {same})')
            continue
        d = far_scores({o: ex[f'emb_{o}'][m] for o in OBS})
        gap = chord((unit(z['style_embs']) * unit(ex['emb_clip-vit-l-14'])).sum(1))[m]
        cs = {o: (unit(z[f'emb_{o}'][m]) * unit(ex[f'emb_{o}'][m])).sum(1) for o in OBS}
        add(f'5 free run {tag}', d, gap)
        say(f'5 {tag}: run file {os.path.basename(f)} (requests style_embs), exact file {exf}; 1,920 rows; cosine of the run file\'s own reading with the exact one, '
            + ', '.join(f'{o} median {np.median(cs[o]):.4f} min {cs[o].min():.4f}' for o in OBS))
    for kind_, tags in (('main', FREE[0:3]), ('rf', FREE[3:6]), ('b0', FREE[6:9])):
        names = [f'5 free run {t}' for t in tags if f'5 free run {t}' in G]
        if len(names) == 3:
            G[f'5 free runs {kind_}, pooled (3 seeds)'] = {kk: np.concatenate([G[nm][kk] for nm in names]) for kk in ('d_dino', 'd_siglip', 'rank_dino', 'rank_siglip', 'minrank', 'gap')}

    head = ['Reference P = push_emb.npz (2,048 pre-1900 repaints); far score = mean chord to the 15 nearest P pictures per observer (DINOv3, SigLIP); rank = 100 x share of P\'s own '
            'leave-one-out (artist out) scores strictly below; minrank = the smaller of the two observers\' ranks. Gap = chord(request, exact CLIP reading).',
            f'Levels of P\'s own score: ' + '; '.join(f'{o} p95 {lev[o][95]:.4f}, p99 {lev[o][99]:.4f}' for o in OBS) + f'. Gap bars: FID {FID:.6f}, T99 {T99:.6f}.',
            'Four counts per bar pair: FF = far and faithful, FN = far and not faithful, NF = near (not far) and faithful, NN = near and not faithful; far = beyond the level in BOTH observers.',
            '']
    say()
    say(f'{"group":92s} {"n":>5s} | minrank p10 p50 p90 | gap p10 p50 p90 | far99,FID FF FN NF NN | far95,FID FF FN NF NN | far99,T99 FF FN NF NN | far95,T99 FF FN NF NN')
    for name, g in G.items():
        n = len(g['gap'])
        rq, gq = np.percentile(g['minrank'], QS), np.percentile(g['gap'], QS)
        s = f'{name:92s} {n:5d} | {rq[0]:5.1f} {rq[1]:5.1f} {rq[2]:5.1f} | {gq[0]:.4f} {gq[1]:.4f} {gq[2]:.4f}'
        cnt = {}
        for gb, gv in (('FID', FID), ('T99', T99)):
            for fq in (99, 95):
                far = (g['d_dino'] > lev[OBS[0]][fq]) & (g['d_siglip'] > lev[OBS[1]][fq])
                fa = g['gap'] <= gv
                c4 = [int((far & fa).sum()), int((far & ~fa).sum()), int((~far & fa).sum()), int((~far & ~fa).sum())]
                assert sum(c4) == n
                cnt[(fq, gb)] = c4
        for key in ((99, 'FID'), (95, 'FID'), (99, 'T99'), (95, 'T99')):
            s += ' | ' + ' '.join(f'{v:5d}' for v in cnt[key])
        say(s)
    OUT[:0] = head

    # CSV: every row of groups up to 2,000; a seeded sample of 2,000 from larger ones; the today 'all dates' group and pooled free-run groups are not leaves and are left out
    rng = np.random.default_rng(SEED)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['group', 'row', 'd_dino', 'd_siglip', 'rank_dino', 'rank_siglip', 'minrank', 'gap', 'hit', 'sampled'])
    for name, g in G.items():
        if name.startswith('1b ') or 'all dates' in name or 'pooled' in name or name == '3 noise requests, all kinds':
            continue
        n = len(g['gap'])
        idx = np.arange(n) if n <= NSAMPLE else np.sort(rng.choice(n, NSAMPLE, replace=False))
        for i in idx:
            w.writerow([name, int(i)] + [f'{g[k][i]:.6f}' for k in ('d_dino', 'd_siglip', 'rank_dino', 'rank_siglip', 'minrank', 'gap')]
                       + [int(g['hit'][i]) if 'hit' in g else '', int(n > NSAMPLE)])
    arrs = {'P_own__dinov3-l': own[OBS[0]], 'P_own__siglip-so400m': own[OBS[1]], 'FID': np.array(FID), 'T99': np.array(T99), 'group_names': np.array(list(G))}
    for j, (name, g) in enumerate(G.items()):
        for kk, v in g.items():
            arrs[f'g{j:02d}__{kk}'] = np.asarray(v)
    with open(prefix + '.npz', 'xb') as fh:
        np.savez_compressed(fh, **arrs)
    open(prefix + '_rows.csv', 'x').write(buf.getvalue())
    open(prefix + '.txt', 'x').write('\n'.join(OUT) + '\n')
    print('\n'.join(OUT), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
