"""The three arms (search V, random-score loops C1, C2; starts 6..10) compared without bars on every saved candidate: far scores, ranks, pair numbers, gaps.
Usage (CPU): python -B arms_barfree_v1.py OUTPREFIX -> OUTPREFIX.txt, OUTPREFIX.npz, OUTPREFIX_rows.csv (each refused if it exists)."""
import csv
import importlib.util
import io
import json
import os
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C, O, L = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere', f'{B}/logs'
BD = f'{O}/eval/bound'
EV_FILE = f'{C}/controls_evaluate.py'
TABLE = f'{BD}/conf_table.out'
CACHE = f'{BD}/ev_safety_conf.json'
# the 15 cells (no glob: nothing else is read)
CELLS = ('qgc_ra_s6', 'qgc_ra_s7', 'qgc_ra_s8', 'qgc_ra_s9', 'qgc_ra_s10', 'qgc_pin_s6', 'qgc_pin_s7', 'qgc_pin_s8', 'qgc_pin_s9', 'qgc_pin_s10',
         'qgc_c2_s6', 'qgc_c2_s7', 'qgc_c2_s8', 'qgc_c2_s9', 'qgc_c2_s10')
ARMS = (('V', 'ra'), ('C1', 'pin'), ('C2', 'c2'))
STARTS = (6, 7, 8, 9, 10)
QS = (10, 25, 50, 75, 90, 99)
LEVELS = (50, 75, 90, 95, 99, 99.9)
OUT = []


def say(s=''):
    OUT.append(s)


def load_ev():
    spec = importlib.util.spec_from_file_location('controls_evaluate', EV_FILE)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def table_anchors():
    """per cell, from the RUN blocks of the evaluator table (conf_table.out): the EXACT row, its DH row, and the LOOP row's n_faithful"""
    out, cur = {}, None
    for ln in open(TABLE):
        m = re.match(r'RUN (\S+):', ln)
        if m:
            cur = m.group(1)
            out[cur] = {}
            continue
        if ln.startswith('PAIR'):
            cur = None
        if cur is None:
            continue
        m = re.match(r'\s+EXACT n_far (\d+) n_faithful (\d+) n_H (\d+) HR \S+ \| evaluator-admitted (\d+) DF (\d+) DH (\d+) .*S flagged among the \d+ evaluator-admitted pictures: (\d+)', ln)
        if m:
            out[cur].update(zip(('n_far', 'n_faithful', 'n_H', 'ev', 'DF', 'DH', 'ev_flag'), map(int, m.groups())))
        m = re.match(r'\s+DH: DH (\d+), pass S (\d+) \(flagged \d+\), inside .*?both\) (\d+) \[DINOv3 (\d+), SigLIP (\d+);.*quality hits QH (\d+),', ln)
        if m and 'U' not in out[cur]:
            out[cur].update(zip(('DH_again', 'pass_S', 'U', 'U_dino', 'U_siglip', 'QH'), map(int, m.groups())))
        m = re.match(r'\s+LOOP \(quickgelu\) n_far \d+ n_faithful (\d+)', ln)
        if m:
            out[cur]['loop_n_faithful'] = int(m.group(1))
    assert sorted(out) == sorted(CELLS) and all(len(v) == 14 for v in out.values()), {k: len(v) for k, v in out.items()}
    return out


def own_scores(E, artist, K, unit, chord):
    """reference's own score: leave-one-out, same-artist rows left out (written as today_rescore.reference does it, for any reader)"""
    E = unit(E)
    D = chord(E @ E.T)
    art = np.array([str(a) for a in artist])
    D[art[:, None] == art[None, :]] = np.inf
    return np.sort(D, axis=1)[:, :K].mean(1)


def cand_scores(X, E, K, unit, chord):
    """candidate's score: mean chord to its K nearest reference rows, nothing left out (as today_rescore.score with artist=None)"""
    return np.sort(chord(unit(X) @ unit(E).T), axis=1)[:, :K].mean(1)


def rank(v, own):
    return 100.0 * np.searchsorted(np.sort(own), v, 'left') / len(own)


def pcts(v, fmt='.4f'):
    return ' | '.join(f'p{q:g} {x:{fmt}}' for q, x in zip(QS, np.percentile(v, QS)))


def beyond(vals, levs):
    """vals: list of score arrays (one per reader), levs: list of {q: level}; count beyond each level in ALL the given readers"""
    return {q: int(np.all([v > lv[q] for v, lv in zip(vals, levs)], axis=0).sum()) for q in LEVELS}


def bline(cnt, n):
    return ' | '.join(f'>p{q:g} {cnt[q]:5d} ({cnt[q] / n:.4f})' for q in LEVELS)


def main(prefix):
    for ext in ('.txt', '.npz', '_rows.csv', '_STOP.txt'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    ev = load_ev()
    from local_rescore import auc
    assert auc(np.array([2.0]), np.array([1.0])) == 1.0 and auc(np.array([1.0]), np.array([2.0])) == 0.0 and auc(np.array([1.0]), np.array([1.0])) == 0.5
    OBS, K, unit, chord, FID = ev.OBS, ev.K, ev.unit, ev.chord, float(ev.FID)
    anchors = table_anchors()
    cache = json.load(open(CACHE))
    P, T, Pz = ev.references(O)
    ids, q = ev.startids(O), ev.qfun(O)
    sup = ev.load_support()
    Tz = np.load(f'{O}/push_today_emb.npz', allow_pickle=True)
    t99 = float(np.percentile(chord((unit(Tz['vec']) * unit(Tz['emb_clip-vit-l-14'])).sum(1)), 99))

    # extra readers, references from P's repaint readings (push_emb.npz); checks that the generic code reproduces the evaluator's own scores
    own = {o: P[o]['own'] for o in OBS}
    for o in OBS:
        assert np.allclose(own_scores(Pz[f'emb_{o}'], Pz['artist'], K, unit, chord), own[o], rtol=0, atol=1e-12), o
    EXTRA = {'aimv2-huge (reading of the picture)': ('emb_aimv2-huge', 'emb_aimv2-huge', 'exact'),
             'clip-vit-l-14 (exact reading of the picture)': ('emb_clip-vit-l-14', 'emb_clip-vit-l-14', 'exact'),
             'clip-vit-l-14 (request vector, labelled extra)': ('vec', 'style_embs', 'run')}
    own_x = {k: own_scores(Pz[pk], Pz['artist'], K, unit, chord) for k, (pk, _, _) in EXTRA.items()}
    lev = {o: {qq: float(np.percentile(own[o], qq)) for qq in LEVELS} for o in OBS}
    lev_x = {k: {qq: float(np.percentile(v, qq)) for qq in LEVELS} for k, v in own_x.items()}

    cells, bad = {}, []
    for arm, tg in ARMS:
        for n in STARTS:
            cell = f'qgc_{tg}_s{n}'
            spec = f'{cell}={O}/runs/{cell}_committee_euclidean_min_seedrow,{O}/eval/{cell}_emb.npz,{L}/{cell}.out'
            r = ev.load(spec, P, T, ids, q, conf=True)         # raises on any refusal of strict mode
            s = ev.quality(r, ev.stats(r, 'exact'), sup, cache, None)
            sl = ev.stats(r, 'loop')
            got = dict(n_far=s['n_far'], n_faithful=s['n_faithful'], n_H=s['n_H'], ev=s['ev'], DF=s['nDF'], DH=s['nDH'], ev_flag=s['ev_flag'],
                       DH_again=s['nDH'], pass_S=s['nS'], U=s['nU'], U_dino=int((s['d'][0] <= s['base'][0]).sum()), U_siglip=int((s['d'][1] <= s['base'][1]).sum()),
                       QH=s['nQH'], loop_n_faithful=sl['n_faithful'])
            ok = got == anchors[cell]
            say(f'anchor {cell}: ' + ', '.join(f'{k} {v}' for k, v in got.items()) + (' | evaluator table: same' if ok else f' | evaluator table DIFFERENT: {anchors[cell]}'))
            if not ok:
                bad.append(cell)
            z = r['z']
            m = z['iteration'] >= 0
            ex = np.load(f'{O}/eval/{cell}_emb.npz', allow_pickle=True)
            assert all(str(a_) == str(b_) for a_, b_ in zip(z['paths'], ex['paths']))
            Pr = ev.without(P, r['wid'])
            assert Pr is P                                    # the start work is not a row of P
            sP = ev.score(Pr, {f'emb_{o}': z[f'emb_{o}'][m] for o in OBS})
            assert (ev.far(sP, Pr) == r['F']).all()
            d = {o: sP[o][0] for o in OBS}
            for o in OBS:
                assert np.allclose(cand_scores(z[f'emb_{o}'][m], Pz[f'emb_{o}'], K, unit, chord), d[o], rtol=0, atol=1e-12)
            rk = {o: rank(d[o], own[o]) for o in OBS}
            gap = chord((unit(z['style_embs']) * unit(ex['emb_clip-vit-l-14'])).sum(1))[m]
            assert ((gap <= FID) == r['ok_exact']).all()
            src = {'exact': ex, 'run': z}
            dx = {k: cand_scores(src[w][ck][m], Pz[pk], K, unit, chord) for k, (pk, ck, w) in EXTRA.items()}
            cells[cell] = dict(arm=arm, start=n, it=z['iteration'][m].astype(int), d_dino=d[OBS[0]], d_siglip=d[OBS[1]], rank_dino=rk[OBS[0]], rank_siglip=rk[OBS[1]],
                               minrank=np.minimum(rk[OBS[0]], rk[OBS[1]]), gap=gap, infeasible=~z['feasible'][m].astype(bool), F=r['F'], EV=r['ev'], DF=s['DF'], DH=s['DH'],
                               nU=s['nU'], nQH=s['nQH'], dx=dx)
            assert beyond([d[o] for o in OBS], [lev[o] for o in OBS])[99] == s['n_far']
            assert int(cells[cell]['infeasible'].sum()) == 1920 - sl['n_faithful']
    if bad:
        open(prefix + '_STOP.txt', 'x').write('\n'.join(OUT) + f'\nSTOP: anchors differ in {bad}; nothing else computed\n')
        raise SystemExit(f'STOP: anchors differ in {bad}')
    say('anchors: all 15 cells reproduce the evaluator table (EXACT row, its DH row, LOOP n_faithful); new numbers follow')

    head = [f'arms_barfree_v1.py; evaluator {EV_FILE}; evaluator table {TABLE}; safety cache {CACHE} (read only)',
            'Runs: V = qgc_ra_s6..10, C1 = qgc_pin_s6..10, C2 = qgc_c2_s6..10; 1,920 candidates per cell (start row dropped); pooled = the five starts of an arm (9,600).',
            f'Reference P = push_emb.npz, 2,048 pre-1900 repaints; P\'s own score = leave-one-out, artist rows left out; candidates scored against all of P (start work not in P). K = {K}.',
            'Levels of P (np.percentile of P\'s own score): ' + '; '.join(f'{o}: ' + ', '.join(f'p{qq:g} {lev[o][qq]:.4f}' for qq in LEVELS) for o in OBS),
            '']
    OUT[:0] = head
    rows = []
    arms = [a for a, _ in ARMS]

    def group(arm):
        return [cells[f'qgc_{dict(ARMS)[arm]}_s{n}'] for n in STARTS]

    def cat(arm, key):
        return np.concatenate([c[key] for c in group(arm)])

    for key, title, fmt in (('d_dino', 'far score DINOv3 (mean chord to the 15 nearest P pictures)', '.4f'),
                            ('d_siglip', 'far score SigLIP (mean chord to the 15 nearest P pictures)', '.4f'),
                            ('rank_dino', 'rank in P\'s own DINOv3 distribution (100 x share of P works with a smaller own score)', '.1f'),
                            ('rank_siglip', 'rank in P\'s own SigLIP distribution', '.1f'),
                            ('minrank', 'smaller rank of the two observers', '.1f')):
        say(f'TABLE 1 {title}; reader {"DINOv3" if "dino" in key else "SigLIP" if "siglip" in key else "DINOv3 and SigLIP"}; reference P; all candidates')
        for arm in arms:
            for c in group(arm):
                say(f'   {arm:2s} s{c["start"]:<2d} n {len(c[key]):5d} | {pcts(c[key], fmt)}')
            say(f'   {arm:2s} pool n {len(cat(arm, key)):5d} | {pcts(cat(arm, key), fmt)}')
        say()
    say('TABLE 1e candidates beyond each level of P\'s own score in BOTH DINOv3 and SigLIP (count, share); reference P; the p99 column = n_far of the evaluator table')
    for arm in arms:
        for c in group(arm):
            say(f'   {arm:2s} s{c["start"]:<2d} n 1920 | {bline(beyond([c["d_dino"], c["d_siglip"]], [lev[o] for o in OBS]), 1920)}')
        say(f'   {arm:2s} pool n 9600 | {bline(beyond([cat(arm, "d_dino"), cat(arm, "d_siglip")], [lev[o] for o in OBS]), 9600)}')
    say()

    aucs = {}
    say('TABLE 2 pair number per start: P(a random V candidate scores above a random control candidate of the same start), ties half; 0.5 = no difference; reference P')
    for key, lab in (('d_dino', 'far score DINOv3'), ('d_siglip', 'far score SigLIP'), ('minrank', 'smaller rank')):
        for x in ('C1', 'C2'):
            v = np.array([auc(cells[f'qgc_ra_s{n}'][key], cells[f'qgc_{dict(ARMS)[x]}_s{n}'][key]) for n in STARTS])
            aucs[f'{key}__V_vs_{x}'] = v
            say(f'   {lab:17s} V vs {x}: ' + ', '.join(f's{n} {a_:.4f}' for n, a_ in zip(STARTS, v)) + f' | mean {v.mean():.4f}')
    say()

    keys3 = ('d_dino', 'd_siglip', 'minrank')
    series = np.full((3, 5, 60, 3), np.nan)
    for i, arm in enumerate(arms):
        for j, c in enumerate(group(arm)):
            for rr in range(60):
                k_ = c['it'] == rr
                assert k_.sum() == 32
                series[i, j, rr] = [np.median(c[kk][k_]) for kk in keys3]
    mean_series = series.mean(1)
    say('TABLE 3 median per round (32 candidates) of each score, averaged over the five starts; reference P. Columns: V, C1, C2 for DINOv3 far score | SigLIP far score | smaller rank')
    for rr in range(60):
        say(f'   round {rr:2d} | ' + ' | '.join(' '.join(f'{mean_series[i, rr, k]:{".4f" if k < 2 else "6.1f"}}' for i in range(3)) for k in range(3)))
    say()

    say(f'TABLE 4 request-to-reading gap = chord(request, exact CLIP reading) (the evaluator\'s faithful score); reader CLIP (exact file); '
        f'counted above FID {FID:.6f} (= 1920 - n_faithful) and above {t99:.6f} (the push_today_emb.npz reference\'s own 99th percentile of the same chord); infeasible = the loop\'s own flag (fidelity_clip above its gate 0.7788)')
    for arm in arms:
        for c in group(arm):
            g = c['gap']
            say(f'   {arm:2s} s{c["start"]:<2d} n 1920 | {pcts(g)} | above {FID:.4f}: {int((g > FID).sum()):4d} | above {t99:.4f}: {int((g > t99).sum()):4d} | infeasible (loop) {int(c["infeasible"].sum()):4d}')
        g = cat(arm, 'gap')
        say(f'   {arm:2s} pool n 9600 | {pcts(g)} | above {FID:.4f}: {int((g > FID).sum()):4d} | above {t99:.4f}: {int((g > t99).sum()):4d} | infeasible (loop) {int(cat(arm, "infeasible").sum()):4d}')
    say()

    say('TABLE 5 candidates left after each condition, in the evaluator\'s order: all -> far in both observers at P\'s 99th percentile (F) -> evaluator-admitted (replay of the loop rule) '
        '-> faithful (DF) -> covered (DH) -> near a real work (U) -> passes the safety filter (QH); DH and QH = the evaluator table')
    for arm in arms:
        tot = np.zeros(7, int)
        for c in group(arm):
            f = np.array([1920, c['F'].sum(), c['EV'].sum(), c['DF'].sum(), c['DH'].sum(), c['nU'], c['nQH']], int)
            tot += f
            say(f'   {arm:2s} s{c["start"]:<2d} ' + ' -> '.join(f'{v:4d}' for v in f))
        say(f'   {arm:2s} pool ' + ' -> '.join(f'{v:4d}' for v in tot))
    say()

    for k in EXTRA:
        what = ('P\'s 2,048 request vectors (push_emb.npz vec) against the candidates\' requests (run file style_embs)' if 'request' in k else
                f'P\'s 2,048 repaint readings (push_emb.npz {EXTRA[k][0]}) against the candidates\' readings (exact file {EXTRA[k][1]})')
        say(f'TABLE 6 reader {k}: reference {what}; same score (mean chord to 15 nearest, P\'s own leave-one-out artist out); share beyond each level of P\'s own score in this reader alone')
        say('   levels: ' + ', '.join(f'p{qq:g} {lev_x[k][qq]:.4f}' for qq in LEVELS))
        for arm in arms:
            for c in group(arm):
                say(f'   {arm:2s} s{c["start"]:<2d} n 1920 | {bline(beyond([c["dx"][k]], [lev_x[k]]), 1920)}')
            say(f'   {arm:2s} pool n 9600 | {bline(beyond([np.concatenate([c["dx"][k] for c in group(arm)])], [lev_x[k]]), 9600)}')
        say()

    for arm in arms:
        for c in group(arm) + [None]:
            g = (lambda kk: c[kk]) if c is not None else (lambda kk: cat(arm, kk))
            n = 1920 if c is not None else 9600
            row = {'arm': arm, 'start': f's{c["start"]}' if c is not None else 'pool', 'n': n}
            for kk in ('d_dino', 'd_siglip', 'rank_dino', 'rank_siglip', 'minrank', 'gap'):
                row.update({f'{kk}_p{qq:g}': round(float(x), 6) for qq, x in zip(QS, np.percentile(g(kk), QS))})
            b2 = beyond([g('d_dino'), g('d_siglip')], [lev[o] for o in OBS])
            row.update({f'both_beyond_p{qq:g}': b2[qq] for qq in LEVELS})
            for k in EXTRA:
                vx = c['dx'][k] if c is not None else np.concatenate([cc['dx'][k] for cc in group(arm)])
                bx = beyond([vx], [lev_x[k]])
                row.update({f'{k.split(" ")[0]}_{"request" if "request" in k else "reading"}_beyond_p{qq:g}': bx[qq] for qq in LEVELS})
            row.update(gap_above_FID=int((g('gap') > FID).sum()), gap_above_T99=int((g('gap') > t99).sum()), infeasible_loop=int(g('infeasible').sum()),
                       F=int(g('F').sum()), evaluator_admitted=int(g('EV').sum()), DF=int(g('DF').sum()), DH=int(g('DH').sum()))
            row.update(U=c['nU'] if c is not None else sum(cc['nU'] for cc in group(arm)), QH=c['nQH'] if c is not None else sum(cc['nQH'] for cc in group(arm)))
            if c is not None and arm == 'V':
                for kk in keys3:
                    for x in ('C1', 'C2'):
                        row[f'auc_{kk}_V_vs_{x}'] = round(float(aucs[f'{kk}__V_vs_{x}'][STARTS.index(c['start'])]), 6)
            rows.append(row)
    buf = io.StringIO()
    cols = list(dict.fromkeys(k for r_ in rows for k in r_))
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

    arrs = {'series_median_per_round': series, 'series_axes': np.array(['arm V,C1,C2', 'start s6..s10', 'round 0..59', 'score d_dino,d_siglip,minrank']),
            't99': np.array(t99), 'FID': np.array(FID)}
    for k_, v in aucs.items():
        arrs[f'auc__{k_}'] = v
    for o in OBS:
        arrs[f'P_own__{o}'] = own[o]
    for k in EXTRA:
        arrs[f'P_own__{k.split(" ")[0]}_{"request" if "request" in k else "reading"}'] = own_x[k]
    for cell, c in cells.items():
        for kk in ('it', 'd_dino', 'd_siglip', 'rank_dino', 'rank_siglip', 'minrank', 'gap', 'infeasible', 'F', 'EV', 'DF', 'DH'):
            arrs[f'{cell}__{kk}'] = c[kk]
        for k in EXTRA:
            arrs[f'{cell}__d_{k.split(" ")[0]}_{"request" if "request" in k else "reading"}'] = c['dx'][k]
    with open(prefix + '.npz', 'xb') as fh:
        np.savez_compressed(fh, **arrs)
    open(prefix + '_rows.csv', 'x').write(buf.getvalue())
    open(prefix + '.txt', 'x').write('\n'.join(OUT) + '\n')
    print('\n'.join(OUT), flush=True)
    print(f'wrote {prefix}.txt, {prefix}.npz, {prefix}_rows.csv')


if __name__ == '__main__':
    main(sys.argv[1])
