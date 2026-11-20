"""Scores the 74 oracle-line pictures (a search start stepped toward a later work's vector) as nosearch_v5.py scores the other no-search families, from exact readings: far score, rank, gap, geometry.
Usage (CPU): python -B nosearch_oracle_v1.py OUTPREFIX -> OUTPREFIX.txt and OUTPREFIX_rows.csv, each refused if it exists."""
import csv
import importlib.util
import io
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
EV_FILE = f'{C}/controls_evaluate.py'
REQ = f'{B}/tests12/nosearch/nosearch_v5_oracle_requests.npz'
EMB = f'{B}/tests12/afterrun/oracle/oracle_emb_h200_v1.npz'
LST = f'{B}/tests12/afterrun/oracle/oracle_list_v1.txt'
INP = f'{B}/inputs/oracle_inputs.npz'
base = lambda p: os.path.basename(str(p))  # noqa: E731
OUT = []


def say(s=''):
    OUT.append(s)


def load_ev():
    spec = importlib.util.spec_from_file_location('controls_evaluate', EV_FILE)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def main(prefix):
    for ext in ('.txt', '_rows.csv'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    ev = load_ev()
    OBS, K, unit, chord, FID = ev.OBS, ev.K, ev.unit, ev.chord, float(ev.FID)
    P, T, Pz = ev.references(O)
    Pid, Part = P['id'], P['artist']
    own = {o: P[o]['own'] for o in OBS}
    lev = {o: float(np.percentile(own[o], 99)) for o in OBS}
    row_in_P = {int(i): k for k, i in enumerate(Pid.tolist())}
    deg = lambda a, b: np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))  # noqa: E731

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
            assert np.isfinite(d[o]).all()
        return d

    def rank(v, o):
        return 100.0 * np.searchsorted(np.sort(own[o]), v, 'left') / len(own[o])

    far_of = lambda d: (d[OBS[0]] > lev[OBS[0]]) & (d[OBS[1]] > lev[OBS[1]])  # noqa: E731

    q, em = np.load(REQ, allow_pickle=True), np.load(EMB, allow_pickle=True)
    lst = [ln.strip() for ln in open(LST) if ln.strip()]
    ff = [str(x) for x in q['frame_file']]
    assert len(lst) == len(ff) == len(em['paths']) == 74
    assert [base(p) for p in lst] == ff and [str(p) for p in em['paths']] == lst, 'readings are not in request order'
    v = unit(q['v'])
    start, target, angle, tang = [str(x) for x in q['start']], q['target'].astype(int), q['angle'].astype(float), q['target_angle'].astype(float)

    inp = np.load(INP, allow_pickle=True)
    starts, stags = unit(inp['starts']), [str(t) for t in inp['start_tags']]
    targets, tnames = unit(inp['targets']), [str(t) for t in inp['target_names']]
    obs_ids = np.load(ev.IDS, allow_pickle=True)['ids'].astype(np.int64)
    Ec = unit(np.load(ev.IDS, allow_pickle=True)['embeddings'])
    st = np.load(f'{O}/pools/starts_pre1900.npz', allow_pickle=True)
    st_art = dict(zip(np.asarray(st['rows_full']).astype(int).tolist(), [str(a) for a in st['artist']]))
    sinfo = {}
    for k, tg in enumerate(stags):
        c = Ec @ starts[k]
        r = int(np.argmax(c))
        assert c[r] > 0.99999, (tg, c[r])
        wid = int(obs_ids[r])
        sinfo[tg] = dict(row=r, id=wid, artist=st_art[r], inP=wid in row_in_P)
        say(f'start {tg}: observer row {r} work id {wid} artist {st_art[r]} (cosine {c[r]:.6f}); a row of P {wid in row_in_P}; P rows by that artist {int((Part == st_art[r]).sum())}')
    pz = np.load(f'{O}/pos_emb.npz', allow_pickle=True)
    pos_v, pos_id, pos_art = unit(pz['vec']), ev.pid(pz['paths']), np.array([str(a) for a in pz['artist']])
    tinfo = {}
    for j in sorted(set(target.tolist()) - {-1}):
        c = pos_v @ targets[j]
        r = int(np.argmax(c))
        assert c[r] > 0.99999 and pos_art[r] == tnames[j], (j, c[r], pos_art[r], tnames[j])
        tinfo[j] = dict(row=r, id=int(pos_id[r]), artist=pos_art[r], inP=int(pos_id[r]) in row_in_P)

    obs = {o: em[f'emb_{o}'] for o in OBS}
    masks = []
    for i in range(74):
        arts = [sinfo[start[i]]['artist']] + ([tinfo[target[i]]['artist']] if target[i] >= 0 else [])
        masks.append(np.flatnonzero(np.isin(Part, arts)))
    da, dv = far_scores(obs), far_scores(obs, masks)
    gap = chord((v * unit(em['emb_clip-vit-l-14'])).sum(1))
    fa_a, fa_v, faith = far_of(da), far_of(dv), gap <= FID
    a_start = np.array([float(deg(v[i], starts[stags.index(start[i])])) for i in range(74)])
    a_targ = np.array([float(deg(v[i], targets[target[i]])) if target[i] >= 0 else np.nan for i in range(74)])
    a_rr = np.degrees(np.arccos(np.clip(1 - gap ** 2 / 2, -1, 1)))
    assert np.abs(a_start - angle).max() < 1e-3, 'angle from start differs from the asked angle'
    # oracle.json holds the render environment's readings: its far scores should be close to these, not equal
    oj = json.load(open(f'{O}/oracle/oracle.json'))['frames']
    dj = {o: np.array([f[f'knn15_{o}'] for f in oj]) for o in OBS}
    say('oracle.json (render-environment DINOv3 / SigLIP, P alone) against these exact readings, max abs difference of the far score: '
        + ', '.join(f'{o} {np.abs(dj[o] - da[o]).max():.4f}' for o in OBS) + f"; json far-and-faithful (render-environment CLIP) {sum(bool(f['passes_both']) for f in oj)}")

    say()
    say('Each target work\'s own repaint in the later 512 (pos_emb.npz): far score alone / artist out, far, gap, faithful')
    pe = {o: pz[f'emb_{o}'] for o in OBS}
    pgap = chord((pos_v * unit(pz['emb_clip-vit-l-14'])).sum(1))
    for j, t in tinfo.items():
        r = t['row']
        d1 = far_scores({o: pe[o][[r]] for o in OBS})
        d2 = far_scores({o: pe[o][[r]] for o in OBS}, [np.flatnonzero(Part == t['artist'])])
        say(f"  target {j} {t['artist']} work id {t['id']} (pos row {r}; a row of P {t['inP']}; P rows by artist {int((Part == t['artist']).sum())}): "
            f"alone {d1[OBS[0]][0]:.4f} / {d1[OBS[1]][0]:.4f} far {int(far_of(d1)[0])}; artist out {d2[OBS[0]][0]:.4f} / {d2[OBS[1]][0]:.4f} far {int(far_of(d2)[0])}; "
            f"gap {pgap[r]:.4f} faithful {int(pgap[r] <= FID)}")

    def cells(m, fa):
        return [int((fa & faith & m).sum()), int((fa & ~faith & m).sum()), int((~fa & faith & m).sum()), int((~fa & ~faith & m).sum())]

    say()
    say(f'far levels (P own 99th): ' + ', '.join(f'{o} {lev[o]:.6f}' for o in OBS) + f'; FID {FID:.15f}')
    say('Per line (start -> target) and pooled. Cells FF FN NF NN; variant = start and target artists left out; alone = P alone.')
    say(f'{"line":44s} {"n":>3s} | variant: far FF FN NF NN | alone: far FF FN NF NN | med gap | start-target angle')
    lines = [('start pictures (angle 0)', target < 0)]
    for tg in stags:
        for j in dict.fromkeys(target[(np.array(start) == tg) & (target >= 0)].tolist()):
            lines.append((f'{tg} -> {j} {tinfo[j]["artist"]}', (np.array(start) == tg) & (target == j)))
    lines.append(('all line pictures (start pictures excluded)', target >= 0))
    for name, m in lines:
        cv, ca = cells(m, fa_v), cells(m, fa_a)
        th = f'{np.nanmedian(tang[m]):.1f}' if (target[m] >= 0).any() else '-'
        say(f'{name:44s} {int(m.sum()):3d} | {cv[0] + cv[1]:11d} {cv[0]:2d} {cv[1]:2d} {cv[2]:2d} {cv[3]:2d} | {ca[0] + ca[1]:9d} {ca[0]:2d} {ca[1]:2d} {ca[2]:2d} {ca[3]:2d} | {np.median(gap[m]):.4f} | {th}')
    say()
    say('Far-and-faithful pictures (either rule): where they sit on their line')
    for i in np.flatnonzero((fa_a | fa_v) & faith):
        say(f'  frame {ff[i]} {start[i]} -> {target[i]} {tinfo[target[i]]["artist"] if target[i] >= 0 else "start"}: asked {angle[i]:.1f} deg of {tang[i]:.1f}; '
            f'from start {a_start[i]:.1f}, from target {a_targ[i]:.1f}; FF alone {int(fa_a[i])} variant {int(fa_v[i])}; gap {gap[i]:.4f}')

    cols = ['frame_file', 'start', 'start_id', 'target', 'target_artist', 'target_id', 'angle_asked', 'start_target_angle', 'position_fraction',
            'angle_from_start', 'angle_from_target', 'angle_request_reading', 'gap', 'faithful', 'alone_d_dino', 'alone_d_siglip', 'alone_rank_dino',
            'alone_rank_siglip', 'alone_far', 'variant_d_dino', 'variant_d_siglip', 'variant_rank_dino', 'variant_rank_siglip', 'variant_far', 'n_left_out']
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for i in range(74):
        t = tinfo.get(int(target[i]), {})
        w.writerow([ff[i], start[i], sinfo[start[i]]['id'], int(target[i]), t.get('artist', ''), t.get('id', ''), f'{angle[i]:.4f}',
                    '' if np.isnan(tang[i]) else f'{tang[i]:.4f}', '' if np.isnan(tang[i]) else f'{angle[i] / tang[i]:.4f}', f'{a_start[i]:.4f}',
                    '' if np.isnan(a_targ[i]) else f'{a_targ[i]:.4f}', f'{a_rr[i]:.4f}', f'{gap[i]:.6f}', int(faith[i]),
                    f'{da[OBS[0]][i]:.6f}', f'{da[OBS[1]][i]:.6f}', f'{rank(da[OBS[0]][i], OBS[0]):.2f}', f'{rank(da[OBS[1]][i], OBS[1]):.2f}', int(fa_a[i]),
                    f'{dv[OBS[0]][i]:.6f}', f'{dv[OBS[1]][i]:.6f}', f'{rank(dv[OBS[0]][i], OBS[0]):.2f}', f'{rank(dv[OBS[1]][i], OBS[1]):.2f}', int(fa_v[i]),
                    len(masks[i])])
    say()
    say(f'CSV: 74 rows. Columns: ' + ', '.join(cols))
    OUT.insert(0, f'nosearch_oracle_v1.py; readings {EMB}. Counts and geometry only.')
    open(prefix + '_rows.csv', 'x').write(buf.getvalue())
    open(prefix + '.txt', 'x').write('\n'.join(OUT) + '\n')
    print('\n'.join(OUT), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
