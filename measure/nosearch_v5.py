"""The requests painted without any search (regions, gate, prior, prior_rerender, chains, oracle), one row per picture: far score, rank, gap, geometry.
Usage (CPU): python -B nosearch_v5.py OUTPREFIX -> OUTPREFIX.txt, OUTPREFIX_rows.csv, OUTPREFIX_oracle_requests.npz (refused if any exists)"""
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
GRID_NPZ = f'{B}/tests12/grid/grid_v1.npz'
R2 = f'{B}/tests12/t2/renders_v1'
DIM644 = 644   # a CLIP coordinate whose reading offset (reading minus request) is reported
base = lambda p: os.path.basename(str(p))
OUT = []


def say(s=''):
    OUT.append(s)


def load_ev():
    spec = importlib.util.spec_from_file_location('controls_evaluate', EV_FILE)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def load_shards(folder, n_expected):
    """Shard manifests + exact readings of one stage, put in frame order (manifest 'row' = frame index). Asserts the paths agree."""
    man, emb = [], []
    for s in range(64):
        d = f'{folder}/shard{s}'
        if not os.path.isdir(d):
            break
        m, e = np.load(f'{d}/manifest.npz', allow_pickle=True), np.load(f'{d}/emb.npz', allow_pickle=True)
        assert len(m['paths']) == len(e['paths']) and [base(p) for p in m['paths']] == [base(p) for p in e['paths']], d
        man.append(m)
        emb.append(e)
    row = np.concatenate([m['row'] for m in man]).astype(int)
    order = np.argsort(row)
    assert (row[order] == np.arange(n_expected)).all(), f'{folder}: shard rows are not exactly the frames'
    cat = lambda parts, k: np.concatenate([p[k] for p in parts])[order]
    M = {k: cat(man, k) for k in ('tag', 'seed', 'v', 'paths')}
    E = {k: cat(emb, k) for k in ('emb_clip-vit-l-14', 'emb_dinov3-l', 'emb_siglip-so400m')}
    return M, E, len(man)


def stored_scores(f):
    """dP k15 per observer and fid_exact of a scored stage (baselines_score.py groups base_shard*), in frame order."""
    z = np.load(f, allow_pickle=True)
    groups = sorted({k.split('__')[0] for k in z.files if k.startswith('base_shard')})
    row = np.concatenate([z[g + '__row'] for g in groups]).astype(int)
    order = np.argsort(row)
    assert (row[order] == np.arange(len(row))).all(), f'{f}: stored rows are not exactly the frames'
    get = lambda k: np.concatenate([z[g + '__' + k] for g in groups])[order]
    return {'dinov3-l': get('dP__dinov3-l__k15'), 'siglip-so400m': get('dP__siglip-so400m__k15'), 'fid': get('fid_exact'), 'row': row[order]}


def main(prefix):
    for ext in ('.txt', '_rows.csv', '_oracle_requests.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    ev = load_ev()
    OBS, K, unit, chord, FID = ev.OBS, ev.K, ev.unit, ev.chord, float(ev.FID)
    P, T, Pz = ev.references(O)
    Pid, Part = P['id'], P['artist']
    own = {o: P[o]['own'] for o in OBS}
    lev = {o: float(np.percentile(own[o], 99)) for o in OBS}
    row_in_P = {int(i): k for k, i in enumerate(Pid.tolist())}
    Pv = unit(Pz['vec'])
    Pgap = chord((Pv * unit(Pz['emb_clip-vit-l-14'])).sum(1))
    mP = unit(Pv.mean(0))
    st = np.load(f'{O}/pools/starts_pre1900.npz', allow_pickle=True)
    mpre = unit(np.asarray(st['mean_pre'], np.float64))
    deg = lambda a, b: np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))
    art_rows = lambda a: np.flatnonzero(Part == a)
    NONE = np.zeros(0, int)

    # as grid_v1.py
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

    say(f'P = push_emb.npz (2,048); far levels (P own 99th): ' + ', '.join(f'{o} {lev[o]:.6f}' for o in OBS) + f'; FID {FID:.15f}; P gap median {np.median(Pgap):.4f}')
    say(f'mean directions: P requests mean vs starts_pre1900 mean_pre: {float(deg(mP, mpre)):.2f} deg apart')

    rows = []   # one dict per picture
    groups = {}   # (family, sub) -> list of row indices

    def add(family, sub, group, v, emb, masks, geo, ids, files, seeds, maskvar=None, exact=True):
        """Score n pictures of one sub-family. geo: dict of per-picture arrays (angles, positions); ids: dict of per-picture work ids."""
        v = unit(v)
        n = len(v)
        obs = {o: emb[f'emb_{o}'] for o in OBS}
        da, dm = far_scores(obs), far_scores(obs, masks)
        dv = far_scores(obs, maskvar) if maskvar is not None else None
        gap = chord((v * unit(emb['emb_clip-vit-l-14'])).sum(1))
        for i in range(n):
            r = dict(family=family, sub=sub, group=group, file=base(files[i]), render_seed=int(seeds[i]),
                     **{k: (int(a[i]) if a[i] >= 0 else '') for k, a in ids.items()},
                     angle_from_mean_P=float(deg(v[i], mP)), angle_from_mean_pre=float(deg(v[i], mpre)),
                     **{k: float(a[i]) for k, a in geo.items()},
                     angle_request_reading=float(np.degrees(np.arccos(np.clip(1 - gap[i] ** 2 / 2, -1, 1)))), gap=float(gap[i]),
                     faithful=int(gap[i] <= FID), n_left_out=len(masks[i]))
            for tag, d in (('', dm), ('alone_', da)) + ((('maskvar_', dv),) if dv is not None else ()):
                rd, rs = rank(d[OBS[0]][i], OBS[0]), rank(d[OBS[1]][i], OBS[1])
                r.update({f'{tag}d_dino': float(d[OBS[0]][i]), f'{tag}d_siglip': float(d[OBS[1]][i]), f'{tag}rank_dino': float(rd),
                          f'{tag}rank_siglip': float(rs), f'{tag}minrank': float(min(rd, rs)),
                          f'{tag}far': int(d[OBS[0]][i] > lev[OBS[0]] and d[OBS[1]][i] > lev[OBS[1]])})
            r['off644'] = float(unit(emb['emb_clip-vit-l-14'][i])[DIM644] - v[i][DIM644])
            r['near_gap_bar'] = int(abs(gap[i] - FID) < 0.01)
            r['near_far_bar'] = int(min(abs(dm[o][i] - lev[o]) for o in OBS) < 0.001)
            groups.setdefault((family, sub), []).append(len(rows))
            rows.append(r)
        return da, gap

    R = np.load(f'{O}/eval/regions/regions_requests.npz', allow_pickle=True)
    M, E, ns = load_shards(f'{O}/eval/regions/stage1', len(R['v']))
    fam = np.array([str(x) for x in R['family']])
    assert (np.array([str(x) for x in M['tag']]) == fam).all() and np.abs(unit(M['v']) - unit(R['v'])).max() < 1e-6
    assert len(fam) == 868 and (R['render_seed'] == M['seed']).all()
    S = stored_scores(f'{O}/eval/regions/stage1.npz')
    a_row, b_row, p2_row = R['a_row'].astype(int), R['b_row'].astype(int), R['p2_row'].astype(int)
    P2z = np.load(f'{O}/push2_emb.npz', allow_pickle=True)
    assert not any(int(i) in row_in_P for i in ev.pid(P2z['paths']))
    GROUP_R = {'IW': 'midpoint of two works (48)', 'CL': 'line between two regions\' centres (36)', 'PP35': '35 deg step past a work (480)',
              'RD35': '35 deg step past a work (480)', 'SH35': '35 deg step past a work (480)', 'GD35': '35 deg step past a work (480)'}
    cen, ra_, rb_ = R['region_centre'], R['region_a'].astype(int), R['region_b'].astype(int)
    chk_d, chk_g = [], []
    for f_ in dict.fromkeys(fam):
        s = np.flatnonzero(fam == f_)
        if f_ == 'IW':
            masks = [np.union1d(art_rows(Part[a_row[i]]), art_rows(Part[b_row[i]])) for i in s]
        elif f_ in ('CL', 'P2RE'):
            masks = [NONE for i in s]
        else:
            masks = [art_rows(Part[b_row[i]]) for i in s]
        if f_ == 'CL':
            g1, g2, pos = deg(R['v'][s], cen[rb_[s]]), deg(R['v'][s], cen[ra_[s]]), R['t'][s]
        elif f_ == 'P2RE':
            g1, g2, pos = deg(R['v'][s], P2z['vec'][p2_row[s]]), np.full(len(s), np.nan), np.zeros(len(s))
        else:
            g1, g2 = deg(R['v'][s], Pv[b_row[s]]), deg(R['v'][s], Pv[a_row[s]]) if (a_row[s] >= 0).all() else np.full(len(s), np.nan)
            pos = R['phi'][s] if f_ != 'IW' else np.full(len(s), np.nan)
        if f_ not in ('CL', 'P2RE'):
            assert np.abs(g1 - R['angle_from_b'][s]).max() < 1e-3, f_
        ids = {'start_id': np.array([Pid[b] if b >= 0 else -1 for b in b_row[s]]), 'second_id': np.array([Pid[a] if a >= 0 else -1 for a in a_row[s]])}
        geo = {'angle_from_start': g1, 'angle_from_second': g2, 'position': pos, 'position2': np.full(len(s), np.nan),
               'in_window': R['in_window'][s].astype(float), 'b_ordinary': R['b_ordinary'][s].astype(float)}
        da, gap = add('regions', f_, GROUP_R.get(f_, ''), R['v'][s], {k: E[k][s] for k in E}, masks, geo, ids, M['paths'][s], M['seed'][s])
        for o in OBS:
            chk_d.append(np.abs(da[o] - S[o][s]).max())
        chk_g.append(np.abs(gap - S['fid'][s]).max())
    assert max(chk_d) < 1e-5 and max(chk_g) < 1e-5, (max(chk_d), max(chk_g))
    say(f'regions: {ns} shards, 868 frames aligned with regions_requests.npz; P-alone far scores and gaps equal stage1.npz (dP k15, fid_exact) to '
        f'{max(chk_d):.1e} / {max(chk_g):.1e}')

    Q = np.load(f'{O}/eval/probe/probe_requests.npz', allow_pickle=True)
    M, E, ns = load_shards(f'{O}/eval/probe/stage1', len(Q['v']))
    fam = np.array([str(x) for x in Q['family']])
    assert (np.array([str(x) for x in M['tag']]) == fam).all() and np.abs(unit(M['v']) - unit(Q['v'])).max() < 1e-6 and len(fam) == 1056
    S = stored_scores(f'{O}/eval/probe/stage1.npz')
    obs_ids = np.load(ev.IDS, allow_pickle=True)['ids'].astype(np.int64)
    Ec = np.load(ev.IDS, allow_pickle=True)['embeddings']   # stored CLIP vectors of the corpus (the starts' x0)
    srow = Q['start_row'].astype(int)
    st_art = dict(zip(np.asarray(st['rows_full']).astype(int).tolist(), [str(a) for a in st['artist']]))
    start_info = {}
    for r_ in sorted(set(srow.tolist()) - {-1}):
        wid = int(obs_ids[r_])
        start_info[r_] = (wid, wid in row_in_P, st_art[r_], unit(Ec[r_]))
        say(f'gate start: observer row {r_} work id {wid} artist {st_art[r_]}: a row of P {wid in row_in_P}; P rows by that artist {len(art_rows(st_art[r_]))}')
    partners = Q['partner_rows'].astype(int)
    GROUP_G = {'LR': 'lines past a real work or toward the sparse edge (450)', 'L2': 'lines past a real work or toward the sparse edge (450)',
              'LT': 'lines past a real work or toward the sparse edge (450)'}
    chk_d, chk_g = [], []
    for f_ in dict.fromkeys(fam):
        for win in (0, 1):
            s = np.flatnonzero((fam == f_) & (Q['filter_fail'].astype(int) == win))
            if not len(s):
                continue
            masks, mvar, g1, g2 = [], [], [], []
            for i in s:
                if srow[i] >= 0:
                    wid, inP, sa, x0 = start_info[srow[i]]
                    g1.append(float(deg(Q['v'][i], x0)))
                else:
                    sa, inP = None, False
                    g1.append(np.nan)
                pr = [j for j in partners[i] if j >= 0] if f_ in ('LR', 'L2', 'LT', 'M3') else []
                mm = [art_rows(Part[j]) for j in pr] + ([art_rows(sa)] if inP else [])   # source works that are rows of P: partners (and the start if it were)
                mv = mm + ([art_rows(sa)] if sa is not None else [])
                masks.append(np.unique(np.concatenate(mm)) if mm else NONE)
                mvar.append(np.unique(np.concatenate(mv)) if mv else NONE)
                g2.append(float(deg(Q['v'][i], Pv[pr[-1]])) if pr else np.nan)
            ids = {'start_id': np.array([start_info[srow[i]][0] if srow[i] >= 0 else -1 for i in s]),
                   'second_id': np.array([Pid[[j for j in partners[i] if j >= 0][-1]] if f_ in ('LR', 'L2', 'LT', 'M3') else -1 for i in s])}
            geo = {'angle_from_start': np.array(g1), 'angle_from_second': np.array(g2), 'position': np.where(np.isnan(Q['t'][s]), Q['T'][s], Q['t'][s]),
                   'position2': Q['t2'][s], 'in_window': 1.0 - Q['filter_fail'][s]}
            ok_ = ~np.isnan(np.array(g1))
            assert not ok_.any() or np.abs(np.array(g1)[ok_] - Q['angle_from_start'][s][ok_]).max() < 1e-3, f_
            sub = f'{f_}, ' + ('in window' if win == 0 else 'out of window')
            group = GROUP_G[f_] if (f_ in GROUP_G and win == 0) else ''
            da, gap = add('gate', sub, group, Q['v'][s], {k: E[k][s] for k in E}, masks, geo, ids, M['paths'][s], M['seed'][s], maskvar=mvar)
            for o in OBS:
                chk_d.append(np.abs(da[o] - S[o][s]).max())
            chk_g.append(np.abs(gap - S['fid'][s]).max())
    assert max(chk_d) < 1e-5 and max(chk_g) < 1e-5, (max(chk_d), max(chk_g))
    say(f'gate: {ns} shards, 1,056 frames aligned with probe_requests.npz; P-alone far scores and gaps equal stage1.npz to {max(chk_d):.1e} / {max(chk_g):.1e}; '
        'angle_from_start uses the start\'s stored CLIP vector (observer file row), position = t for lines, T for record steps, position2 = t2 (L2)')

    sa_ = np.load(f'{O}/eval/sens_all.npz', allow_pickle=True)
    for label, mf, ef, k0 in (('prior', f'{O}/baselines/prior/manifest.npz', f'{O}/baselines/prior/emb.npz', 0),
                              ('prior_rerender', f'{R2}/prior256_v1/prior/manifest.npz', f'{R2}/prior256_v1/emb.npz', None)):
        man, em = np.load(mf, allow_pickle=True), np.load(ef, allow_pickle=True)
        if k0 is None:
            kat = [ln.strip() for ln in open(f'{R2}/prior256_v1/inputs/expected_kat.txt') if ln.strip()]
            assert [str(p) for p in em['paths'][:len(kat)]] == kat
            k0 = len(kat)
        assert [base(p) for p in em['paths'][k0:]] == [base(p) for p in man['paths']] and len(em['paths']) == k0 + len(man['paths'])
        kind = np.array([str(x) for x in man['kind']])
        for kd, n in (('cone', 192), ('sphere', 64)):
            s = np.flatnonzero(kind == kd)
            assert len(s) == n
            geo = {'angle_from_start': np.full(n, np.nan), 'angle_from_second': np.full(n, np.nan), 'position': np.full(n, np.nan),
                   'position2': np.full(n, np.nan), 'in_window': np.full(n, np.nan)}
            ids = {'start_id': -np.ones(n, int), 'second_id': -np.ones(n, int)}
            group = (f'random requests inside the pre-1900 region (192)' if kd == 'cone' else 'random requests anywhere on the sphere (64)') if label == 'prior' \
                else ''
            emb = {k: em[k][k0:][s] for k in ('emb_clip-vit-l-14', 'emb_dinov3-l', 'emb_siglip-so400m')}
            da, gap = add(label, kd, group, man['v'][s], emb, [NONE] * n, geo, ids, man['paths'][s], man['seed'][s])
            if label == 'prior':
                assert [str(t) for t in sa_['base_prior__tag']] == kind.tolist() and (sa_['base_prior__row'] == man['row']).all()
                e1 = max(np.abs(da[o] - sa_[f'base_prior__dP__{o}__k15'][s]).max() for o in OBS)
                e2 = np.abs(gap - sa_['base_prior__fid_exact'][s]).max()
                assert e1 < 1e-5 and e2 < 1e-5, (e1, e2)
                say(f'prior {kd}: P-alone far scores and gaps equal sens_all.npz base_prior__* to {e1:.1e} / {e2:.1e}')
            else:
                G = np.load(GRID_NPZ, allow_pickle=True)
                j = [str(x) for x in G['group_names']].index(f'2 random requests, {kd}')
                e = max(np.abs(da[OBS[0]] - G[f'g{j:02d}__d_dino']).max(), np.abs(da[OBS[1]] - G[f'g{j:02d}__d_siglip']).max(), np.abs(gap - G[f'g{j:02d}__gap']).max())
                say(f'prior_rerender {kd}: far scores and gaps vs grid_v1.npz group "2 random requests, {kd}": max difference {e:.1e}')

    CH = (('chains', 'chains_h200_qg', 'round trips (48 = step 6)'),)
    for label, folder, group in CH:
        man, em = np.load(f'{O}/baselines/{folder}/manifest.npz', allow_pickle=True), np.load(f'{O}/baselines/{folder}/emb.npz', allow_pickle=True)
        assert [base(p) for p in man['paths']] == [base(p) for p in em['paths']] and len(man['paths']) == 336
        step, src = man['step'].astype(int), man['row'].astype(int)
        s0 = step == 0
        assert np.abs(unit(man['v'][s0]) - Pv[src[s0]]).max() < 1e-6, 'step 0 is not the source work\'s vector'
        tags = np.array([str(t) for t in man['tag']])
        if folder == 'chains_h200_qg':   # request of step k+1 against the exact reading of picture k
            cs = []
            for k in range(6):
                a_, b_ = np.flatnonzero(step == k), np.flatnonzero(step == k + 1)
                assert (tags[a_] == tags[b_]).all()
                cs.append((unit(man['v'][b_]) * unit(em['emb_clip-vit-l-14'][a_])).sum(1))
            say(f'chains_h200_qg: cosine of request k+1 with the exact reading of picture k: min {np.concatenate(cs).min():.6f}, median {np.median(np.concatenate(cs)):.6f}')
        farby = []
        for k in range(7):
            s = np.flatnonzero(step == k)
            masks = [art_rows(Part[src[i]]) for i in s]
            geo = {'angle_from_start': deg(man['v'][s], Pv[src[s]]), 'angle_from_second': np.full(len(s), np.nan), 'position': np.full(len(s), float(k)),
                   'position2': np.full(len(s), np.nan), 'in_window': np.full(len(s), np.nan)}
            ids = {'start_id': Pid[src[s]], 'second_id': -np.ones(len(s), int)}
            da, gap = add(label, f'step {k}', group if (k == 6 or label != 'chains') else '', man['v'][s],
                          {kk: em[kk][s] for kk in ('emb_clip-vit-l-14', 'emb_dinov3-l', 'emb_siglip-so400m')}, masks, geo, ids, man['paths'][s], man['seed'][s])
            farby.append(int(((da[OBS[0]] > lev[OBS[0]]) & (da[OBS[1]] > lev[OBS[1]])).sum()))
        say(f'{folder}: P-alone far by step {farby}')

    inp = np.load(f'{B}/inputs/oracle_inputs.npz', allow_pickle=True)
    oj = json.load(open(f'{O}/oracle/oracle.json'))
    starts, start_tags = np.asarray(inp['starts'], np.float64), [str(t) for t in inp['start_tags']]
    targets, nearest = np.asarray(inp['targets'], np.float64), np.asarray(inp['nearest'])
    angles = [float(t) for t in oj['angles']]
    ov, orec = [], []
    for si, (x, tag) in enumerate(zip(starts, start_tags)):   # as in oracle.py
        x = x / np.linalg.norm(x)
        ov.append(x)
        orec.append((tag, -1, 0.0, np.nan))
        for tj in nearest[si]:
            t = targets[int(tj)] / np.linalg.norm(targets[int(tj)])
            th = float(np.degrees(np.arccos(np.clip(t @ x, -1, 1))))
            d = t - (t @ x) * x
            d = d / np.linalg.norm(d)
            for ang in sorted(set(angles) | {round(th, 1)}):
                r = np.radians(ang)
                v = np.cos(r) * x + np.sin(r) * d
                ov.append(v / np.linalg.norm(v))
                orec.append((tag, int(tj), float(ang), th))
    fr = oj['frames']
    assert len(fr) == len(ov) == 74
    for i, (f, (tag, tj, ang, th)) in enumerate(zip(fr, orec)):
        assert f['start'] == tag and int(f['target']) == tj and abs(float(f['angle']) - ang) < 1e-9 and base(f['path']) == f'{i:04d}.png', i
    ov = np.array(ov)
    sx = {tag: starts[k] / np.linalg.norm(starts[k]) for k, tag in enumerate(start_tags)}
    for i, (tag, tj, ang, th) in enumerate(orec):
        rows.append(dict(family='oracle', sub=('start' if tj < 0 else f'line to target {tj}'), group='',
                         file=f'{i:04d}.png', render_seed=0, start_id='', second_id='', angle_from_mean_P=float(deg(ov[i], mP)), angle_from_mean_pre=float(deg(ov[i], mpre)),
                         angle_from_start=float(deg(ov[i], sx[tag])), angle_from_second=(float(deg(ov[i], targets[tj])) if tj >= 0 else np.nan),
                         position=ang, position2=th, in_window=np.nan))
        groups.setdefault(('oracle', 'all frames (requests only)'), []).append(len(rows) - 1)
    with open(prefix + '_oracle_requests.npz', 'xb') as fh:
        np.savez(fh, v=ov, start=np.array([r[0] for r in orec]), target=np.array([r[1] for r in orec]), angle=np.array([r[2] for r in orec]),
                 target_angle=np.array([r[3] for r in orec]), frame_file=np.array([f'{i:04d}.png' for i in range(74)]))
    say(f'oracle: 74 requests rebuilt from oracle_inputs.npz by oracle.py\'s formula; start, target and angle equal oracle.json frame by frame; '
        f'no exact reading of these pictures exists (oracle.py read them with the render environment\'s CLIP), so no far / faithful cell is computed')

    say()
    say('reader signatures: gap median of pictures whose request is a real work\'s own vector, against the same works\' reference repaints; '
        'median offset in CLIP coordinate 644 (reading minus request)')
    say(f'  P (push_emb.npz) gap median {np.median(Pgap):.4f}; off644 median {np.median(unit(Pz["emb_clip-vit-l-14"])[:, DIM644] - Pv[:, DIM644]):+.4f}')
    ix = lambda fam_, sub_: groups[(fam_, sub_)]
    bb = [rows[i] for i in ix('regions', 'BASE')]
    say(f'  regions BASE: gap median {np.median([r["gap"] for r in bb]):.4f} vs P repaints of the same b works {np.median([Pgap[row_in_P[r["start_id"]]] for r in bb]):.4f}')
    p2 = [rows[i] for i in ix('regions', 'P2RE')]
    P2gap = chord((unit(P2z['vec']) * unit(P2z['emb_clip-vit-l-14'])).sum(1))
    say(f'  regions P2RE: gap median {np.median([r["gap"] for r in p2]):.4f} vs push2_emb repaints of the same works {np.median(P2gap[p2_row[np.array([str(x) for x in R["family"]]) == "P2RE"]]):.4f}')
    c0 = [rows[i] for i in ix('chains', 'step 0')]
    say(f'  chains step 0: gap median {np.median([r["gap"] for r in c0]):.4f} vs P repaints of the same works {np.median([Pgap[row_in_P[r["start_id"]]] for r in c0]):.4f}')
    for (fa, sb), ii in groups.items():
        if not ((fa == 'regions' and sb in ('BASE', 'P2RE')) or (fa.startswith('chains') and sb == 'step 0')):
            continue   # only requests that are a real work's own vector: elsewhere the request itself leaves the corpus and the offset says nothing about the reader
        say(f'  off644 median {np.median([rows[i]["off644"] for i in ii]):+.4f}  {fa} / {sb}')

    def cells(ii, tag=''):
        far = np.array([rows[i][f'{tag}far'] for i in ii], bool)
        fa = np.array([rows[i]['faithful'] for i in ii], bool)
        return [int((far & fa).sum()), int((far & ~fa).sum()), int((~far & fa).sum()), int((~far & ~fa).sum())]

    say()
    say('Per family / sub-family. far = beyond P\'s own 99th in BOTH observers; masked = source work\'s P row and its artist left out, '
        'alone = P alone. Cells FF FN NF NN = far&faithful, far&not, near&faithful, near&not. Medians: minrank (masked), gap. '
        'near-bar: pictures within 0.01 of the gap bar / within 0.001 of a far level in either observer.')
    say(f'{"family / sub-family":58s} {"n":>4s} | masked: far  FF  FN  NF  NN | alone: far  FF | maskvar: far FF | med minrank  med gap | near gap / far')
    for (fa, sb), ii in groups.items():
        if fa == 'oracle':
            say(f'{fa + " / " + sb:58s} {len(ii):4d} | requests only, no exact readings')
            continue
        cm, ca = cells(ii), cells(ii, 'alone_')
        mv = cells(ii, 'maskvar_') if 'maskvar_far' in rows[ii[0]] else None
        mvs = f'{mv[0] + mv[1]:4d} {mv[0]:3d}' if mv else '   -   -'
        say(f'{fa + " / " + sb:58s} {len(ii):4d} | {cm[0] + cm[1]:11d} {cm[0]:3d} {cm[1]:3d} {cm[2]:3d} {cm[3]:3d} | {ca[0] + ca[1]:10d} {ca[0]:3d} | {mvs:>15s} | '
            f'{np.median([rows[i]["minrank"] for i in ii]):11.1f} {np.median([rows[i]["gap"] for i in ii]):8.4f} | '
            f'{sum(rows[i]["near_gap_bar"] for i in ii):4d} {sum(rows[i]["near_far_bar"] for i in ii):4d}')
    say()
    say('Per group (pictures pooled by group).')
    pages = {}
    for i, r in enumerate(rows):
        if r['family'] != 'oracle' and r['group']:
            pages.setdefault(r['group'], []).append(i)
    for pg, ii in pages.items():
        cm, ca = cells(ii), cells(ii, 'alone_')
        say(f'{pg:58s} n {len(ii):4d} | masked far {cm[0] + cm[1]:3d} far&faithful {cm[0]:3d} (FN {cm[1]}, NF {cm[2]}, NN {cm[3]}) | alone far {ca[0] + ca[1]:3d} far&faithful {ca[0]:3d} | '
            f'med minrank {np.median([rows[i]["minrank"] for i in ii]):.1f} med gap {np.median([rows[i]["gap"] for i in ii]):.4f}')

    cols = ['family', 'sub', 'group', 'file', 'render_seed', 'start_id', 'second_id', 'angle_from_start', 'angle_from_second', 'angle_from_mean_P',
            'angle_from_mean_pre', 'position', 'position2', 'in_window', 'b_ordinary', 'angle_request_reading', 'gap', 'faithful', 'n_left_out',
            'd_dino', 'd_siglip', 'rank_dino', 'rank_siglip', 'minrank', 'far', 'alone_d_dino', 'alone_d_siglip', 'alone_minrank', 'alone_far',
            'maskvar_d_dino', 'maskvar_d_siglip', 'maskvar_minrank', 'maskvar_far', 'off644', 'near_gap_bar', 'near_far_bar']
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        out = []
        for c in cols:
            x = r.get(c, '')
            if isinstance(x, float):
                x = '' if np.isnan(x) else f'{x:.6f}'
            out.append(x)
        w.writerow(out)
    say()
    say(f'CSV: {len(rows)} rows, one per picture (oracle rows: requests only). Columns: ' + ', '.join(cols))
    say('Column notes: angles in degrees; angle_from_second = regions a (CL: centre A, start = centre B), gate last line partner (M3: last of its three), '
        'oracle target; position = regions phi (CL t), gate t (record families T), chains step, oracle angle; position2 = gate t2, oracle target angle; '
        'b_ordinary = regions base below the 90th percentile of P of own distance in both observers (regions_requests.npz); start_id / second_id = work ids; d_*, rank_*, minrank, far = masked rule; alone_* = P alone; maskvar_* = gate variant.')
    open(prefix + '_rows.csv', 'x').write(buf.getvalue())
    open(prefix + '.txt', 'x').write('\n'.join(OUT) + '\n')
    print('\n'.join(OUT), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
