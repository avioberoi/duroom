"""Evaluator for the controls: replays the loop's admission rule on stored runs and decides the paired verdicts.
Usage: controls_evaluate.py selftest|table|safety OUT_ROOT [--pair V:X ...] [--verdict NAME=V:X,...] [--safety-cache FILE] [--dev] [--dump FILE] LABEL=RUN_DIR[,EXACT_EMB|-[,JOB_LOG|-]] ..."""
import argparse
import ast
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from local_rescore import BARS, FID, FID_LOOP, K, OBS, chord, unit  # noqa: E402,F401
from today_rescore import far, pid, reference, score  # noqa: E402

DELTA, WARMUP, DINO = 0.575, 1, 'dinov3-l'   # the loop's flags: --delta 0.575 --warmup 1 (--admit-q99, --fidelity-max FID_LOOP)
CODE = HERE
SAFETY = os.environ['DUROOM_OUT'] + '/cache/safety_checker'   # CompVis/stable-diffusion-safety-checker
GATE_EXACT = 0.7785    # the exact faithful line rounded to four places
GATE_QG = 0.7788       # the 99th percentile of the reference's request-to-reading chord on the loop reader itself (push_fidelity_loop_qg.npz)
GATES = (FID_LOOP, GATE_EXACT, GATE_QG)   # the only fidelity gates a run may carry: 0.8735 = loop encoder scale, 0.7785 = exact reader's percentile, 0.7788 = loop reader's percentile
GATE_NAME = {FID_LOOP: 'loop encoder scale', GATE_EXACT: 'exact faithful line', GATE_QG: "loop reader's 99th percentile"}
CONF_GATE, CONF_REQUIRE_QUICKGELU = GATE_QG, True   # strict mode refuses a cell at gate 0.7785; --dev accepts it
CONF_START = {6: 182034, 7: 126645, 8: 56762, 9: 56179, 10: 71536}   # cell number (= --seed) -> the start work id
CONF_STARTS = set(CONF_START.values())
ROLE_OF = {'ra': 'V', 'pin': 'C1', 'c2': 'C2'}   # tags qgc_ra_sN (V), qgc_pin_sN (C1), qgc_c2_sN (C2), N = 6 to 10
KAT_FLOOR = 0.999   # floor for each of the 8 known-answer cosines
IDS = os.environ['DUROOM_EMB'] + '/clip-vit-l-14.npz'   # row of the unfiltered observer file -> work id (start_idx indexes it)


def references(O):
    Pz, Tz = np.load(f'{O}/push_emb.npz', allow_pickle=True), np.load(f'{O}/push_today_emb.npz', allow_pickle=True)
    pool = np.load(f'{O}/pools/today_cap3.npz', allow_pickle=True)
    yr = dict(zip(pool['ids'].astype(int).tolist(), pool['year'].astype(float).tolist()))
    tid = pid(Tz['paths'])
    assert len(tid) == 2048 and len(set(tid.tolist())) == 2048 and all(i in yr for i in tid.tolist())
    P, T = reference(Pz), reference(Tz, np.array([yr[i] for i in tid.tolist()]))
    assert not np.isnan(T['year']).any()
    P['id'], T['id'] = pid(Pz['paths']), tid
    return P, T, Pz


def without(ref, wid):
    """ref minus the row of work id wid (the start work); thresholds stay the reference's own."""
    keep = ref['id'] != wid
    if keep.all():
        return ref
    out = dict(ref, id=ref['id'][keep], artist=ref['artist'][keep], year=None if ref['year'] is None else ref['year'][keep])
    for o in OBS:
        out[o] = dict(ref[o], E=ref[o]['E'][keep])
    return out


def gate_of(z):
    """The run's own --fidelity-max, from its stored flags; one of the known values or the cell stops."""
    g = re.findall(r"'fidelity_max': ([0-9.]+)", str(z['flags']))
    assert len(g) == 1, 'the stored flags carry no single fidelity_max'
    g = float(g[0])
    assert g in GATES, f'fidelity_max {g} is not one of the known gates {GATES}'
    return g


def replay(z, Pr, live=False):
    """The loop's admission rule on stored rows, 15-NN scores taken at round start (live=True: against everything admitted so far); returns (admitted incl. start row 0, scores (2, n))."""
    it, n = z['iteration'], len(z['iteration'])
    E = {o: unit(z[f'emb_{o}']) for o in OBS}
    D = {o: chord(E[o] @ Pr[o]['E'].T) for o in OBS}
    feas = (z['coherence'] >= 0.0) & (z['fidelity_clip'] <= gate_of(z))    # --gate-threshold 0, --fidelity-max from the run's own flags
    bar = np.array([Pr[o]['bar'] for o in OBS])
    arch, adm, rawlog = [0], np.zeros(n, bool), np.full((2, n), np.nan)
    knn = lambda o, rows: np.sort(np.concatenate([D[o][rows], chord(E[o][rows] @ E[o][arch].T)], 1), 1)[:, :K].mean(1)  # noqa: E731
    for r in range(int(it.max()) + 1):
        rows = np.flatnonzero(it == r)
        stale = np.array([knn(o, rows) for o in OBS])
        rawlog[:, rows] = stale
        for k, j in enumerate(rows):
            raw = np.array([knn(o, np.array([j]))[0] for o in OBS]) if live else stale[:, k]
            if feas[j] and r >= WARMUP and (raw > bar).all() and chord(E[DINO][arch] @ E[DINO][j]).min() > DELTA:
                adm[j] = True
                arch.append(j)
    return adm, rawlog


def spacing(dino, idx):
    """DH-spacing: pictures of idx (in order) kept when DINOv3 L2 to the start (row 0) and to every kept one is above DELTA."""
    kept = [0]
    for j in idx:
        if chord(dino[kept] @ dino[j]).min() > DELTA:
            kept.append(j)
    return len(kept) - 1


def qfun(O):
    """Score: cosine of a vector's off-mean part with the pre-1900 shift direction, pre-1900-mean component removed."""
    d = unit(json.load(open(f'{os.path.dirname(O)}/newaxis/k32/newaxis.json'))['cutoffs']['1900']['directions']['shift'])
    a = unit(np.load(f'{O}/pools/starts_pre1900.npz', allow_pickle=True)['mean_pre'])
    dp = unit(d - (d @ a) * a)
    return lambda x: (unit(x) @ dp) / np.sqrt(np.clip(1 - (unit(x) @ a) ** 2, 1e-9, None))


def load_support():
    """Runs on_support.py's statements before its run loop in a namespace of its own; returns it and the p99 chord of P + P2 renders to the nearest dated work, source's artist out."""
    raw = open(f'{CODE}/on_support.py', 'rb').read()
    tree = ast.parse(raw.decode())
    tree.body = [n for n in tree.body if n.end_lineno < 85]
    ns = {'__name__': 'on_support'}
    exec(compile(tree, 'on_support.py', 'exec'), ns)  # noqa: S102
    base = {n: float(np.nanpercentile(np.concatenate([ns['res'][f'P_artistout__{n}__all'], ns['res'][f'P2_artistout__{n}__all']]), 99)) for n in OBS}
    print('on_support.py baseline (P + P2 renders, source artist left out) p99: ' + ', '.join(f'{n} {base[n]:.6f}' for n in OBS))
    assert (np.asarray(ns['raw_ids']) == np.load(IDS, allow_pickle=True)['ids'].astype(str)).all(), 'the dinov3 and clip observer files are not in one row order'
    return ns, base


def support(sup, r, sel):
    """Chord to the nearest dated real work (DINOv3, SigLIP) of the selected pictures, start work and its artist left out as in on_support.py's artistout rows."""
    ns, base = sup
    rows = 1 + np.flatnonzero(sel)
    if not len(rows):
        return np.zeros((2, 0))
    z = r['z']
    start_row = ns['row_of'].get(ns['raw_ids'][int(z['start_idx'])], -1)
    assert start_row >= 0, f"{r['label']}: start work not found in the corpus"
    mask = ns['rows_of_artist'].get(ns['artist'][start_row], [start_row])
    d = ns['nearest']({n: z[f'emb_{n}'][rows] for n in OBS}, [mask] * len(rows))
    return np.array([d[f'{n}__all'] for n in OBS])


def safety(paths, cache, path=None, compute=False):
    """S: True where the decoder's safety checker flags the saved 8-bit PNG; the checker runs only with compute=True, otherwise flags come from the cache."""
    key = lambda p: f'{p}|{os.stat(p).st_size}|{os.stat(p).st_mtime_ns}'   # the cache is keyed by the file, not the name: a rerun into the same folder gets new flags
    todo = [p for p in dict.fromkeys(map(str, paths)) if key(p) not in cache]
    assert compute or not todo, f'{len(todo)} pictures have no safety flag in the cache (run `controls_evaluate.py safety` first), e.g. {todo[0]}'
    if todo:
        import torch
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        from PIL import Image
        from transformers import CLIPImageProcessor
        chk, proc = StableDiffusionSafetyChecker.from_pretrained(SAFETY).eval(), CLIPImageProcessor.from_pretrained(SAFETY)
        for i in range(0, len(todo), 16):
            ims = [Image.open(p).convert('RGB') for p in todo[i:i + 16]]
            with torch.no_grad():
                _, has = chk(clip_input=proc(ims, return_tensors='pt').pixel_values, images=[np.asarray(im, np.float32) / 255 for im in ims])
            cache.update(zip(map(key, todo[i:i + 16]), (bool(h) for h in has)))
            if path and (i // 16) % 20 == 0:
                json.dump(cache, open(path, 'w'))
        if path:
            json.dump(cache, open(path, 'w'))
    return np.array([cache[key(str(p))] for p in paths], bool)


def quality(r, s, sup, cache, cpath):
    """QH = DH that pass S and U. S is read for the evaluator-admitted pictures (a superset of DH); None when no checker cache is given."""
    DH, base = s['DH'], sup[1]
    idx = np.flatnonzero(DH)
    d = support(sup, r, DH)
    U = (d[0] <= base[OBS[0]]) & (d[1] <= base[OBS[1]])
    Ur = (d[0] <= 0.78) & (d[1] <= 0.81)                              # the rounded values, printed beside
    paths = np.array([str(p) for p in r['z']['paths']])[1:]
    flag = None if cache is None else safety(paths[idx], cache, cpath)
    if cache is not None:
        s['ev_flag'] = int(safety(paths[np.flatnonzero(r['ev'])], cache, cpath).sum())   # flagged among all evaluator-admitted pictures
    ok = U if flag is None else U & ~flag
    s.update(U=U, Ur=Ur, flag=flag, d=d, base=(base[OBS[0]], base[OBS[1]]), QH=ok, nU=int(U.sum()), nUr=int(Ur.sum()), nS=None if flag is None else int((~flag).sum()), nQH=int(ok.sum()),
             qrows=1 + idx[ok])
    s['share'] = s['nQH'] / s['nDH'] if s['nDH'] else float('nan')
    s['nSD'] = s['nDH'] if s['nS'] is None else s['nS']                  # distinct hits that pass S: printed only
    return s


def share_fails(qv, dv, qx, dx):
    """V's pooled share qv/dv must be at least X's qx/dx minus 0.05, compared in integers; void when sum DH_X = 0."""
    if dx == 0:
        return False
    return 20 * qx > dx if dv == 0 else 20 * qv * dx < 20 * qx * dv - dv * dx


def decide(av, ax, sh=None):
    """Verdict over the five pairs; av, ax = V's and X's counts per pair; sh = pooled (QH_V, DH_V, QH_X, DH_X) for Q's share rule (None for Y)."""
    sv, sx, ahead = sum(av), sum(ax), sum(v > x for v, x in zip(av, ax))
    if len(av) != 5:
        return 'CANNOT TELL (not five valid pairs)'
    if (5 - ahead) >= 2 or (sx > 0 and sv < sx) or (sh is not None and share_fails(*sh)):
        return 'DOES NOT BEAT'
    return 'BEATS' if ahead == 5 and (sv >= 10 if sx == 0 else sv / sx >= 1.25) else 'CANNOT TELL'


class InvalidCell(Exception):
    pass


def loader(log):
    """Which CLIP loader the job used: quickgelu (quickgelu line in the .out, no QuickGELU warning in the .err), gelu (the warning, no such line) or unknown."""
    if not log or not os.path.exists(log):
        return 'unknown'
    err = os.path.splitext(log)[0] + '.err'
    out_ok = any('CLIP loader: open_clip ViT-L-14-quickgelu' in ln for ln in open(log, errors='replace'))
    warn = os.path.exists(err) and any('QuickGELU mismatch' in ln for ln in open(err, errors='replace'))
    if out_ok and not warn and os.path.exists(err):
        return 'quickgelu'
    return 'gelu' if warn and not out_ok else 'unknown'


def kat(log):
    """Checks the job log for the known-answer line (8 cosines at or above 0.999, stated minimum equal to theirs) and returns (reasons, summary)."""
    if not log or not os.path.exists(log):
        return ['no job log'], 'no job log'
    out_t = open(log, errors='replace').read()
    bad, found = [], re.findall(r'CLIP known-answer test \(.*?\): 8 reference repaints, cosine with the stored exact readings ((?:[0-9.]+ ){7}[0-9.]+); min ([0-9.]+)', out_t)
    if not found:
        bad.append('no known-answer line with 8 cosines in the .out')
    for cs, mn in found:
        c = [float(t) for t in cs.split()]
        if len(c) != 8 or min(c) < KAT_FLOOR or abs(min(c) - float(mn)) > 1e-5:
            bad.append(f'known-answer cosines not all at or above {KAT_FLOOR} ({cs}; stated min {mn})')
    return bad, ('known-answer: ' + '; '.join(f'8 cosines, min {mn}' for _, mn in found) if found else 'known-answer: no line')


def tag_role(fl, d):
    """The run's own tag from its stored flags, and its role from the pattern qgc_(ra|pin|c2)_sN: (tag, role in {V, C1, C2} or None, N or None)."""
    t = re.search(r"'tag': '([^']*)'", fl)
    t = t.group(1) if t else None
    mm = re.fullmatch(r'qgc_(ra|pin|c2)_s([0-9]+)', t or '')
    return t, (ROLE_OF[mm.group(1)] if mm else None), (int(mm.group(2)) if mm else None)


def card(log):
    if not log or not os.path.exists(log):
        return 'unknown', None, '?'
    head, err = open(log, errors='replace').read(6000), os.path.splitext(log)[0] + '.err'
    host, gpu = re.search(r'host=([\w.-]+)', head), re.search(r'^NVIDIA ([\w ]+?),', head, re.M)
    tcl = sum('init.tcl' in s for s in open(err, errors='replace')) if os.path.exists(err) else None   # None: no .err to check
    g = gpu.group(1) if gpu else '?'
    return (f"{g} {host.group(1) if host else '?'}" if (host or gpu) else 'unknown'), tcl, g


def startids(O):
    pools = np.load(f'{O}/pools/starts_pre1900.npz', allow_pickle=True)
    return pools['rows_full'], pools['ids'].astype(np.int64), np.load(IDS, allow_pickle=True)['ids'].astype(np.int64)


def load(spec, P, T, ids, q, conf=False):
    label, rest = spec.split('=', 1)
    d, exact, log = (rest.split(',') + ['-', '-'])[:3]
    z = np.load(f'{d}/all_candidates.npz', allow_pickle=True)
    m = z['iteration'] >= 0
    assert m.sum() == 1920 and z['iteration'].max() == 59 and not m[0] and m[1:].all(), f'{label}: not 60 x 32 rows after the start'
    assert np.all(np.diff(z['iteration'].astype(np.int64) * 1000 + z['index'].astype(np.int64)) > 0), f'{label}: rows are not in loop order'
    fl = str(z['flags'])
    assert all(t in fl for t in ("'admit_q99': True", "'delta': 0.575", "'warmup': 1", "'gate_threshold': 0.0", "'band': None")), f'{label}: admission flags differ from the loop rule the replay assumes; the replay is not valid for this run'
    gate = gate_of(z)    # each run is replayed and judged by its own gate; gates are never mixed
    assert ((z['coherence'] >= 0.0) & (z['fidelity_clip'] <= gate)).tolist() == z['feasible'].tolist(), f'{label}: replay feasibility differs from the run own feasible flags'
    j = np.flatnonzero(ids[0] == int(z['start_idx']))      # the start work's WikiArt id is starts_pre1900.npz `ids` at its seedrow index
    assert len(j) == 1, f'{label}: start_idx is not one row of starts_pre1900.rows_full'
    wid = int(ids[1][j[0]])
    assert wid == int(ids[2][int(z['start_idx'])]), f'{label}: starts_pre1900 ids and the observer file ids disagree for the start work'
    log = None if log == '-' else log
    crd, ldr, in_ref = card(log), loader(log), (int((P['id'] == wid).sum()), int((T['id'] == wid).sum()))
    tag, role, cellN = tag_role(fl, d)
    kbad, ktxt = kat(log)
    rc, ex = None, None
    if exact != '-':
        ex = np.load(exact, allow_pickle=True)
        assert len(z['paths']) == len(ex['paths']) and all(str(a) == str(b) for a, b in zip(z['paths'], ex['paths'])), label
        cs = (unit(z['render_clip']) * unit(ex['emb_clip-vit-l-14'])).sum(1)[m]   # the loop's own CLIP reading against the exact one, same pictures
        rc = (float(np.median(cs)), float(cs.min()), int((cs < 0.999).sum()), int((cs < 0.9995).sum()))   # the refusal below reads the median; the minimum and the two counts are printed only
    if conf:
        bad = []
        if crd[0] == 'unknown':
            bad.append('card unknown (job log missing)')
        if crd[1] != 0:
            bad.append('init.tcl check not clean (' + ('no .err file' if crd[1] is None else f'{crd[1]} init.tcl lines') + ')')
        if in_ref != (0, 0):
            bad.append('the start work is a row of P or T')
        if wid not in CONF_STARTS:
            bad.append(f'start work id {wid} is not one of the five starts')
        if rc is None:
            bad.append('no exact embedding file')
        if CONF_REQUIRE_QUICKGELU and ldr != 'quickgelu':
            bad.append(f'CLIP loader is {ldr}, not quickgelu')
        if gate != CONF_GATE:
            bad.append(f'fidelity gate {gate} is not {CONF_GATE}')
        if ldr == 'quickgelu' and rc is not None and rc[0] < 0.999:
            bad.append(f'a quickgelu run whose own CLIP reading differs from the exact one (median cosine {rc[0]:.4f} < 0.999)')
        bad += kbad
        if "'reanchor': 4" not in fl:
            bad.append('flags lack reanchor 4')
        if role is None:
            bad.append(f"tag {tag!r} is not qgc_ra_sN, qgc_pin_sN or qgc_c2_sN")
        else:
            if cellN not in CONF_START or CONF_START[cellN] != wid or f"'seed': {cellN}," not in fl:
                bad.append(f'tag {tag} (cell {cellN}) does not match start work {wid} and the run seed')
            if not os.path.basename(os.path.normpath(d)).startswith(tag + '_'):
                bad.append(f'run folder {os.path.basename(os.path.normpath(d))} does not carry the tag {tag}')
            rf, sp = "'random_fitness': True" in fl, re.search(r"'sigma_path': '[^']+'", fl) is not None
            if role == 'V' and (rf or sp):
                bad.append('a V cell with random_fitness or sigma_path in its flags')
            if role in ('C1', 'C2') and not rf:
                bad.append(f'a {role} cell without random_fitness in its flags')
            if role == 'C1' and not sp:
                bad.append('a C1 cell without sigma_path in its flags')
        if bad:
            raise InvalidCell('; '.join(bad))
    Pr, Tr = without(P, wid), without(T, wid)
    emb = {f'emb_{o}': z[f'emb_{o}'][m] for o in OBS}
    sP, sT = score(Pr, emb), score(Tr, emb)
    r = dict(label=label, z=z, wid=wid, in_ref=in_ref, F=far(sP, Pr), notT=~far(sT, Pr),
             later=np.mean([(Tr['year'][sT[o][1]] >= 1900).mean(1) for o in OBS], axis=0), m=np.min([sP[o][0] / Pr[o]['bar'] for o in OBS], axis=0),
             ok_loop=z['fidelity_clip'][m] <= gate, ok_exact=None, card=crd, gate=gate, loader=ldr, rc=rc, tag=tag, role=role, kat=ktxt,
             hash=hashlib.md5(z[f'emb_{DINO}'][0].astype(np.float32).tobytes()).hexdigest(), q_loop=np.full(1920, np.nan))   # D needs the exact CLIP reading; the loop's render_clip is not it, so LOOP gives no D
    r['C'] = r['notT'] & (r['later'] > 0.5)
    r['ev'] = replay(z, P)[0][m]   # the loop's own rule, start row left in P as the loop had it; F and C above drop it
    r['dino'] = unit(z[f'emb_{DINO}'])
    if ex is not None:
        r['ok_exact'] = chord((unit(z['style_embs']) * unit(ex['emb_clip-vit-l-14'])).sum(1))[m] <= FID
        r['q_exact'] = q(ex['emb_clip-vit-l-14'][m])
    assert Pr is not P or not (r['ev'] & ~r['F']).any(), f'{label}: an evaluator-admitted picture is not far from P alone'
    return r


def stats(r, mode):
    ok = r['ok_exact'] if mode == 'exact' else r['ok_loop']
    qv = r['q_exact'] if mode == 'exact' else r['q_loop']
    H, DF = r['F'] & ok & r['C'], r['ev'] & ok
    DH = DF & r['C']
    own = r['z']['admitted'][r['z']['iteration'] >= 0]
    # spacing-only count over F alone, start picture first
    return dict(mode=mode, n_far=int(r['F'].sum()), n_faithful=int(ok.sum()), n_H=int(H.sum()), HR=H.sum() / 1920, DF=DF, DH=DH, nDF=int(DF.sum()), nDH=int(DH.sum()),
                ev=int(r['ev'].sum()), own=int(own.sum()), own_hits=int((own & r['F'] & ok & r['notT']).sum()), spF=spacing(r['dino'], 1 + np.flatnonzero(r['F'])),
                own_same=bool((r['ev'] == own.astype(bool)).all()), qmed=float(np.median(qv[DH])) if DH.any() else float('nan'), q=qv)


def mlabel(r, mode):
    """EXACT, or LOOP split by the loader that read the run's pictures."""
    return 'EXACT' if mode == 'exact' else {'quickgelu': 'LOOP (quickgelu)', 'gelu': 'LOOP (gelu)'}.get(r['loader'], 'LOOP (loader unknown)')


nf = lambda x, f='.4f': 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, f)  # noqa: E731


def qline(s, who):
    return (f"{who}: DH {s['nDH']}, pass S {s['nS'] if s['nS'] is not None else 'not checked'} (flagged {s['nDH'] - s['nS'] if s['nS'] is not None else 'n/a'}), "
            f"inside the repainted works' own range of distance to a real work (U, DINOv3 and SigLIP both) {s['nU']} [DINOv3 {int((s['d'][0] <= s['base'][0]).sum())}, SigLIP {int((s['d'][1] <= s['base'][1]).sum())}; "
            f"at the rounded 0.78 / 0.81: {s['nUr']}], quality hits QH {s['nQH']}{'' if s['nS'] is not None else ' (safety checker not applied)'}, quality share QH / DH {nf(s['share'], '.3f')} (decisive); share QH / (DH that pass S) {nf(s['nQH'] / s['nSD'] if s['nSD'] else None, '.3f')} (of {s['nSD']}) [not decisive], quality count without the safety filter {s['nU']} (never used for a verdict)")


def check_pair(v, x):
    """Refuses a pair unless start picture and work, card type, CLIP loader (known) and fidelity gate agree."""
    bad = []
    if v['hash'] != x['hash']:
        bad.append('start pictures differ (DINOv3 hash)')
    if v['wid'] != x['wid']:
        bad.append('start work ids differ')
    for r in (v, x):
        if r['card'][0] == 'unknown':
            bad.append(f"{r['label']}: card unknown (job log missing)")
        if r['card'][1] != 0:
            bad.append(f"{r['label']}: init.tcl check not clean ({'no .err file' if r['card'][1] is None else str(r['card'][1]) + ' init.tcl lines'})")
        if r['loader'] == 'unknown':
            bad.append(f"{r['label']}: CLIP loader unknown (job log lacks the loader line and the QuickGELU warning)")
    if v['card'][2] != x['card'][2]:
        bad.append(f"cards differ ({v['card'][2]} against {x['card'][2]}): never paired")
    if v['loader'] != x['loader']:
        bad.append(f"CLIP loaders differ ({v['label']} {v['loader']}, {x['label']} {x['loader']}): never paired")
    if v['gate'] != x['gate']:
        bad.append(f"fidelity gates differ ({v['gate']} against {x['gate']}): never paired")
    if bad:
        raise InvalidCell('; '.join(bad))


def pair(v, x, sv, sx):
    m = np.concatenate([v['m'][sv['DF']], x['m'][sx['DF']]])
    e1, e2 = np.percentile(m, [100 / 3, 200 / 3]) if len(m) else (np.nan, np.nan)
    tert = lambda r, s: (r['m'][s['DF']] > e1).astype(int) + (r['m'][s['DF']] > e2).astype(int)  # noqa: E731
    tv, tx, Cv, Cx = tert(v, sv), tert(x, sx), v['C'][sv['DF']], x['C'][sx['DF']]
    rows, num, den = [], 0.0, 0
    for t in range(3):
        nv, nx = int((tv == t).sum()), int((tx == t).sum())
        if min(nv, nx) < 5:
            rows.append(f'     tertile {t}: V {nv}, X {nx} pictures: dropped')
            continue
        dif = Cv[tv == t].mean() - Cx[tx == t].mean()
        num, den = num + (nv + nx) * dif, den + nv + nx
        rows.append(f'     tertile {t}: V {nv} covered {Cv[tv == t].mean():.3f}, X {nx} covered {Cx[tx == t].mean():.3f}, diff {dif:+.3f}, weight {nv + nx}')
    sh = lambda c: f'{c.mean():.3f} of {len(c)}' if len(c) else 'n/a'  # noqa: E731
    Dtxt = (f"{sv['qmed'] - sx['qmed']:+.4f}" if not np.isnan(sv['qmed'] - sx['qmed']) else 'n/a, not reported: ' + ('no exact CLIP reading (LOOP)' if sv['mode'] != 'exact' else
            'no distinct hit in ' + ('V' if not sv['nDH'] else 'X') + ': counts as not positive'))
    Etxt = f'{num / den:+.4f}' if den else 'no tertile left: counts as not positive'
    print(f"PAIR {v['label']} (V) against {x['label']} (X), measure {mlabel(v, sv['mode'])}{'' if sv['mode'] == 'exact' else ' fidelity'}; start hash same: True; cards {v['card'][0]} / {x['card'][0]}; CLIP loader {v['loader']} / {x['loader']}; fidelity gate {v['gate']} / {x['gate']}\n"
          f"   DH{'' if sv['mode'] == 'exact' else ' (' + mlabel(v, sv['mode']) + ')'} {sv['nDH']} vs {sx['nDH']} | DF {sv['nDF']} vs {sx['nDF']} | n_H {sv['n_H']} vs {sx['n_H']} | HR {sv['HR']:.4f} vs {sx['HR']:.4f}\n"
          f"   E {Etxt} (pooled m tertile edges {nf(float(e1))}, {nf(float(e2))}; unmatched covered share V {sh(Cv)}, X {sh(Cx)}; ceiling 25 of 28 real far later works)\n"
          + '\n'.join(rows) + f"\n   {qline(sv, 'V')}\n   {qline(sx, 'X')}\n   D median q over DH: V {nf(sv['qmed'], '+.4f')} (n {sv['nDH']}), X {nf(sx['qmed'], '+.4f')} (n {sx['nDH']}), difference {Dtxt} (scale: real 1900-1969 works +0.197)")


def selftest(a, P):
    """The replay on the full P reproduces every stored admitted flag picture for picture; exits 1 on any difference."""
    print('SELFTEST: replay of the loop admission rule against the stored admitted flags (rows of the 1,920 pictures), full P as the loop ran')
    fails = []
    for spec in a.runs:
        lab = spec.split('=')[0]
        z = np.load(spec.split('=', 1)[1].split(',')[0] + '/all_candidates.npz', allow_pickle=True)
        m = z['iteration'] >= 0
        adm, raw = replay(z, P)
        live, _ = replay(z, P, True)
        st, feas = z['admitted'][m].astype(bool), (z['fidelity_clip'] <= gate_of(z)) & (z['coherence'] >= 0)
        assert np.isfinite(raw[:, m]).all()
        rawd = max(np.abs(raw[i, m] - z[f'raw_{o}'][m]).max() for i, o in enumerate(OBS))
        first = np.flatnonzero(adm[m] != st)
        fm = int((feas[m] != z['feasible'][m]).sum())
        print(f"  {lab:12s} gate {gate_of(z)} | stored admitted {int(st.sum()):3d} | replay (round-start score, as the loop) {int(adm[m].sum()):3d}, mismatching pictures {len(first)}"
              f" | picture-by-picture insertion {int(live[m].sum()):3d}, mismatching {int((live[m] != st).sum())} | max |score - stored raw| {rawd:.2e} | feasible mismatches {fm}"
              + (f" | first mismatch row {first[0] + 1} (iteration {z['iteration'][m][first[0]]}, slot {z['index'][m][first[0]]})" if len(first) else ''))
        if len(first) or fm or rawd > 1e-5:
            fails.append(lab)
    print('SELFTEST ' + ('FAILED for ' + ', '.join(fails) if fails else 'PASSED: every stored admitted flag reproduced, same pictures'))
    if fails:
        sys.exit(1)


def main(a):
    O, conf = a.root, not a.dev
    labs = [t.split('=')[0] for t in a.runs]
    if len(set(labs)) != len(labs):
        raise SystemExit(f'duplicate run labels: {sorted({l for l in labs if labs.count(l) > 1})}')
    P, T, Pz = references(O)
    ids, q = startids(O), qfun(O)
    if a.mode == 'selftest':
        return selftest(a, P)
    cache = json.load(open(a.safety_cache)) if a.safety_cache and os.path.exists(a.safety_cache) else ({} if a.safety_cache else None)
    if a.mode == 'safety':
        for name, f in ([] if a.runs else (('P: the 1900 reference renders', 'push_emb.npz'), ('later works 1900-1969', 'pos_emb.npz'), ('calibration works', 'push2_emb.npz'))):
            fl = safety(np.load(f'{O}/{f}', allow_pickle=True)['paths'], cache, a.safety_cache, True)
            print(f'SAFETY flag rate, {name}: {int(fl.sum())} of {len(fl)} ({fl.mean():.1%})', flush=True)
        for spec in a.runs:   # every evaluator-admitted picture of each run (a superset of its DH, in either fidelity mode)
            r = load(spec, P, T, ids, q)   # conf=False: the safety step reads pictures of any run
            fl = safety(np.array([str(p) for p in r['z']['paths']])[1:][r['ev']], cache, a.safety_cache, True)
            print(f"SAFETY {r['label']}: {int(fl.sum())} of {len(fl)} evaluator-admitted pictures flagged", flush=True)
        return
    print(f"mode: {'strict (every refusal applies)' if conf else 'DEV (the refusals of strict mode are relaxed)'}")
    chk = [float(np.median(q(Pz['vec']))), float(np.median(q(Pz['emb_clip-vit-l-14']))), float(np.median(q(np.load(f'{O}/pos_emb.npz', allow_pickle=True)['vec'])))]
    print('median q: reference requests %+.3f, their repainted pictures %+.3f, real 1900-1969 works %+.3f' % tuple(chk))
    print('header: the admission replay (DH, DF) runs on the full P, start row left in, as the loop ran and as the self-test does; '
          'F, C and m are scored against P and T with the start work removed.')
    sup, memo = load_support(), {}

    def S(r, mode):
        if (r['label'], mode) not in memo:
            memo[(r['label'], mode)] = quality(r, stats(r, mode), sup, cache, a.safety_cache)
        return memo[(r['label'], mode)]
    if cache is None:
        print('safety checker not applied (no --safety-cache): QH = DH that pass U; the best possible verdict is cannot tell')
    runs = {}
    for sp in a.runs:   # one bad cell is reported as invalid and does not stop the others
        try:
            runs[sp.split('=')[0]] = load(sp, P, T, ids, q, conf)
        except InvalidCell as e:
            print(f"INVALID cell {sp.split('=')[0]}: {e}; not computed")
        except AssertionError as e:   # a KeyError or any other exception is a bug or a file-layout fault and stops the run
            print(f"INVALID cell {sp.split('=')[0]}: {e!r}; not read, to be rerun")
    # a cell whose every pair is refused prints no counts and goes to no dump; a cell in no listed pair (a lone development run) is shown
    plist = list(a.pair) + [t for sp_ in a.verdict for t in sp_.split('=', 1)[1].split(',')]

    def pair_ok(p):
        v_, x_ = (runs.get(k) for k in p.split(':'))
        if v_ is None or x_ is None:
            return False
        try:
            check_pair(v_, x_)
            return True
        except InvalidCell:
            return False
    shown = {lab for lab in runs if not any(lab in p.split(':') for p in plist) or any(lab in p.split(':') and pair_ok(p) for p in plist)}
    for lab in runs:
        if lab not in shown:
            print(f'RUN {lab}: every pair this cell is in was refused; no counts printed')
    dump = {}
    for lab, r in ((k, v) for k, v in runs.items() if k in shown):
        print(f"RUN {lab}: start work id {r['wid']} (rows in P, T: {r['in_ref']}), start DINOv3 md5 {r['hash']}, card {r['card'][0]} (init.tcl lines {r['card'][1]}); "
              f"tag {r['tag']} (role {r['role']}); fidelity gate {r['gate']} ({GATE_NAME[r['gate']]}), CLIP loader {r['loader']}; {r['kat']}; "
              + ('own CLIP reading against the exact one: no exact file' if r['rc'] is None else f"cosine of the run's own CLIP reading with the exact one: median {r['rc'][0]:.4f}, minimum {r['rc'][1]:.4f}, pictures under 0.999: {r['rc'][2]}, under 0.9995: {r['rc'][3]}")
              + "; admission replay on the full P as the loop ran")
        for mode in (['exact'] if r['ok_exact'] is not None else []) + ['loop']:
            s = S(r, mode)
            print(f"   {mlabel(r, mode):5s} n_far {s['n_far']} n_faithful {s['n_faithful']} n_H {s['n_H']} HR {s['HR']:.4f} | evaluator-admitted {s['ev']} DF {s['nDF']} DH{'' if mode == 'exact' else ' (' + mlabel(r, mode) + ')'} {s['nDH']} "
                  f"(spacing-only count, NOT DH: over far {s['spF']}) | run's own admitted {s['own']} ({'same pictures as' if s['own_same'] else 'differs from'} the evaluator's), "
                  f"admitted hits (no condition ii) {s['own_hits']} | median q over DH {nf(s['qmed'], '+.4f')}"
                  + ('' if s.get('ev_flag') is None else ' | S flagged among the %d evaluator-admitted pictures: %d' % (s['ev'], s['ev_flag']))
                  + f"\n      {qline(s, 'DH')}")
            dump[f'{lab}:{mode}'] = dict(DF_m=r['m'][s['DF']].tolist(), DF_C=r['C'][s['DF']].astype(int).tolist(), DH_q=s['q'][s['DH']].tolist())

    structural = []   # strict-mode problems that make the overall line "cannot tell"

    def usable(p):
        v, x = (runs.get(k) for k in p.split(':'))
        if v is None or x is None:
            raise InvalidCell('a cell of the pair was not read')
        check_pair(v, x)
        return v, x, ('exact' if v['ok_exact'] is not None and x['ok_exact'] is not None else 'loop')
    for p in a.pair:
        try:
            v, x, mode = usable(p)
        except InvalidCell as e:
            print(f'INVALID pair {p}: {e}; no number printed')
            continue
        pair(v, x, S(v, mode), S(x, mode))
    final = {}
    for spec in a.verdict:
        name, prs = spec.split('=', 1)
        pl = [t.split(':') for t in prs.split(',')]
        if conf:
            why = []
            if len(pl) != 5:
                why.append(f'{len(pl)} pairs listed, not five')
            if len({t[0] for t in pl}) != len(pl) or len({t[1] for t in pl}) != len(pl):
                why.append('a run label is repeated among the pairs')
            wids = {runs[t[0]]['wid'] for t in pl if t[0] in runs}
            if wids != CONF_STARTS or sum(t[0] in runs for t in pl) != 5:
                why.append(f'the V arms\' start works {sorted(wids)} are not the five starts {sorted(CONF_STARTS)} (or a V cell was not computed)')
            if name in ('C1', 'C2'):
                bad_role = [t[0] for t in pl if t[0] in runs and runs[t[0]]['role'] != 'V'] + [t[1] for t in pl if t[1] in runs and runs[t[1]]['role'] != name]
                if bad_role:
                    why.append(f'cells {bad_role} do not carry the roles of their tags (V = qgc_ra_, {name} = qgc_{"pin" if name == "C1" else "c2"}_)')
            if why:
                structural.append(f'{name}: ' + '; '.join(why))
                print(f"INVALID verdict {name}: {'; '.join(why)}; no VERDICT line is printed for it")
                continue
        ss = []
        for p in prs.split(','):
            try:
                v, x, mode = usable(p)
            except InvalidCell as e:
                print(f'INVALID pair {p} in {name}: {e}; the verdict cannot have five valid pairs')
                continue
            ss.append((S(v, mode), S(x, mode)))
        col = lambda k, i: [t[i][k] for t in ss]  # noqa: E731
        tot = lambda k, i: sum(col(k, i))  # noqa: E731
        shr = lambda k, i, den='nDH': (tot(k, i) / tot(den, i) if tot(den, i) else None)  # noqa: E731  pooled over the pairs; None when the arm has no hit
        alt = lambda k, i: (tot(k, i) / tot('nSD', i) if tot('nSD', i) else None)  # noqa: E731  printed only: QH / distinct hits that pass S
        fl = lambda i: (sum(t[i]['nDH'] - t[i]['nSD'] for t in ss), sum(t[i]['nDH'] for t in ss), sum(t[i].get('ev_flag') or 0 for t in ss), sum(t[i]['ev'] for t in ss))  # noqa: E731
        y = decide(col('nDH', 0), col('nDH', 1))
        qa = decide(col('nQH', 0), col('nQH', 1), (tot('nQH', 0), tot('nDH', 0), tot('nQH', 1), tot('nDH', 1)))
        if cache is None and qa == 'BEATS':
            qa = 'CANNOT TELL (safety checker not applied)'
        modes = sorted({mlabel(runs[k], 'exact' if runs[k]['ok_exact'] is not None else 'loop') for p in prs.split(',') for k in p.split(':') if k in runs})
        print(f"VERDICT [{'strict' if conf else 'DEV'}; measure {'/'.join(modes)}] vs {name} over {len(ss)} valid pairs: Y (DH) {y} [DH V {col('nDH', 0)} X {col('nDH', 1)}]")
        print(f"   Q (QH) on all quality hits: {qa} [QH V {col('nQH', 0)} X {col('nQH', 1)}; share QH / DH (decisive), pooled: V {nf(shr('nQH', 0), '.3f')} X {nf(shr('nQH', 1), '.3f')}]")
        print(f"   share QH / (DH that pass S): V {nf(alt('nQH', 0), '.3f')} (of {tot('nSD', 0)}) X {nf(alt('nQH', 1), '.3f')} (of {tot('nSD', 1)}) [not decisive]; quality count without the safety filter (U alone, never used for a verdict): V {col('nU', 0)} X {col('nU', 1)}")
        print(f"   Safety flags, V: {fl(0)[0]} of {fl(0)[1]} distinct hits, {fl(0)[2]} of {fl(0)[3]} evaluator-admitted pictures; X: {fl(1)[0]} of {fl(1)[1]} distinct hits, {fl(1)[2]} of {fl(1)[3]} evaluator-admitted pictures" + ('' if cache is not None else ' (no checker cache: not applied)'))
        final[name] = (y, qa)
    if final or structural:
        vs = [v for t in final.values() for v in t]
        names = [sp_.split('=', 1)[0] for sp_ in a.verdict]
        if conf and sorted(names) != ['C1', 'C2']:
            structural.append(f'verdicts given {sorted(names)}, not exactly C1 and C2: a control was not evaluated')
        print('OVERALL: ' + (f"cannot tell: {'; '.join(structural)}" if structural else 'the search beats the controls' if all(v == 'BEATS' for v in vs) else 'does not beat' if any(v.startswith('DOES NOT') for v in vs) else 'cannot tell') + f'  ({final})')
    if a.dump:
        json.dump(dump, open(a.dump, 'w'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['selftest', 'table', 'safety'])
    p.add_argument('--safety-cache')
    p.add_argument('--dev', action='store_true', help='relax the refusals of strict mode (exact files, CLIP loader and gate, the five starts, C1 and C2 structure)')
    p.add_argument('--verdict', action='append', default=[])
    p.add_argument('root')
    p.add_argument('--pair', action='append', default=[])
    p.add_argument('--dump')
    p.add_argument('runs', nargs='*')
    main(p.parse_intermixed_args())
