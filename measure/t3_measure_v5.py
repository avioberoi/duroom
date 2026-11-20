"""Place measure (CPU): scales, guards and the verdict on a set of cells.
Usage: python t3_measure_v5.py scales|guards|measure --out OUT [--scales F --guards F --freeze F --cells "V:tags;C1:tags;C2:tags" ...]   (writes OUT.txt and OUT.npz, never over an existing file)"""
import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
from scipy import sparse

B = os.environ['DUROOM_OUT']
O, OE, RETRO = f'{B}/sphere', os.environ['DUROOM_EMB'], os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
T1 = f'{B}/tests12/t1/run_headline_{{d}}/{{o}}_{{f}}'
OBS, AIM, CLIP = ('dinov3-l', 'siglip-so400m'), 'aimv2-huge', 'clip-vit-l-14'
FAITH, FAITH_T, DUP = 0.7785, 0.8872, 0.99
Q, MIN_ROUNDS, QBLOCK, SEED, NDRAW = 5, 2, 5, 20261114, 200
FAR_T = (1.044886, 0.957734)
LS, MS, JS = (95, 99), (2, 3, 4, 5), (1, 15, 50)
CELL = dict(L=99, m=3, r='r_15', strict=True)
REAL = dict(rounds=60, per_round=32)
R0_REQGAP_MAX = 1e-2   # r_0 needs one request per C2 round; the largest within-round request chord must stay under this (real corpus only)
QGT = {'V': [f'qgt_ra_s{n}' for n in range(11, 16)], 'C1': [f'qgt_pin_s{n}' for n in range(11, 16)], 'C2': [f'qgt_c2_s{n}' for n in range(11, 16)]}
QGC_C2 = [f'qgc_c2_s{n}' for n in range(6, 11)]
LINES = []


def say(s=''):
    print(s, flush=True)
    LINES.append(str(s))


def md5(f):
    return hashlib.md5(open(f, 'rb').read()).hexdigest()


def unit(X):
    X = np.asarray(X, np.float32)
    nr = np.linalg.norm(X, axis=-1, keepdims=True)
    assert np.isfinite(X).all() and (nr > 0).all(), 'an embedding row is not finite or is zero'
    return X / nr


def chord(c):
    return np.sqrt(np.clip(2 - 2 * np.asarray(c, np.float64), 0, None))


# ---------------------------------------------------------------- corpus and repaints
def finish_corpus(c):
    c['row_of'] = {i: r for r, i in enumerate(c['ids'])}
    c['by_artist'] = {}
    for r, a in enumerate(c['artist']):
        if a:
            c['by_artist'].setdefault(a, []).append(r)
    c['by_artist'] = {a: np.array(v) for a, v in c['by_artist'].items()}
    c['dated'] = np.isfinite(c['year']) & (c['year'] > 0)
    return c


def load_corpus(spec):
    """real = data.py's 172,171 filtered rows (as on_support.py loads them); otherwise a fixture npz with the same fields."""
    if spec == 'real':
        sys.path.insert(0, RETRO)
        import data  # noqa: E402
        raw = np.load(f'{OE}/dinov3-l.npz', allow_pickle=True)['ids'].astype(str)
        # a run's start_idx is a row of the CLIP corpus file (the search picks it there); it is resolved through DINOv3's id order, so the orders must be one
        for n in OBS + (CLIP,):
            assert np.array_equal(np.load(f'{OE}/{n}.npz', allow_pickle=True)['ids'].astype(str), raw), f'{n}: corpus id order differs from dinov3-l'
        return finish_corpus(dict(E={n: unit(data.lens(f'{OE}/{n}.npz')) for n in OBS}, ids=np.asarray(data.ids).astype(str), raw_ids=raw,
                                  artist=np.asarray(data.labels['artist']).astype(str), year=np.asarray(data.year, np.float64),
                                  title=np.asarray(data.title).astype(str), style=np.asarray(data.labels['style']).astype(str)))
    z = np.load(spec, allow_pickle=True)
    return finish_corpus(dict(E={n: unit(z[f'emb_{n}']) for n in OBS}, **{k: z[k].astype(str) for k in ('ids', 'raw_ids', 'artist', 'title', 'style')},
                              year=z['year'].astype(np.float64)))


REPS = (('T', 'push_today_emb.npz'), ('P', 'push_emb.npz'), ('P2', 'push2_emb.npz'), ('pos', 'pos_emb.npz'))


def load_reps(root, C):
    S = {}
    for name, f in REPS:
        z = np.load(f'{root}/{f}', allow_pickle=True)
        ids = [re.search(r'push_(\d+)\.png$', str(p)).group(1) for p in z['paths']]
        S[name] = dict(src=np.array([C['row_of'][i] for i in ids]), E={n: unit(z[f'emb_{n}']) for n in OBS}, aim=unit(z[f'emb_{AIM}']),
                       req=unit(z['vec']), clip=unit(z[f'emb_{CLIP}']), paths=np.asarray(z['paths']).astype(str))
        S[name]['artist'] = C['artist'][S[name]['src']]
    # One row per work in the AIMv2 index, T's reading where a work has two: a second reading of the same work would be a T repaint's own twin.
    keep = {}
    for name, _ in REPS:
        keep[name] = np.array([s not in keep.get('_seen', set()) for s in S[name]['src']])
        keep.setdefault('_seen', set()).update(S[name]['src'].tolist())
    keep.pop('_seen')
    idx = dict(aim=np.concatenate([S[n]['aim'][keep[n]] for n, _ in REPS]), src=np.concatenate([S[n]['src'][keep[n]] for n, _ in REPS]))
    assert len(np.unique(idx['src'])) == len(idx['src'])
    return S, idx


def dup_rows(C, rows, chunk=256):
    """{row: other corpus rows with cosine > DUP to it in DINOv3 or SigLIP}"""
    rows = np.unique(np.asarray([r for r in rows if r >= 0], int))
    out = {int(r): set() for r in rows}
    for n in OBS:
        for i in range(0, len(rows), chunk):
            S = C['E'][n][rows[i:i + chunk]] @ C['E'][n].T
            for j, r in enumerate(rows[i:i + chunk]):
                out[int(r)].update(int(h) for h in np.flatnonzero(S[j] > DUP) if h != r)
    return {k: np.array(sorted(v), int) for k, v in out.items()}


# ---------------------------------------------------------------- nearest real work, nearest repaint
def nearest(Qs, Cs, masks, cols=None, k=1, chunk=512):
    """{obs: (n, k) ascending chords} from each query row to its k nearest rows of Cs[obs], after removing masks[i] and rows outside cols; every nearest-work question goes through it."""
    keep = np.arange(len(next(iter(Cs.values())))) if cols is None else np.flatnonzero(cols)
    pos = np.full(len(next(iter(Cs.values()))), -1)
    pos[keep] = np.arange(len(keep))
    out = {}
    for n in Qs:
        X, Cn = unit(Qs[n]), Cs[n][keep]
        res = np.empty((len(X), k))
        for i in range(0, len(X), chunk):
            S = X[i:i + chunk] @ Cn.T
            for j, rows in enumerate(masks[i:i + chunk]):
                if len(rows):
                    p = pos[np.asarray(rows, int)]
                    S[j, p[p >= 0]] = -2.0
            top = S.max(1, keepdims=True) if k == 1 else -np.partition(-S, k - 1, axis=1)[:, :k]
            res[i:i + chunk] = np.sort(chord(top), axis=1)
        out[n] = res
    return out


class Ctx:
    """the index a picture is measured against: corpus rows (cols), the AIMv2 repaint index (index rows ix_cols), the mask rule."""
    def __init__(self, C, idx, dups, cols=None, ix_cols=None, artist_mask=False):
        self.C, self.idx, self.dups, self.cols, self.ix_cols, self.artist_mask = C, idx, dups, cols, ix_cols, artist_mask
        self._m = {}

    def rows(self, s):
        """corpus rows masked for a picture whose source (or start) work is row s: the work and its near-duplicates (its artist too if artist_mask)"""
        if s < 0:
            return np.zeros(0, int)
        if s not in self._m:
            base = self.C['by_artist'].get(self.C['artist'][s], np.array([s])) if self.artist_mask else np.array([s])
            self._m[s] = np.union1d(np.union1d(base, [s]), self.dups[s]).astype(int)
        return self._m[s]


def readings(pics, ctx, k=1):
    """e: chord to the nearest original per observer (e15: mean of the 15 nearest, when k = 15); a: AIMv2 chord to the nearest repaint; f: request to exact CLIP reading chord; source work, duplicates and their repaints masked."""
    src = np.asarray(pics['src'])
    uq = np.unique(src)
    mr = {s: ctx.rows(int(s)) for s in uq}
    d = nearest(pics['E'], ctx.C['E'], [mr[s] for s in src], ctx.cols, k)
    out = dict(e=np.stack([d[n][:, 0] for n in OBS], 1))
    if k > 1:
        out['e15'] = np.stack([d[n][:, :15].mean(1) for n in OBS], 1)
    if pics.get('aim') is not None:
        hr = {s: np.flatnonzero(np.isin(ctx.idx['src'], mr[s])) for s in uq}
        out['a'] = nearest({'a': pics['aim']}, {'a': ctx.idx['aim']}, [hr[s] for s in src], ctx.ix_cols)['a'][:, 0]
    if pics.get('req') is not None:
        out['f'] = chord((unit(pics['req']) * unit(pics['clip'])).sum(1))
    return out


def calibrate(e, a, L):
    med = np.median(e, axis=0)
    return dict(med=med, theta=float(np.percentile((e / med).min(1), L)), hbar=float(np.percentile(a, L)))


def eligible(rd, cal, faith=FAITH, use_f=True, ekey='e'):
    E = np.all(rd[ekey] > cal['theta'] * cal['med'], axis=1)
    H = rd['a'] > cal['hbar']
    F = rd['f'] <= faith if use_f else np.ones(len(E), bool)
    return E, H, F, E & H & F


def radius(X, artist, js=JS):
    """r_j per observer: median over the repaints of the chord to the j-th nearest repaint by another artist."""
    out = {}
    for n in OBS:
        D = chord(X[n] @ X[n].T)
        D[artist[:, None] == artist[None, :]] = np.inf
        D.sort(axis=1)
        for j in js:
            out[j] = out.get(j, []) + [float(np.median(D[:, j - 1]))]
    return {f'r_{j}': np.array(v) for j, v in out.items()}


# ---------------------------------------------------------------- places
def together(Xa, Xb, r, chunk=1024):
    """sparse (na, nb): chord <= r in both observers"""
    I, J = [np.zeros(0, int)], [np.zeros(0, int)]
    for i in range(0, len(Xa[OBS[0]]), chunk):
        ok = np.ones((len(Xa[OBS[0]][i:i + chunk]), len(Xb[OBS[0]])), bool)
        for o, n in enumerate(OBS):
            ok &= chord(Xa[n][i:i + chunk] @ Xb[n].T) <= r[o]
        a, b = np.nonzero(ok)
        I.append(a + i)
        J.append(b)
    I, J = np.concatenate(I), np.concatenate(J)
    return sparse.csr_matrix((np.ones(len(I)), (I, J)), shape=(len(Xa[OBS[0]]), len(Xb[OBS[0]])))


def greedy(Tm, unit_, rnd, row, q, min_rounds, m_min):
    """Place: repeatedly take the centre with the most supporting units (ties: more pictures, lower unit, lower row), record it and remove the pictures within r, while support >= m_min; places at a larger m are a prefix of the list."""
    n = Tm.shape[0]
    if n == 0:
        return []
    nu, nr = int(unit_.max()) + 1, int(rnd.max()) + 1
    U1 = sparse.csr_matrix((np.ones(n), (np.arange(n), unit_)), shape=(n, nu))
    UR = sparse.csr_matrix((np.ones(n), (np.arange(n), unit_ * nr + rnd)), shape=(n, nu * nr))
    Tm, U1, UR = Tm.tocsr(), U1.tocsr(), UR.tocsr()
    alive, places = np.ones(n, bool), []
    cand = np.flatnonzero(np.diff(Tm.indptr) >= q * m_min)
    while len(cand):
        ai = np.flatnonzero(alive)
        Ta = Tm[cand][:, ai].tocsr()
        cnt = (Ta @ U1[ai]).toarray()
        rounds = ((Ta @ UR[ai]).toarray() > 0).reshape(len(cand), nu, nr).sum(2)
        sup = (cnt >= q) & (rounds >= min_rounds)
        support, members = sup.sum(1), cnt.sum(1)
        k = np.lexsort((row[cand], unit_[cand], -members, -support))[0]
        if support[k] < m_min:
            break
        mem = ai[Ta[k].indices]
        places.append(dict(c=int(cand[k]), support=int(support[k]), units=np.flatnonzero(sup[k]), members=np.sort(mem)))
        alive[mem] = False
        cand = cand[alive[cand] & (members >= q * m_min)]
    assert all(a['support'] >= b['support'] for a, b in zip(places, places[1:]))
    return places


def unit_counts(Tm, unit_other, nunits):
    return (Tm @ sparse.csr_matrix((np.ones(len(unit_other)), (np.arange(len(unit_other)), unit_other)), shape=(len(unit_other), nunits))).toarray()


def centre_info(arms, name, el, r, acc_X, nunits):
    """For each eligible picture of arm `name` as a centre: starts of each other arm with >= QBLOCK pictures within r, and whether any ACC picture is within r."""
    Xc = {n: arms[name]['X'][n][el] for n in OBS}
    blk = {o: (unit_counts(together(Xc, arms[o]['X'], r), arms[o]['unit'], nunits) >= QBLOCK).sum(1) for o in arms if o != name}
    acc = np.asarray(together(Xc, acc_X, r).sum(1)).ravel() > 0 if len(acc_X[OBS[0]]) else np.zeros(len(el), bool)
    return dict(blk=blk, acc=acc)


def arm_pairs(arm, r):
    el = np.flatnonzero(arm['elig'])
    Xe = {n: arm['X'][n][el] for n in OBS}
    return el, together(Xe, Xe, r).tocsr()


def places_of(arm, el, Tm, sub=None, q=Q, min_rounds=MIN_ROUNDS, m_min=2):
    """greedy places among the arm's eligible pictures el (or the positions sub of el); centres and members returned as positions in el"""
    s = np.arange(len(el)) if sub is None else np.asarray(sub, int)
    g = el[s]
    pl = greedy(Tm[s][:, s], arm['unit'][g], arm['rnd'][g], arm['row'][g], q, min_rounds, m_min)
    for p in pl:
        p['c'], p['members'] = int(s[p['c']]), s[p['members']]
    return pl


def exclusive(info, c, m, strict):
    if info['acc'][c]:
        return False
    return all((b[c] == 0) if strict else (b[c] < m) for b in info['blk'].values())


def excl_count(places, info, m, strict):
    return sum(1 for p in places if p['support'] >= m and exclusive(info, p['c'], m, strict))


def thin(el_u, other_u, reading, rng):
    """Equal numbers per unit; A: per start the larger arm is thinned to the other's count; B: only the arm with more eligible pictures in total is thinned, start by start."""
    tot_s, tot_o = sum(len(v) for v in el_u), sum(len(v) for v in other_u)
    out_s, out_o = [], []
    for a, b in zip(el_u, other_u):
        k = min(len(a), len(b))
        ts = reading == 'A' or tot_s > tot_o
        to = reading == 'A' or tot_o > tot_s
        out_s.append(np.sort(rng.choice(a, k, replace=False)) if ts and len(a) > k else a)
        out_o.append(np.sort(rng.choice(b, k, replace=False)) if to and len(b) > k else b)
    return np.sort(np.concatenate(out_s)).astype(int), np.sort(np.concatenate(out_o)).astype(int)


# ---------------------------------------------------------------- cells
def arm_flags_check(z, tag, arm, vtag, layout):
    """Hard check that the run's stored flags match its arm (V: search; C1: random score with V's step sizes; C2: random score with the 1e-5 step file)."""
    assert 'flags' in z.files, f'{tag}: run file stores no flags'
    f = ast.literal_eval(str(z['flags']))
    assert f.get('tag') == tag, f"{tag}: run flags name the tag {f.get('tag')}"
    assert layout is None or (f.get('maxiter'), f.get('popsize')) == tuple(layout), f'{tag}: maxiter / popsize {f.get("maxiter")} / {f.get("popsize")}, layout {layout}'
    rf, sp = f.get('random_fitness'), f.get('sigma_path')
    if arm == 'V':
        assert rf is False and sp is None, f'{tag}: not the search (random_fitness {rf}, sigma_path {sp})'
    elif arm == 'C1':
        assert rf is True and sp is not None and f'/{vtag}_committee' in str(sp), f'{tag}: not the random-score loop of {vtag} (random_fitness {rf}, sigma_path {sp})'
    elif arm == 'C2':
        assert rf is True and sp is not None and os.path.basename(str(sp)) == 'c2_sigma_1e-5.npz', f'{tag}: not the no-step loop (random_fitness {rf}, sigma_path {sp})'
    return f


def load_cell(tag, run_tmpl, exact_tmpl, rows, C, safety_tmpl=None, layout=None, arm=None, vtag=None):
    """One run: requests and rounds from the run file, the four readings from the exact file; validity and layout are asserted, counts are not read."""
    rf, ef = run_tmpl.format(tag=tag), exact_tmpl.format(tag=tag)
    bad = [f'{tag}: missing {f}' for f in (rf, ef) if not os.path.exists(f)]
    if bad:
        return dict(tag=tag, valid=False, why=bad, lacks_exact=not os.path.exists(ef), lacks_safety=True)
    z, ex = np.load(rf, allow_pickle=True), np.load(ef, allow_pickle=True)
    why = []
    it = np.asarray(z['iteration'])
    if len(it) != rows or it[0] != -1 or (it[1:] < 0).any():
        why.append(f'{tag}: {len(it)} rows / row 0 iteration {it[0]}')
    if [str(p) for p in z['paths']] != [str(p) for p in ex['paths']]:
        why.append(f'{tag}: run and exact paths differ')
    lacks_exact = any(f'emb_{n}' not in ex.files for n in OBS + (AIM, CLIP))
    lacks_safety = True
    if safety_tmpl:
        sf = safety_tmpl.format(tag=tag)
        if os.path.exists(sf):
            keys = {k.rsplit('|', 2)[0] for k in json.load(open(sf))}
            lacks_safety = not all(str(p) in keys for p in z['paths'])
    if why or lacks_exact:
        return dict(tag=tag, valid=False, why=why + ([f'{tag}: exact file lacks an encoder'] if lacks_exact else []), lacks_exact=lacks_exact, lacks_safety=lacks_safety)
    paths = [str(p) for p in z['paths']]
    assert len(set(paths)) == len(paths), f'{tag}: repeated candidate paths'
    if layout is not None:
        R, P = layout
        assert rows == 1 + R * P and np.array_equal(it, np.concatenate([[-1], np.repeat(np.arange(R), P)])), f'{tag}: not {R} rounds of {P} with the start row first'
    for n in OBS + (AIM, CLIP):
        assert np.isfinite(ex[f'emb_{n}']).all() and (np.linalg.norm(ex[f'emb_{n}'], axis=1) > 0).all(), f'{tag}: exact {n} readings not finite / zero'
    flags = arm_flags_check(z, tag, arm, vtag, layout) if arm is not None else None
    start = C['row_of'][C['raw_ids'][int(z['start_idx'])]]
    s = slice(1, None)
    return dict(tag=tag, valid=True, why=[], lacks_exact=False, lacks_safety=lacks_safety, start=start, start_idx=int(z['start_idx']), paths=np.array(paths),
                seed=None if flags is None else flags.get('seed'),
                E={n: unit(ex[f'emb_{n}'][s]) for n in OBS}, aim=unit(ex[f'emb_{AIM}'][s]), req=unit(z['style_embs'][s]), clip=unit(ex[f'emb_{CLIP}'][s]),
                rnd=it[s].astype(int), row=np.arange(1, rows), E0={n: unit(ex[f'emb_{n}'][:1]) for n in OBS}, src=np.full(rows - 1, start),
                runE={n: unit(z[f'emb_{n}'][s]) for n in OBS} if all(f'emb_{n}' in z.files for n in OBS) else None)


def pool_arm(runs, rd, cal, faith=FAITH, ekey='e'):
    """one arm: all pictures of its runs (unit = start position), with the eligibility of each"""
    X = {n: np.concatenate([r['E'][n] for r in runs]) for n in OBS}
    u = np.concatenate([np.full(len(r['rnd']), i) for i, r in enumerate(runs)])
    el = np.concatenate([eligible(d, cal, faith, ekey=ekey)[3] for d in rd])
    return dict(X=X, unit=u, rnd=np.concatenate([r['rnd'] for r in runs]), row=np.concatenate([r['row'] for r in runs]), elig=el)


def evaluate(arms, r, acc_X, nunits, draws=NDRAW, q=Q, min_rounds=MIN_ROUNDS):
    """Raw places of each arm, centre info and the equal-number draws (readings A and B) against each loop; every m and blocking rule is read from these."""
    res = {}
    for a in arms:
        el, Tm = arm_pairs(arms[a], r)
        res[a] = dict(el=el, Tm=Tm, places=places_of(arms[a], el, Tm, q=q, min_rounds=min_rounds), info=centre_info(arms, a, el, r, acc_X, nunits))
    for loop in ('C1', 'C2'):
        byu = {a: [np.flatnonzero(arms[a]['unit'][res[a]['el']] == u) for u in range(nunits)] for a in ('V', loop)}
        for reading in ('A', 'B'):
            rng = np.random.default_rng(SEED)
            out = []
            for _ in range(draws):
                sv, so = thin(byu['V'], byu[loop], reading, rng)
                out.append((places_of(arms['V'], res['V']['el'], res['V']['Tm'], sv, q, min_rounds),
                            places_of(arms[loop], res[loop]['el'], res[loop]['Tm'], so, q, min_rounds)))
            res[f'{loop}_{reading}'] = out
    return res


def counts_at(res, m, strict):
    """per loop: (V's exclusive places, the loop's) raw and as the median over draws of readings A and B"""
    ex = {a: excl_count(res[a]['places'], res[a]['info'], m, strict) for a in ('V', 'C1', 'C2')}
    out = {}
    for loop in ('C1', 'C2'):
        out[loop] = {'raw': (ex['V'], ex[loop])}
        for reading in ('A', 'B'):
            d = res[f'{loop}_{reading}']
            v = [excl_count(pv, res['V']['info'], m, strict) for pv, _ in d]
            o = [excl_count(po, res[loop]['info'], m, strict) for _, po in d]
            out[loop][reading] = (float(np.median(v)), float(np.median(o)))
    return out


def verdict(valid_starts, lacks, m, strict, res, gyes, gno):
    """first match wins; returns (rule, verdict, note)"""
    if valid_starts < m or lacks:
        return 1, 'CANNOT TELL', f'{valid_starts} starts valid in all arms; cells lacking exact/AIMv2 or safety: {lacks}'
    V = res['V']
    Pm = [p for p in V['places'] if p['support'] >= m]
    nonacc = [p for p in Pm if not V['info']['acc'][p['c']]]
    if not nonacc:
        return (2, 'NO', 'no place' if not Pm else 'only accidents') if gyes else (2, 'CANNOT TELL', 'NO, but G-yes failed')
    reached = [p for p in nonacc if any(b[p['c']] >= m for b in V['info']['blk'].values())]
    if len(reached) == len(nonacc):
        return 3, 'REACHED, NOT THE SEARCH\'S', ', '.join(f"{o} from {V['info']['blk'][o][p['c']]}" for p in nonacc for o in V['info']['blk'])
    excl = [p for p in nonacc if exclusive(V['info'], p['c'], m, strict)]
    if not excl:
        return 4, 'CANNOT TELL', 'V has no exclusive place (blocked by a loop from 1 to m - 1 starts)'
    cnt = counts_at(res, m, strict)
    if any(all(lo >= v for v, lo in cnt[loop].values()) for loop in cnt):
        return 5, 'NOT MORE THAN THE LOOPS', str(cnt)
    if not all(all(v > lo for v, lo in cnt[loop].values()) for loop in cnt):
        return 6, 'CANNOT TELL', f'the counts disagree: {cnt}'
    if not gno:
        return 6, 'CANNOT TELL', 'G-no failed'
    return 6, 'CANNOT TELL', f'{len(excl)} exclusive V place(s) and V ahead in every count; this measure does not issue YES; counts {cnt}'


# ---------------------------------------------------------------- data that every mode shares
def load_common(a):
    C = load_corpus(a.corpus)
    S, idx = load_reps(a.root, C)
    say(f'corpus {len(C["ids"])} rows; repaints ' + ', '.join(f'{n} {len(S[n]["src"])}' for n in S) + f'; AIMv2 index {len(idx["src"])} works (one row per work, T first)')
    if a.corpus == 'real':
        shared = len(set(S['T']['src'].tolist()) & set(S['P']['src'].tolist()))
        say(f"AIMv2 index {len(idx['src'])}; {shared} works in both T and P; corpus id order one across {OBS + (CLIP,)}")
    return C, S, idx


def write(a, res):
    for f in (a.out + '.npz', a.out + '.txt'):
        assert not os.path.exists(f), f'{f} exists: refusing to overwrite'
    np.savez(a.out + '.npz', **res)
    open(a.out + '.txt', 'x').write('\n'.join(LINES) + '\n')
    say(f'wrote {a.out}.npz; {a.out}.txt')


def load_acc(a, need_aim):
    """ACC = 192 cone + 64 sphere random requests and 160 noise requests, known-answer rows dropped."""
    out = {'E': {n: [] for n in OBS}, 'aim': [], 'req': [], 'clip': [], 'kind': [], 'paths': []}
    for d, sub, aimf in ((a.acc_prior, 'prior', a.acc_aim_prior), (a.acc_noise, 'noise', a.acc_aim_noise)):
        em = np.load(f'{d}/emb.npz', allow_pickle=True)
        k = len([ln for ln in open(f'{d}/inputs/expected_kat.txt') if ln.strip()])
        man = np.load(f'{d}/{sub}/manifest.npz', allow_pickle=True)
        paths = [str(p) for p in em['paths'][k:]]
        assert paths == [str(p) for p in man['paths']], f'{d}: emb rows are not the manifest rows'
        for n in OBS:
            out['E'][n].append(unit(em[f'emb_{n}'][k:]))
        out['req'].append(unit(man['v']))
        out['clip'].append(unit(em[f'emb_{CLIP}'][k:]))
        out['kind'] += [f'{sub}_{t}' for t in np.asarray(man['tag']).astype(str)]
        out['paths'] += paths
        if need_aim:
            z = np.load(aimf, allow_pickle=True)
            pz = {str(p): i for i, p in enumerate(z['paths'])}
            assert all(p in pz for p in paths), f'{aimf}: not every ACC picture has an AIMv2 reading'
            out['aim'].append(unit(z[f'emb_{AIM}'][[pz[p] for p in paths]]))
    acc = dict(E={n: np.concatenate(v) for n, v in out['E'].items()}, req=np.concatenate(out['req']), clip=np.concatenate(out['clip']),
               aim=np.concatenate(out['aim']) if need_aim else None, kind=np.array(out['kind']), paths=np.array(out['paths']))
    acc['src'] = np.full(len(acc['kind']), -1)
    return acc


# ---------------------------------------------------------------- mode: scales
def r1_values(C, S, idx_src, fit_tmpl):
    """R1 per observer: median over regions of the median chord from the other member repaints to the region's medoid (regions as in t1core regions())."""
    reps = {n: [] for n in OBS}
    for name, _ in REPS:
        s = S[name]
        for n in OBS:
            reps[n].append((s['src'], s['E'][n]))
    out = {}
    for fitname, f in fit_tmpl.items():
        vals = []
        for n, d in zip(OBS, ('dinov3', 'siglip')):
            z = np.load(f.format(d=d, o=n), allow_pickle=True)
            m, Cn = (z['mean'], z['centres']) if 'mean' in z.files else (z['K32_mean'], z['K32_centres'])
            qid, qreg = (z['Q_ids'], z['Q_region']) if 'Q_region' in z.files else (z['Q_ids'], z['K32_Q_region'])
            src = np.concatenate([x[0] for x in reps[n]])
            X = np.concatenate([x[1] for x in reps[n]]).astype(np.float64)
            first = np.unique(src, return_index=True)[1]       # one repaint per work, T's reading first (REPS order)
            src, X = src[first], X[first]
            orig = C['E'][n][src].astype(np.float64)
            Yo = orig - m
            lab = np.argmax((Yo / np.linalg.norm(Yo, axis=1, keepdims=True)) @ Cn.T, axis=1)
            # the saved region assignment where it exists; the same argmax rule for the originals it never scored
            qmap = dict(zip(qid.astype(str).tolist(), qreg.tolist()))
            saved = np.array([C['ids'][s] in qmap for s in src])
            agree = int(sum(lab[i] == qmap[C['ids'][s]] for i, s in enumerate(src) if saved[i]))
            lab = np.array([qmap[C['ids'][s]] if saved[i] else lab[i] for i, s in enumerate(src)])
            assert agree >= 0.99 * saved.sum(), f'{fitname} {n}: the argmax rule agrees with the saved assignment for only {agree} of {saved.sum()}'
            Y = (X - m) / np.linalg.norm(X - m, axis=1, keepdims=True)
            med = []
            for k in range(len(Cn)):
                mem = np.flatnonzero(lab == k)
                if len(mem) < 2:
                    continue
                c = mem[np.argmax(Y[mem] @ Cn[k])]
                med.append(np.median(chord(X[mem[mem != c]] @ X[c])))
            vals.append(float(np.median(med)))
            say(f'R1 [{fitname}] {n}: {len(med)} regions with >= 2 repaints; {int(saved.sum())} of {len(src)} repaints have a saved region, '
                f'the argmax rule agrees for {agree}; R1 {vals[-1]:.4f}')
        out[fitname] = np.array(vals)
    return out


def mode_scales(a):
    C, S, idx = load_common(a)
    T = S['T']
    dups = dup_rows(C, T['src'])
    ctx = Ctx(C, idx, dups)
    rd = readings(dict(E=T['E'], aim=T['aim'], req=T['req'], clip=T['clip'], src=T['src']), ctx)
    res = {'T_e': rd['e'], 'T_a': rd['a'], 'T_f': rd['f'], 'T_src': T['src']}
    say(f"T: {len(T['src'])} repaints; sources with a near-duplicate (cosine > {DUP}, DINOv3 or SigLIP): {sum(len(v) > 0 for v in dups.values())}")
    for L in LS:
        cal = calibrate(rd['e'], rd['a'], L)
        E, H, F, el = eligible(rd, cal)
        res[f'theta_{L}'], res[f'hbar_{L}'], res['med'] = np.float64(cal['theta']), np.float64(cal['hbar']), cal['med']
        say(f"L {L}: med_o DINOv3 {cal['med'][0]:.6f}, SigLIP {cal['med'][1]:.6f}; theta_L {cal['theta']:.6f} (E bars {cal['theta'] * cal['med'][0]:.6f} / "
            f"{cal['theta'] * cal['med'][1]:.6f}); H bar {cal['hbar']:.6f}; T passing E {int(E.sum())}, H {int(H.sum())}, F {int(F.sum())}, eligible {int(el.sum())} of {len(E)}")
    rr = radius(T['E'], T['artist'])
    for k, v in rr.items():
        res[k] = v
    # r_0: two renders of one request in the same round of a closed C2 cell (32 seeds of one request, sigma 1e-5)
    pairs, reqgap = {n: [] for n in OBS}, 0.0
    for tag in a.r0_cells.split(','):
        cell = load_cell(tag, a.run_tmpl, a.exact_tmpl, a.rows, C, layout=layout_of(a), arm='C2')
        assert cell['valid'], cell['why']
        for rnd in np.unique(cell['rnd']):
            k = np.flatnonzero(cell['rnd'] == rnd)
            iu = np.triu_indices(len(k), 1)
            reqgap = max(reqgap, float(chord(cell['req'][k] @ cell['req'][k].T)[iu].max()))
            for n in OBS:
                pairs[n].append(chord(cell['E'][n][k] @ cell['E'][n][k].T)[iu])
    res['r_0'] = np.array([float(np.median(np.concatenate(pairs[n]))) for n in OBS])
    say(f"r_0 ({a.r0_cells}): DINOv3 {res['r_0'][0]:.4f}, SigLIP {res['r_0'][1]:.4f}; largest request chord inside a round {reqgap:.2e}")
    if a.corpus == 'real':
        assert reqgap < R0_REQGAP_MAX, f'largest request chord inside a C2 round {reqgap:.2e} >= {R0_REQGAP_MAX}: not one request per round'
    r1 = r1_values(C, S, idx['src'], {'all-works fit (H2_allfit_K32)': a.r1_fit, 'centres_radii fit (K32), printed only': a.r1_fit_past})
    res['R1'] = r1['all-works fit (H2_allfit_K32)']
    res['R1_pastA'] = r1['centres_radii fit (K32), printed only']
    for k in ('r_0', 'r_1', 'r_15', 'r_50', 'R1'):
        say(f"{k}: DINOv3 {res[k][0]:.4f}, SigLIP {res[k][1]:.4f}" + ('' if k == 'r_0' else ('  (at or below r_0 in an observer: NOT OFFERED)' if (res[k] <= res['r_0']).any() else '')))
    write(a, res)


def layout_of(a):
    return (a.rounds, a.per_round)


def load_scales(a):
    z = np.load(a.scales, allow_pickle=True)
    say(f'scales {a.scales}')
    sc = {k: z[k] for k in z.files}
    sc['cal'] = {L: dict(med=sc['med'], theta=float(sc[f'theta_{L}']), hbar=float(sc[f'hbar_{L}'])) for L in LS}
    return sc


def t_readings(C, S, idx, sc, k=1):
    T = S['T']
    ctx = Ctx(C, idx, dup_rows(C, T['src']))
    rd = readings(dict(E=T['E'], aim=T['aim'], req=T['req'], clip=T['clip'], src=T['src']), ctx, k)
    assert np.array_equal(sc['T_src'], T['src']), "the scales file's T source ids are not T's"
    assert np.allclose(rd['e'], sc['T_e'], atol=1e-6) and np.allclose(rd['a'], sc['T_a'], atol=1e-6), 'T readings differ from the scales file'
    return rd


# ---------------------------------------------------------------- mode: guards
def unit_places(pics, rd, cal, r, m, use_f=True, faith=FAITH, q=1, rnd=None):
    """places among pictures whose units are artists / works / pictures (q pictures, no rounds rule unless rnd is given)"""
    E, H, F, el = eligible(rd, cal, faith, use_f)
    k = np.flatnonzero(el)
    if not len(k):
        return [], k, (E, H, F, el)
    Xe = {n: pics['E'][n][k] for n in OBS}
    u = np.unique(pics['unit'][k], return_inverse=True)[1]
    rr = np.zeros(len(k), int) if rnd is None else rnd[k]
    pl = greedy(together(Xe, Xe, r), u, rr, k, q, 1 if rnd is None else MIN_ROUNDS, m)
    return pl, k, (E, H, F, el)


def place_lists(out, key, unit_all, pl, k):
    """Save a guard part's place list; centre and members are positions in the part's picture set."""
    ulab = np.unique(np.asarray(unit_all)[k]).astype(str) if len(k) else np.zeros(0, '<U1')
    out[f'{key}__n_places'] = np.int64(len(pl))
    out[f'{key}__eligible_pos'] = np.asarray(k, np.int64)
    for j, p in enumerate(pl):
        out[f'{key}__place{j}__centre'] = np.int64(k[p['c']])
        out[f'{key}__place{j}__support'] = np.int64(p['support'])
        out[f'{key}__place{j}__units'] = ulab[p['units']]
        out[f'{key}__place{j}__members'] = np.asarray(k[p['members']], np.int64)


def pic_arrays(out, key, rd, flags, unit_all, pos, rnd=None, row=None, F_real=None):
    """Save: per picture of a guard part: chords e (per observer), a, the gap, flags E, H, F, eligible, unit ids, position in the parent set, round"""
    E, H, F, el = flags
    out[f'{key}__e'], out[f'{key}__a'] = np.asarray(rd['e'], np.float64), np.asarray(rd['a'], np.float64)
    if 'f' in rd:
        out[f'{key}__gap'] = np.asarray(rd['f'], np.float64)
    out[f'{key}__E'], out[f'{key}__H'] = np.asarray(E, bool), np.asarray(H, bool)
    out[f'{key}__F'] = np.asarray(F if F_real is None else F_real, bool)
    out[f'{key}__eligible'] = np.asarray(el, bool)
    out[f'{key}__unit'] = np.asarray(unit_all).astype(str)
    out[f'{key}__pos'] = np.asarray(pos, np.int64)
    if rnd is not None:
        out[f'{key}__round'] = np.asarray(rnd, np.int64)
    if row is not None:
        out[f'{key}__row'] = np.asarray(row, np.int64)


def gyes_diagnostics(out, key, X, unit_all, flags, radii):
    """How many of G-yes's test repaints pass each condition, and whether the eligible ones sit near each other (printed only)."""
    E, H, F, el = flags
    k = np.flatnonzero(el)
    u = np.asarray(unit_all).astype(str)[k]
    say(f'   diagnostics, printed only ({key}): of {len(E)} test repaints: E {int(E.sum())}, H {int(H.sum())}, F {int(F.sum())}, E and H (before F) '
        f'{int((E & H).sum())}, F failures inside E and H {int((E & H & ~F).sum())}, eligible {len(k)}; distinct artists among the eligible {len(np.unique(u))}')
    out[f'{key}__diag_counts'] = np.array([len(E), E.sum(), H.sum(), F.sum(), (E & H).sum(), (E & H & ~F).sum(), len(k), len(np.unique(u))], np.int64)
    if not len(k):
        return
    same = u[:, None] == u[None, :]
    D = {n: chord(X[n][k] @ X[n][k].T) for n in OBS}
    nn = np.stack([np.where(same, np.inf, D[n]).min(1) for n in OBS], 1)
    out[f'{key}__diag_nn_other_artist'] = nn
    out[f'{key}__diag_unit'] = u
    for rk, r in radii.items():
        inb = (D[OBS[0]] <= r[0]) & (D[OBS[1]] <= r[1])
        nart = np.array([len(np.unique(u[inb[i]])) for i in range(len(k))], np.int64)
        pair = (inb & ~same).any(1)
        out[f'{key}__diag_artists_in_ball_{rk}'], out[f'{key}__diag_pair_within_{rk}'] = nart, pair
        say(f'      {rk} ({r[0]:.4f} / {r[1]:.4f}): largest number of distinct artists in an eligible centre\'s ball {int(nart.max())}; '
            f'eligible pictures with an eligible picture by another artist within r in both observers {int(pair.sum())} of {len(k)}')
    fin = np.isfinite(nn).all(1)
    if fin.any():
        say('      chord to the nearest eligible picture by another artist: ' + ', '.join(
            f'{n} min {nn[fin, o].min():.4f} median {np.median(nn[fin, o]):.4f}' for o, n in enumerate(OBS)))


def mode_guards(a):
    C, S, idx = load_common(a)
    sc = load_scales(a)
    L, m, r = CELL['L'], CELL['m'], sc[CELL['r']]
    assert CELL['r'] == 'r_15', 'G-yes and G-no (ii) recompute r_15 by name; the cell radius must be r_15'
    assert (r > sc['r_0']).all(), 'the cell radius is at or below r_0 (not offered)'
    cal = sc['cal'][L]
    acc = load_acc(a, need_aim=True)
    cells = [load_cell(t, a.run_tmpl, a.exact_tmpl, a.rows, C, layout=layout_of(a), arm='C2') for t in a.r0_cells.split(',')]
    assert all(c['valid'] for c in cells), [c['why'] for c in cells]
    assert len({c['start'] for c in cells}) == len(cells), 'the closed C2 cells do not have distinct starts'
    man, em = np.load(f'{a.chains}/manifest.npz', allow_pickle=True), np.load(f'{a.chains}/emb.npz', allow_pickle=True)
    assert [str(p) for p in man['paths']] == [str(p) for p in em['paths']]
    ch = dict(E={n: unit(em[f'emb_{n}']) for n in OBS}, aim=unit(em[f'emb_{AIM}']), req=unit(man['v']), clip=unit(em[f'emb_{CLIP}']),
              src=S['P']['src'][np.asarray(man['row'])], unit=np.asarray(man['tag']).astype(str))
    nz = np.load(a.raw_noise, allow_pickle=True)
    raw = dict(E={n: unit(nz[f'emb_{n}']) for n in OBS}, src=np.full(len(nz['paths']), -1), unit=np.arange(len(nz['paths'])))
    T = S['T']
    assert np.array_equal(sc['T_src'], T['src']), "the scales file's T source ids are not T's"
    res, ok, sv = {}, {}, {}
    # G-yes: index = originals dated before 1970; calibration on T's repaints of works before 1970; test = T's 1970-on repaints, artists as units.
    yr = C['year'][T['src']]
    early, late = np.flatnonzero(yr < 1970), np.flatnonzero(yr >= 1970)
    idx_early = np.isin(idx['src'], np.flatnonzero(C['dated'] & (C['year'] < 1970)))
    dups = dup_rows(C, T['src'])
    ctx = Ctx(C, idx, dups, cols=C['dated'] & (C['year'] < 1970), ix_cols=idx_early)
    pick = lambda s, ix: dict(E={n: s['E'][n][ix] for n in OBS}, aim=s['aim'][ix], req=s['req'][ix], clip=s['clip'][ix], src=s['src'][ix], unit=s['artist'][ix])  # noqa: E731
    rE, rL = readings(pick(T, early), ctx), readings(pick(T, late), ctx)
    calY = calibrate(rE['e'], rE['a'], L)
    radY = radius({n: T['E'][n][early] for n in OBS}, T['artist'][early])
    rY = radY['r_15']
    pl, k, (E, H, F, el) = unit_places(pick(T, late), rL, calY, rY, m)
    accY = [bool(together({n: T['E'][n][late[k[p['c']]]][None] for n in OBS}, acc['E'], rY).sum() > 0) for p in pl]
    nonacc = sum(1 for x in accY if not x)
    ok['G-yes'] = nonacc >= 1
    say(f'G-yes: index {int(ctx.cols.sum())} originals dated before 1970, AIMv2 index {int(idx_early.sum())} repaints; calibration {len(early)} T repaints '
        f'(theta {calY["theta"]:.4f}, H bar {calY["hbar"]:.4f}, r_15 {rY[0]:.4f} / {rY[1]:.4f}); test {len(late)} repaints of 1970-on works: '
        f'E {int(E.sum())}, H {int(H.sum())}, F {int(F.sum())}, eligible {int(el.sum())}; places (>= {m} artists) {len(pl)}, non-accident {nonacc}: ' + ('PASS' if ok['G-yes'] else 'FAIL'))
    for p in pl:
        say(f'   place: {p["support"]} artists, {len(p["members"])} pictures')
    pic_arrays(sv, 'gyes', rL, (E, H, F, el), T['artist'][late], late)
    place_lists(sv, 'gyes', T['artist'][late], pl, k)
    sv['gyes__place_accident'] = np.array(accY, bool)
    sv['gyes__paths'] = T['paths'][late]
    gyes_diagnostics(sv, 'gyes', {n: T['E'][n][late] for n in OBS}, T['artist'][late], (E, H, F, el), {'r_15': rY, 'r_50': radY['r_50'], 'R1': sc['R1']})
    o1 = nearest({n: C['E'][n][T['src'][late]] for n in OBS}, C['E'], [ctx.rows(int(s)) for s in T['src'][late]], ctx.cols)
    say('   printed only: chord from each 1970-on original to its nearest original before 1970 (original against original): '
        + ', '.join(f'{n} median {np.median(o1[n][:, 0]):.4f} p10 {np.percentile(o1[n][:, 0], 10):.4f} p90 {np.percentile(o1[n][:, 0], 90):.4f}' for n in OBS))
    full = Ctx(C, idx, dup_rows(C, np.concatenate([T['src'], S['P2']['src'], S['P']['src']])))
    # G-no (i): P2's 1,000 repaints against the full corpus at the cell's scales, artists as units
    P2 = S['P2']
    rP = readings(pick(P2, np.arange(len(P2['src']))), full)
    pl, k, (E, H, F, el) = unit_places(pick(P2, np.arange(len(P2['src']))), rP, cal, r, m)
    ok['G-no i'] = len(pl) == 0
    say(f'G-no (i) P2: E {int(E.sum())}, H {int(H.sum())}, F {int(F.sum())}, eligible {int(el.sum())} of {len(el)}; places {len(pl)}: ' + ('PASS' if ok['G-no i'] else 'FAIL'))
    pic_arrays(sv, 'gno1', rP, (E, H, F, el), P2['artist'], np.arange(len(P2['src'])))
    place_lists(sv, 'gno1', P2['artist'], pl, k)
    sv['gno1__paths'] = P2['paths']
    # G-no (ii): T split by artist; half 1 sets the bars (and a radius), half 2 is the test
    arts = np.unique(T['artist'])
    perm = np.random.default_rng(SEED).permutation(len(arts))
    h1 = np.isin(T['artist'], arts[perm[:len(arts) // 2]])
    rT = dict(e=sc['T_e'], a=sc['T_a'], f=sc['T_f'])
    sub = lambda d, ix: {kk: v[ix] for kk, v in d.items()}  # noqa: E731
    cal1 = calibrate(rT['e'][h1], rT['a'][h1], L)
    r1h = radius({n: T['E'][n][h1] for n in OBS}, T['artist'][h1])['r_15']
    t2i = np.flatnonzero(~h1)
    E2 = eligible(sub(rT, t2i), cal1)
    say(f'G-no (ii): {h1.sum()} half-1 repaints ({len(arts) // 2} artists), {len(t2i)} half-2; half-1 theta {cal1["theta"]:.4f}, H bar {cal1["hbar"]:.4f}, r_15 {r1h[0]:.4f} / {r1h[1]:.4f}; '
        f'held-out joint E rate {E2[0].mean():.4f}, E and H and F {E2[3].mean():.4f} (scale)')
    n2 = []
    for lab, rad in (('half-1 radius', r1h), ('full-T radius', r)):
        pl, k, fl = unit_places(pick(T, t2i), sub(rT, t2i), cal1, rad, m)
        n2.append(len(pl))
        say(f'   places with the {lab}: {len(pl)}')
        place_lists(sv, 'gno2_' + ('r1h' if lab.startswith('half') else 'rT'), T['artist'][t2i], pl, k)
    pic_arrays(sv, 'gno2', sub(rT, t2i), fl, T['artist'][t2i], t2i)
    sv['gno2__half1'] = h1
    ok['G-no ii'] = sum(n2) == 0
    say('G-no (ii): ' + ('PASS' if ok['G-no ii'] else 'FAIL'))
    say(f"   printed beside: G-no (ii) under the cell's radius alone (r_15 of full T): places {n2[1]}: " + ('PASS' if n2[1] == 0 else 'FAIL')
        + '; in force: a place under either radius fails it')
    # G-no (iii): ACC, each picture its own unit, E and H without F (the count with F beside)
    acc['unit'] = np.arange(len(acc['kind']))
    rA = readings(acc, full)
    pl, k, (E, H, F, el) = unit_places(acc, rA, cal, r, m, use_f=False)
    plF, kF, flF = unit_places(acc, rA, cal, r, m, use_f=True)
    ok['G-no iii'] = len(pl) == 0
    rR = readings(raw, full)
    ER = np.all(rR['e'] > cal['theta'] * cal['med'], axis=1)
    kR = np.flatnonzero(ER)
    plR = greedy(together({n: raw['E'][n][kR] for n in OBS}, {n: raw['E'][n][kR] for n in OBS}, r), np.arange(len(kR)), np.zeros(len(kR), int), kR, 1, 1, m) if len(kR) else []
    say(f'   probe, printed only: the {len(ER)} raw noise pictures (no AIMv2), two observers only: E {int(ER.sum())}; places on E alone (each picture its own unit) {len(plR)}')
    Freal = flF[2]
    pri = np.char.startswith(acc['kind'].astype(str), 'prior')
    say(f'G-no (iii) ACC {len(el)}: E {int(E.sum())}, H {int(H.sum())}, F {int(Freal.sum())} (random requests {int(Freal[pri].sum())} of {int(pri.sum())}, '
        f'noise requests {int(Freal[~pri].sum())} of {int((~pri).sum())}); eligible without F (E and H, in force) {int(el.sum())}; eligible with F {int(flF[3].sum())}; '
        f'places {len(pl)} (with F: {len(plF)}): ' + ('PASS' if ok['G-no iii'] else 'FAIL'))
    pic_arrays(sv, 'gno3', rA, (E, H, F, el), acc['unit'], np.arange(len(el)), F_real=Freal)
    sv['gno3__eligible_withF'], sv['gno3__kind'], sv['gno3__paths'] = np.asarray(flF[3], bool), acc['kind'].astype(str), acc['paths']
    place_lists(sv, 'gno3', acc['unit'], pl, k)
    place_lists(sv, 'gno3_withF', acc['unit'], plF, kF)
    # G-no (iv): the five closed qgc C2 cells (starts as units, q 5, >= 2 rounds) and chains_h200_qg (48 works as units, q 1)
    full2 = Ctx(C, idx, dup_rows(C, [c['start'] for c in cells] + S['P']['src'].tolist()))
    pc = dict(E={n: np.concatenate([c['E'][n] for c in cells]) for n in OBS}, aim=np.concatenate([c['aim'] for c in cells]), req=np.concatenate([c['req'] for c in cells]),
              clip=np.concatenate([c['clip'] for c in cells]), src=np.concatenate([c['src'] for c in cells]), unit=np.concatenate([np.full(len(c['rnd']), i) for i, c in enumerate(cells)]))
    rC = readings(pc, full2)
    rndC = np.concatenate([c['rnd'] for c in cells])
    pl, k, (E, H, F, el) = unit_places(pc, rC, cal, r, m, q=Q, rnd=rndC)
    say(f'G-no (iv) qgc C2 cells: E {int(E.sum())}, H {int(H.sum())}, F {int(F.sum())}, eligible {int(el.sum())} of {len(el)}; places {len(pl)}')
    pic_arrays(sv, 'gno4c2', rC, (E, H, F, el), pc['unit'], np.arange(len(el)), rnd=rndC, row=np.concatenate([c['row'] for c in cells]))
    place_lists(sv, 'gno4c2', pc['unit'], pl, k)
    sv['gno4c2__cell_tags'] = np.array([c['tag'] for c in cells])
    rH = readings(ch, full2)
    plh, kh, (E, H, F, el) = unit_places(ch, rH, cal, r, m)
    say(f'G-no (iv) chains: {len(np.unique(ch["unit"]))} works, {len(el)} pictures: E {int(E.sum())}, H {int(H.sum())}, F {int(F.sum())}, eligible {int(el.sum())}; places {len(plh)}')
    pic_arrays(sv, 'gno4ch', rH, (E, H, F, el), ch['unit'], np.arange(len(el)))
    place_lists(sv, 'gno4ch', ch['unit'], plh, kh)
    sv['gno4ch__paths'] = np.asarray(man['paths']).astype(str)
    ok['G-no iv'] = len(pl) == 0 and len(plh) == 0
    say('G-no (iv): ' + ('PASS' if ok['G-no iv'] else 'FAIL'))
    gno = all(v for kk, v in ok.items() if kk.startswith('G-no'))
    say(f'GUARDS: G-yes {"PASS" if ok["G-yes"] else "FAIL"}; G-no {"PASS" if gno else "FAIL"} ({ok})')
    res.update({'G_yes': np.bool_(ok['G-yes']), 'G_no': np.bool_(gno), **{f'ok_{kk}': np.bool_(v) for kk, v in ok.items()},
                'acc_e': rA['e'], 'acc_a': rA['a'], 'P2_e': rP['e'], 'P2_a': rP['a'], 'chains_e': rH['e'], 'chains_a': rH['a'], 'qgc_c2_e': rC['e'], 'qgc_c2_a': rC['a']})
    res.update(sv)   # per-picture arrays and place lists of every guard part (keys gyes__, gno1__, gno2__, gno3__, gno4c2__, gno4ch__)
    write(a, res)


# ---------------------------------------------------------------- mode: measure
def census(arms, runs, res, p, r, sc, S, C, acc, rd_all, Paxis, Pvec):
    V = res['V']
    ix = V['el'][p['members']]
    c = V['el'][p['c']]
    cX = {n: arms['V']['X'][n][[c]] for n in OBS}
    say(f"  centre: start {arms['V']['unit'][c]} row {arms['V']['row'][c]}; support {p['support']} starts {p['units'].tolist()}; members {len(ix)}; "
        f"rounds {sorted(set(arms['V']['rnd'][ix].tolist()))}")
    for o in arms:
        t = together(cX, arms[o]['X'], r).toarray()[0]
        say(f'  pictures of arm {o} within r of the centre (any eligibility), per start: {np.bincount(arms[o]["unit"][t > 0], minlength=len(runs[o])).tolist()}')
    sp = [chord(arms['V']['X'][n][ix] @ arms['V']['X'][n][c]).max() for n in OBS]
    say(f"  spread (largest chord centre to member) {sp[0]:.4f} / {sp[1]:.4f}; r {r[0]:.4f} / {r[1]:.4f}; R1 {sc['R1'][0]:.4f} / {sc['R1'][1]:.4f}; r_0 {sc['r_0'][0]:.4f} / {sc['r_0'][1]:.4f}")
    st = [runs['V'][u]['E0'] for u in p['units']]
    say('  chords between supporting starts\' start pictures: ' + '; '.join(f'{i}-{j} ' + '/'.join(f'{float(chord(st[i][n] @ st[j][n].T)[0, 0]):.3f}' for n in OBS)
                                                                    for i in range(len(st)) for j in range(i + 1, len(st))))
    req = np.concatenate([runs['V'][u]['req'] for u in range(len(runs['V']))])[ix]
    say(f'  median CLIP chord between members\' requests {np.median(chord(req @ req.T)[np.triu_indices(len(ix), 1)]):.4f}')
    for lab, Y in (('nearest T repaint', S['T']['E']), ('nearest ACC picture', acc['E']), ('nearest C1 picture', arms['C1']['X']), ('nearest C2 picture', arms['C2']['X'])):
        say(f'  {lab}: ' + ' / '.join(f'{chord(Y[n] @ cX[n][0]).min():.4f}' for n in OBS))
    eV, aV, fV = (np.concatenate([d[k] for d in rd_all['V']])[ix] for k in ('e', 'a', 'f'))
    for j, i in enumerate(ix):
        say(f"   member start {arms['V']['unit'][i]} row {arms['V']['row'][i]} round {arms['V']['rnd'][i]}: e {eV[j, 0]:.4f}/{eV[j, 1]:.4f} a {aV[j]:.4f} f {fV[j]:.4f} "
            f"(a) {float(req[j] @ Paxis):.4f} (b) {float((Pvec @ req[j]).max()):.4f}")
    for n in OBS:
        s = C['E'][n] @ cX[n][0]
        s[runs['V'][int(arms['V']['unit'][c])]['start']] = -2
        for t in np.argsort(-s)[:5]:
            say(f'   5 nearest real works [{n}]: id {C["ids"][t]} | {C["artist"][t]} | {C["year"][t]:.0f} | {C["title"][t][:60]} | {C["style"][t]} | chord {chord(s[t]):.4f}')


def verdict_with(valid_starts, lacks, m, strict, res, gyes, gno, readings):
    """Printed beside only: verdict() with the equal-numbers comparison restricted to `readings` (all three = verdict() itself)."""
    if valid_starts < m or lacks:
        return 1, 'CANNOT TELL', f'{valid_starts} starts valid in all arms; cells lacking exact/AIMv2 or safety: {lacks}'
    V = res['V']
    Pm = [p for p in V['places'] if p['support'] >= m]
    nonacc = [p for p in Pm if not V['info']['acc'][p['c']]]
    if not nonacc:
        return (2, 'NO', 'no place' if not Pm else 'only accidents') if gyes else (2, 'CANNOT TELL', 'NO, but G-yes failed')
    reached = [p for p in nonacc if any(b[p['c']] >= m for b in V['info']['blk'].values())]
    if len(reached) == len(nonacc):
        return 3, 'REACHED, NOT THE SEARCH\'S', ', '.join(f"{o} from {V['info']['blk'][o][p['c']]}" for p in nonacc for o in V['info']['blk'])
    excl = [p for p in nonacc if exclusive(V['info'], p['c'], m, strict)]
    if not excl:
        return 4, 'CANNOT TELL', 'V has no exclusive place (blocked by a loop from 1 to m - 1 starts)'
    cnt = {loop: {k: v for k, v in d.items() if k in readings} for loop, d in counts_at(res, m, strict).items()}
    if any(all(lo >= v for v, lo in cnt[loop].values()) for loop in cnt):
        return 5, 'NOT MORE THAN THE LOOPS', str(cnt)
    if not all(all(v > lo for v, lo in cnt[loop].values()) for loop in cnt):
        return 6, 'CANNOT TELL', f'the counts disagree: {cnt}'
    if not gno:
        return 6, 'CANNOT TELL', 'G-no failed'
    return 6, 'CANNOT TELL', f'{len(excl)} exclusive V place(s) and V ahead in every count; this measure does not issue YES; counts {cnt}'


READINGS = (('raw + A + B (in force)', ('raw', 'A', 'B')), ('raw + B', ('raw', 'B')), ('raw alone', ('raw',)))


def freeze_check(a):
    """measure starts only if the marker's scales_md5 and guards_md5 equal those of the scales and guard files; checked before any other file is opened."""
    if not a.freeze or not os.path.isfile(a.freeze):
        raise SystemExit(f'measure refuses to start: no marker ({a.freeze})')
    for f in (a.scales, a.guards):
        if not f or not os.path.isfile(f):
            raise SystemExit(f'measure refuses to start: {f} missing')
    want = {}
    for ln in open(a.freeze):
        if '=' in ln:
            k, v = ln.strip().split('=', 1)
            want[k.strip()] = v.strip()
    got = {'scales_md5': md5(a.scales), 'guards_md5': md5(a.guards)}
    for k, v in got.items():
        if want.get(k) != v:
            raise SystemExit(f'measure refuses to start: {k} is {v}, the marker {a.freeze} says {want.get(k)}')
    say(f'marker {a.freeze}: scales and guard files equal the marker')


def per_start_counts(arms, res, loop, nunits):
    """Printed only: eligible pictures per start that raw, reading A and reading B leave (the sizes do not depend on the draw)"""
    byu = {x: [np.flatnonzero(arms[x]['unit'][res[x]['el']] == u) for u in range(nunits)] for x in ('V', loop)}
    out = {'raw': ([len(v) for v in byu['V']], [len(v) for v in byu[loop]])}
    for reading in ('A', 'B'):
        sv, so = thin(byu['V'], byu[loop], reading, np.random.default_rng(0))
        out[reading] = (np.bincount(arms['V']['unit'][res['V']['el'][sv]], minlength=nunits).tolist(),
                        np.bincount(arms[loop]['unit'][res[loop]['el'][so]], minlength=nunits).tolist())
    return out


def measure_record(out, arms, res, runs, rd_all, cal, m, strict):
    """Saved at the cell: per arm the picture chords, flags and ids, per place its centre, members, support, accident, blocking and exclusivity, and the counts raw / A / B per loop; nothing is recomputed."""
    for arm in arms:
        A_ = arms[arm]
        out[f'{arm}__tags'] = np.array([r_['tag'] for r_ in runs[arm]])
        out[f'{arm}__unit'], out[f'{arm}__round'], out[f'{arm}__row'] = (np.asarray(A_[k], np.int64) for k in ('unit', 'rnd', 'row'))
        out[f'{arm}__paths'] = np.concatenate([np.asarray(r_['paths'])[1:] for r_ in runs[arm]]) if runs[arm] else np.zeros(0, '<U1')
        fl = [eligible(d, cal) for d in rd_all[arm]]
        for i, nm in enumerate(('E', 'H', 'F')):
            out[f'{arm}__cell_{nm}'] = np.concatenate([f[i] for f in fl]) if fl else np.zeros(0, bool)
        for k, nm in (('e', 'e'), ('a', 'a'), ('f', 'gap')):
            out[f'{arm}__cell_{nm}'] = np.concatenate([d[k] for d in rd_all[arm]]) if rd_all[arm] else np.zeros((0, 2) if k == 'e' else 0)
        out[f'{arm}__eligible'] = A_['elig']
        assert np.array_equal(out[f'{arm}__eligible'], out[f'{arm}__cell_E'] & out[f'{arm}__cell_H'] & out[f'{arm}__cell_F'])
        el = res[arm]['el']
        out[f'{arm}__n_places'] = np.int64(len(res[arm]['places']))
        for k, p in enumerate(res[arm]['places']):
            pre, mem, c = f'place__{arm}__{k}__', el[p['members']], el[p['c']]
            out[pre + 'members_unit'], out[pre + 'members_row'] = A_['unit'][mem], A_['row'][mem]
            out[pre + 'members_round'] = A_['rnd'][mem]
            out[pre + 'support'] = np.int64(p['support'])
            out[pre + 'centre_unit'], out[pre + 'centre_row'] = np.int64(A_['unit'][c]), np.int64(A_['row'][c])
            out[pre + 'units'] = np.asarray(p['units'], np.int64)
            out[pre + 'accident'] = np.bool_(res[arm]['info']['acc'][p['c']])
            for o, b in res[arm]['info']['blk'].items():
                out[pre + f'blocking_{o}'] = np.int64(b[p['c']])
            out[pre + 'exclusive'] = np.bool_(p['support'] >= m and exclusive(res[arm]['info'], p['c'], m, strict))
    for loop, d in counts_at(res, m, strict).items():
        for reading, vl in d.items():
            out[f'cell_counts__{loop}__{reading}'] = np.array(vl, np.float64)


def provenance(a, t):
    """Refuses unless the cell's exact file md5 equals its record in its COMPLETE marker and the task.log has a card line."""
    f = a.exact_tmpl.format(tag=t)
    comp, log = f'{a.afterrun}/cells/{t}_h200/COMPLETE', f'{a.afterrun}/logs/{t}_h200/task.log'
    for g in (comp, log):
        if not os.path.isfile(g):
            raise SystemExit(f'measure refuses to start: {t}: {g} missing')
    rec = [ln.split(None, 1) for ln in open(comp) if ln.strip()]
    rec = [w[0] for w in rec if len(w) == 2 and os.path.normpath(w[1].strip().lstrip('*')) == os.path.normpath(f)]
    if len(rec) != 1:
        raise SystemExit(f'measure refuses to start: {t}: {comp} holds {len(rec)} records for {f}')
    now = md5(f)
    if now != rec[0]:
        raise SystemExit(f'measure refuses to start: {t}: exact file md5 {now} differs from the {rec[0]} its COMPLETE records')
    card = [ln.strip() for ln in open(log) if 'card line' in ln]
    if not card: raise SystemExit(f'measure refuses to start: {t}: {log} has no card line')
    return f'{t}: exact file {f} = COMPLETE; {card[-1]}'


def mode_measure(a):
    freeze_check(a)
    tags = {arm: v.split(',') for arm, v in (x.split(':') for x in a.cells.split(';'))}
    # preflight: every cell's run, exact and safety file exist before any cell array (or the corpus) is read; otherwise rule 1 and stop
    need = [(t, f) for arm in ('V', 'C1', 'C2') for t in tags[arm] for f in (a.run_tmpl.format(tag=t), a.exact_tmpl.format(tag=t),
                                                                            a.safety_tmpl.format(tag=t) if a.safety_tmpl else None)]
    missing = [f'{t}: {f}' for t, f in need if f is None or not os.path.isfile(f)]
    if missing:
        say(f'rule 1: CANNOT TELL (preflight: {len(missing)} of {len(need)} cell inputs missing: {missing}); nothing read, nothing computed')
        write(a, {'verdict_rule': np.int64(1), 'verdict': np.array('CANNOT TELL')})
        return
    # provenance preflight: each cell's COMPLETE record must match its exact file's md5, else measure refuses before any cell array is read
    for arm in ('V', 'C1', 'C2'):
        for t in tags[arm]:
            say(provenance(a, t))
    say('NOTE: the equal-numbers comparison in force uses three counts, raw, reading A (each start\'s '
        'larger arm thinned to the smaller) and reading B (only the arm with more eligible pictures in total thinned, start by start); V is ahead only if '
        'ahead in all three, a loop is "as many" only if so in all three, any split gives rule 6. This can withhold a YES and can also turn the '
        'rule 5 (NOT MORE THAN THE LOOPS) into CANNOT TELL. The side-by-side lines at the cell give the verdict under raw + B and raw alone.')
    say('NOTE: rule 1 fires whenever a listed cell lacks its run, its exact file with AIMv2 or its '
        'safety pass, whatever m; a start missing after a failed rerun gives CANNOT TELL unless it is left out of --cells.')
    say('NOTE: G-no (ii) in force fails on a place under either radius (half 1\'s r_15 or full T\'s r_15); the side-by-side lines '
        'also give the cell\'s radius alone. With G-yes failed the possible outcomes are CANNOT TELL, REACHED, NOT '
        'THE SEARCH\'S and NOT MORE THAN THE LOOPS; only NO is replaced by CANNOT TELL. A would-be YES prints as rule 6.')
    C, S, idx = load_common(a)
    sc = load_scales(a)
    g = np.load(a.guards, allow_pickle=True)
    say(f'guards {a.guards}: G-yes {bool(g["G_yes"])}, G-no {bool(g["G_no"])}')
    gyes, gno = bool(g['G_yes']), bool(g['G_no'])
    gno_cellr = (bool(g['ok_G-no i']) and bool(g['ok_G-no iii']) and bool(g['ok_G-no iv']) and int(g['gno2_rT__n_places']) == 0) if 'gno2_rT__n_places' in g.files else None
    say(f'G-no with G-no (ii) under the cell\'s radius alone (printed beside): {gno_cellr}')
    cells = {arm: [load_cell(t, a.run_tmpl, a.exact_tmpl, a.rows, C, a.safety_tmpl, layout=layout_of(a), arm=arm, vtag=tags['V'][i] if i < len(tags['V']) else None)
                   for i, t in enumerate(tags[arm])] for arm in ('V', 'C1', 'C2')}
    lacks = [c['tag'] for arm in cells for c in cells[arm] if c['lacks_exact'] or c['lacks_safety']]
    for arm in cells:
        for c in cells[arm]:
            say(f"{c['tag']}: valid {c['valid']}" + (f" ({'; '.join(c['why'])})" if c['why'] else '') + f"; exact with AIMv2 {not c['lacks_exact']}; safety pass {not c['lacks_safety']}")
    keep = [i for i in range(len(tags['V'])) if all(cells[arm][i]['valid'] for arm in cells)]
    for i in keep:
        assert len({cells[arm][i]['start_idx'] for arm in cells}) == 1, f'start {i}: the arms do not share a start'
        assert len({cells[arm][i]['seed'] for arm in cells}) == 1, f'start {i}: the arms do not share a seed'
    for arm in cells:
        assert len({cells[arm][i]['start_idx'] for i in keep}) == len(keep), f'{arm}: two positions share a start'
    runs = {arm: [cells[arm][i] for i in keep] for arm in cells}
    nunits = len(keep)
    if lacks or nunits < CELL['m']:
        say(f'rule 1: CANNOT TELL ({nunits} starts valid in all arms; cells lacking the exact file with AIMv2 or the safety pass: {lacks}); nothing computed')
        write(a, {'verdict_rule': np.int64(1), 'verdict': np.array('CANNOT TELL')})
        return
    dups = dup_rows(C, [r_['start'] for r_ in runs['V']] + S['T']['src'].tolist())
    acc = load_acc(a, need_aim=False)
    variants = {'cell': Ctx(C, idx, dups), 'dated rows only': Ctx(C, idx, dups, cols=C['dated']), 'artist masked': Ctx(C, idx, dups, artist_mask=True)}
    rd_all = {arm: [readings(r_, variants['cell'], k=15) for r_ in runs[arm]] for arm in runs}
    rdT = t_readings(C, S, idx, sc, k=15)
    out = {f'{arm}__{r_["tag"]}__{k}': v for arm in runs for r_, d in zip(runs[arm], rd_all[arm]) for k, v in d.items()}
    Paxis = unit(unit(S['P']['req']).mean(0))
    rads = {}
    for k in ('r_1', 'r_15', 'r_50', 'R1'):
        v = sc[k]
        if (v <= sc['r_0']).any():   # a radius at or below r_0 is not offered and not evaluated
            say(f'{k} {v[0]:.4f} / {v[1]:.4f}  NOT OFFERED (at or below r_0): not evaluated')
        else:
            rads[k] = v
            say(f'{k} {v[0]:.4f} / {v[1]:.4f}')
    assert CELL['r'] in rads, 'the cell radius is not offered'
    assert LS[-1] == CELL['L'], 'the census below reuses the last L of the loop as the cell L'
    say(f'\n== verdicts (valid starts {nunits}; G-yes {gyes}; G-no {gno}); started {time.strftime("%H:%M:%S")}')
    say('   (lookup lines at L, m, r other than the cell reuse the guard result of the one cell; the guards were run at the cell only)')
    cellres = None
    for L in LS:
        cal = sc['cal'][L]
        arms = {arm: pool_arm(runs[arm], rd_all[arm], cal) for arm in runs}
        say(f'L {L}: eligible per start ' + '; '.join(f"{arm} {[int(arms[arm]['elig'][arms[arm]['unit'] == u].sum()) for u in range(nunits)]}" for arm in arms))
        for rk, r in rads.items():
            res = evaluate(arms, r, acc['E'], nunits, draws=a.draws)
            for m in MS:
                for strict in (True, False):
                    rule, v, note = verdict(nunits, lacks, m, strict, res, gyes, gno)
                    assert verdict_with(nunits, lacks, m, strict, res, gyes, gno, ('raw', 'A', 'B')) == (rule, v, note), 'verdict_with differs from verdict'
                    alt = [verdict_with(nunits, lacks, m, strict, res, gyes, gno, rdg)[0] for _, rdg in READINGS[1:]]
                    places = {arm: sum(1 for p in res[arm]['places'] if p['support'] >= m) for arm in arms}
                    is_cell = (L, rk, m, strict) == (CELL['L'], CELL['r'], CELL['m'], CELL['strict'])
                    say(f"{'CELL ' if is_cell else '     '}L {L} {rk:4s} m {m} {'strict ' if strict else 'lenient'}: places V/C1/C2 {places['V']}/{places['C1']}/{places['C2']}; "
                        f"exclusive V/C1/C2 {'/'.join(str(excl_count(res[x]['places'], res[x]['info'], m, strict)) for x in arms)} -> rule {rule}: {v} [{note[:300]}]"
                        f" | beside: raw + B rule {alt[0]}, raw alone rule {alt[1]}")
                    if is_cell:
                        cellres = (arms, res, r, cal, rule, v, note)
            say(f'   (L {L} {rk} done {time.strftime("%H:%M:%S")})')
    arms, res, r, cal, rule, v, note = cellres
    say(f'\n== THE CELL (L 99, m 3, r_15, strict): rule {rule}: {v}. {note}')
    say('-- side by side at the cell (first-match order from the top in each line; printed beside, only the first line is in force):')
    for gl, gn_ in (('G-no (ii) both radii (in force)', gno), ("G-no (ii) the cell's radius alone", gno_cellr)):
        for lab, rdg in READINGS:
            if gn_ is None:
                say(f'   thinning {lab}; {gl}: not available in this guard file')
                continue
            rr = verdict_with(nunits, lacks, CELL['m'], CELL['strict'], res, gyes, gn_, rdg)
            say(f'   thinning {lab}; {gl}: rule {rr[0]}: {rr[1]} [{rr[2][:300]}]')
    for loop in ('C1', 'C2'):
        pc_ = per_start_counts(arms, res, loop, nunits)
        say(f'-- eligible pictures per start left by each reading, V against {loop}: ' + '; '.join(f'{k} V {vv} / {loop} {ll}' for k, (vv, ll) in pc_.items()))
    measure_record(out, arms, res, runs, rd_all, cal, CELL['m'], CELL['strict'])
    for k, p in enumerate(res['V']['places']):
        if p['support'] >= CELL['m']:
            say(f"-- V place {k}: accident {bool(res['V']['info']['acc'][p['c']])}; blocking starts " + ', '.join(f"{o} {int(b[p['c']])}" for o, b in res['V']['info']['blk'].items()))
            census(arms, runs, res, p, r, sc, S, C, acc, rd_all, Paxis, S['P']['req'])
    for arm in ('C1', 'C2'):
        for k, p in enumerate(res[arm]['places']):
            if p['support'] >= CELL['m']:
                say(f"-- {arm} place {k}: support {p['support']}, members {len(p['members'])}, accident {bool(res[arm]['info']['acc'][p['c']])}, blocking " +
                    ', '.join(f"{o} {int(b[p['c']])}" for o, b in res[arm]['info']['blk'].items()))
    say('\n== printed beside (at the cell)')
    farT = {}
    for arm in arms:
        el = arms[arm]['elig']
        d = nearest({n: arms[arm]['X'][n][el] for n in OBS}, S['T']['E'], [[]] * int(el.sum()), k=15)
        farT[arm] = int(np.all([d[n].mean(1) > FAR_T[o] for o, n in enumerate(OBS)], axis=0).sum())
    say(f'eligible pictures far from T at T\'s q99 ({FAR_T[0]} / {FAR_T[1]}, mean chord to 15 nearest T repaints, both): ' + ', '.join(f'{x} {farT[x]}' for x in farT))
    sw = []
    for bits in range(2 ** nunits):
        flip = [(bits >> u) & 1 for u in range(nunits)]
        rv = [runs['C1'][u] if flip[u] else runs['V'][u] for u in range(nunits)]
        rc = [runs['V'][u] if flip[u] else runs['C1'][u] for u in range(nunits)]
        dv = [rd_all['C1'][u] if flip[u] else rd_all['V'][u] for u in range(nunits)]
        dc = [rd_all['V'][u] if flip[u] else rd_all['C1'][u] for u in range(nunits)]
        sa = {'V': pool_arm(rv, dv, cal), 'C1': pool_arm(rc, dc, cal), 'C2': arms['C2']}
        el, Tm = arm_pairs(sa['V'], r)
        sw.append(excl_count(places_of(sa['V'], el, Tm), centre_info(sa, 'V', el, r, acc['E'], nunits), CELL['m'], CELL['strict']))
    say(f'swap (V and C1 exchanged within start, all {2 ** nunits} ways): exclusive V places {sw}; share >= the actual {sw[0]}: {np.mean(np.array(sw) >= sw[0]):.3f}')
    say('lookups printed as counts only (cell otherwise): eligible per arm, places per arm (support >= 3), exclusive V')
    def counts_line(lab, ax, **kw):
        res_ = {}
        for x in ax:
            el, Tm = arm_pairs(ax[x], r)
            res_[x] = (el, places_of(ax[x], el, Tm, **kw))
        info = centre_info(ax, 'V', res_['V'][0], r, acc['E'], nunits)
        elig = ', '.join(f"{x} {int(ax[x]['elig'].sum())}" for x in ax)
        npl = ', '.join(f"{x} {sum(1 for p_ in res_[x][1] if p_['support'] >= CELL['m'])}" for x in ax)
        say(f"  {lab}: eligible {elig}; places {npl}; exclusive V {excl_count(res_['V'][1], info, CELL['m'], CELL['strict'])}")

    counts_line('q 1 (no rounds rule)', arms, q=1, min_rounds=1)
    counts_line('q 20', arms, q=20)

    def alt(lab, rdx, calx, faith=FAITH, ekey='e'):
        counts_line(lab, {x: pool_arm(runs[x], rdx[x], calx, faith, ekey) for x in runs})

    alt(f'F at {FAITH_T}', rd_all, cal, FAITH_T)
    c15 = dict(med=np.median(rdT['e15'], 0), theta=float(np.percentile((rdT['e15'] / np.median(rdT['e15'], 0)).min(1), CELL['L'])), hbar=cal['hbar'])
    alt('E on the mean of the 15 nearest', rd_all, c15, ekey='e15')
    for lab in ('dated rows only', 'artist masked'):
        rdv = {x: [readings(r_, variants[lab]) for r_ in runs[x]] for x in runs}
        T = S['T']
        rt = readings(dict(E=T['E'], aim=T['aim'], req=T['req'], clip=T['clip'], src=T['src']), variants[lab])
        alt(lab, {x: [dict(d, a=d0['a'], f=d0['f']) for d, d0 in zip(rdv[x], rd_all[x])] for x in runs}, dict(calibrate(rt['e'], sc['T_a'], CELL['L']), hbar=cal['hbar']))
    P2 = S['P2']
    PP = {k: np.concatenate([S['P'][k], P2[k]]) for k in ('src', 'aim', 'req', 'clip')}
    PP['E'] = {n: np.concatenate([S['P']['E'][n], P2['E'][n]]) for n in OBS}
    rp = readings(PP, Ctx(C, idx, dup_rows(C, PP['src'])))
    alt(f"calibration on P + P2 ({len(PP['src'])} repaints)", rd_all, calibrate(rp['e'], rp['a'], CELL['L']))
    out['verdict_rule'] = np.int64(rule)
    out['verdict'] = np.array(v)
    out['verdict_note'] = np.array(note)
    out['cell_m'], out['cell_strict'] = np.int64(CELL['m']), np.bool_(CELL['strict'])
    out['T__paths'], out['acc__paths'], out['acc__kind'] = S['T']['paths'], acc['paths'], acc['kind'].astype(str)
    write(a, out)


def support_of(units, rounds, q=Q, min_rounds=MIN_ROUNDS):
    """starts with >= q pictures from >= min_rounds rounds among the given members"""
    out = []
    for u in np.unique(units):
        k = units == u
        if k.sum() >= q and len(np.unique(rounds[k])) >= min_rounds:
            out.append(int(u))
    return out


def main(argv=None):
    if not __debug__:
        raise SystemExit('asserts are off (python -O); refusing to run')
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('scales', 'guards', 'measure'))
    p.add_argument('--out', required=True)
    p.add_argument('--corpus', default='real')
    p.add_argument('--root', default=O)
    p.add_argument('--rows', type=int, default=1921)
    p.add_argument('--rounds', type=int, default=REAL['rounds'])
    p.add_argument('--per-round', type=int, default=REAL['per_round'])
    p.add_argument('--run-tmpl', default='{root}/runs/{tag}_committee_euclidean_min_seedrow/all_candidates.npz')
    p.add_argument('--exact-tmpl', default='{root}/eval/{tag}_emb.npz')
    p.add_argument('--safety-tmpl', default=None)
    p.add_argument('--r0-cells', default=','.join(QGC_C2))
    p.add_argument('--r1-fit', default=T1.replace('{f}', 'H2_allfit_K32.npz'))
    p.add_argument('--r1-fit-past', default=T1.replace('{f}', 'centres_radii.npz'))
    p.add_argument('--acc-prior', default=f'{B}/tests12/t2/renders_v1/prior256_v1')
    p.add_argument('--acc-noise', default=f'{B}/tests12/t2/renders_v1/noise160_v1')
    p.add_argument('--acc-aim-prior', default=None)
    p.add_argument('--acc-aim-noise', default=None)
    p.add_argument('--chains', default=f'{O}/baselines/chains_h200_qg')
    p.add_argument('--raw-noise', default=f'{O}/eval/bound/noise_check/noise_emb.npz')
    p.add_argument('--scales', default=None)
    p.add_argument('--guards', default=None)
    p.add_argument('--freeze', default=None)
    p.add_argument('--afterrun', default=f'{B}/tests12/afterrun')   # measure: the cells' COMPLETE markers and task.logs
    p.add_argument('--cells', default=';'.join(f'{k}:{",".join(v)}' for k, v in QGT.items()))
    p.add_argument('--draws', type=int, default=NDRAW)
    a = p.parse_args(argv)
    LINES.clear()
    a.run_tmpl, a.exact_tmpl = a.run_tmpl.replace('{root}', a.root), a.exact_tmpl.replace('{root}', a.root)
    say(f'{os.path.basename(__file__)}; mode {a.mode}; args {vars(a)}')
    {'scales': mode_scales, 'guards': mode_guards, 'measure': mode_measure}[a.mode](a)


if __name__ == '__main__':
    main()
