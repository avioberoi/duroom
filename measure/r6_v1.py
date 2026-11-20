"""The manifold line set on other sets of works, and the on/off and dated labels of the far and faithful pictures under each line (counts only).
Usage (cwd = the search folder): python -B r6_v1.py OUTPREFIX -> OUTPREFIX.txt + OUTPREFIX.npz (refused if either exists)"""
import glob
import importlib.util
import os
import sys
import time
import types

import numpy as np

B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
OO = f'{B}/tests12/onoff/out'
T3_FILE = f'{C_}/t3_measure_v5.py'
ONC, ONF, DAT = f'{OO}/onoff_v3_C.npz', f'{OO}/onoff_ff_v1.npz', f'{OO}/onoff_dated_v2.npz'
STARTS = (6, 7, 8, 9, 10)
ARMS = (('V', 'ra', 'search'), ('C1', 'pin', 'random-score loop'), ('C2', 'c2', 'no-step loop'))
SEED = 20261120
FH = [None]


def say(s=''):
    print(s, flush=True)
    FH[0].write(str(s) + '\n')
    FH[0].flush()


def line_of(e, L=99.0):
    """med per observer, theta = L-th percentile of min over observers of e / med, line = theta x med"""
    e = np.asarray(e, np.float64)
    med = np.median(e, axis=0)
    theta = float(np.percentile((e / med).min(1), L))
    return med, theta, theta * med


def chord(s):
    return np.sqrt(np.clip(2.0 - 2.0 * s, 0.0, None))


def all_pairs_nearest(Ed, Es, dup=0.99, chunk=512):
    """every original's nearest other original in each observer: (a) near-duplicates (cosine > dup in either observer) and itself left out, (b) only itself"""
    n = len(Ed)
    nd = np.empty((n, 2), np.float64)
    na = np.empty((n, 2), np.float64)
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        Sd = Ed[i:j] @ Ed.T
        Ss = Es[i:j] @ Es.T
        r = np.arange(j - i)
        Sd[r, np.arange(i, j)] = -2.0
        Ss[r, np.arange(i, j)] = -2.0
        na[i:j, 0], na[i:j, 1] = chord(Sd.max(1)), chord(Ss.max(1))
        D = (Sd > dup) | (Ss > dup)
        Sd[D] = -2.0
        Ss[D] = -2.0
        nd[i:j, 0], nd[i:j, 1] = chord(Sd.max(1)), chord(Ss.max(1))
    return nd, na


def main(prefix):
    t0 = time.time()
    for ext in ('.txt', '.npz'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    spec = importlib.util.spec_from_file_location('t3_measure_v5', T3_FILE)
    t3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t3)
    say(f'numpy {np.__version__}')
    C, S, idx = t3.load_common(types.SimpleNamespace(corpus='real', root=O))
    OBS = t3.OBS
    say(f'   corpus {len(C["ids"])} originals; DUP {t3.DUP}; {time.time() - t0:.0f} s')

    def rep_e(names):
        src = np.concatenate([S[n]['src'] for n in names])
        E = {o: np.concatenate([S[n]['E'][o] for n in names]) for o in OBS}
        ctx = t3.Ctx(C, idx, t3.dup_rows(C, src))
        return t3.readings({'E': E, 'src': src}, ctx)['e'], src
    # the line from today's 2,048 repaints T
    eT, srcT = rep_e(['T'])
    med, theta, line = line_of(eT)
    FROZ = line.copy()
    onc, onf, dat = (np.load(f, allow_pickle=True) for f in (ONC, ONF, DAT))
    R = {}
    for arm, tg, w in ARMS:
        for n in STARTS:
            t = f'qgc_{tg}_s{n}'
            R[t] = dict(e=np.asarray(onc[f'{t}__e'], np.float64), FF=np.asarray(onf[f'{t}__FF'], bool),
                        pre=np.asarray(dat[f'{t}__pool_PRE'], np.float64), post=np.asarray(dat[f'{t}__pool_POST'], np.float64))
            assert R[t]['e'].shape == (1920, 2) and R[t]['pre'].shape == (1920, 2) and len(R[t]['FF']) == 1920

    def labels(L):
        # OFF: far and faithful, nearest original (masked) beyond the line in both observers; SINCE: no original before 1900 within the line in either observer, one from 1900 on within it in at least one (unmasked pools)
        out = {}
        for arm, tg, w in ARMS:
            off, since, ff = [], [], []
            for n in STARTS:
                r = R[f'qgc_{tg}_s{n}']
                o = r['FF'] & np.all(r['e'] > L, axis=1)
                s_ = r['FF'] & ~np.any(r['pre'] <= L, axis=1) & np.any(r['post'] <= L, axis=1)
                off.append(int(o.sum())); since.append(int(s_.sum())); ff.append(int(r['FF'].sum()))
            out[arm] = dict(off=off, since=since, ff=ff)
        return out
    sv = {'eT': eT}
    LINES = {'used (T, 99%)': FROZ}
    # (i)(a) all originals
    nd, na = all_pairs_nearest(C['E'][OBS[0]], C['E'][OBS[1]], t3.DUP)
    sv['orig_nearest_nodup'], sv['orig_nearest_any'] = nd, na
    say()
    say('== (i) ALTERNATIVE LINES (theta x median per observer, 99% rule): med D / med S | theta | line D / line S | line vs used')
    say(f'   used, today\'s 2,048 repaints T: {med[0]:.4f} / {med[1]:.4f} | {theta:.4f} | {FROZ[0]:.4f} / {FROZ[1]:.4f}')
    for lab_, e in (('(a) originals to nearest other original, near-duplicates out', nd), ('(a) originals to nearest other original, nothing out but itself', na)):
        m_, th, ln = line_of(e)
        say(f'   {lab_}: {m_[0]:.4f} / {m_[1]:.4f} | {th:.4f} | {ln[0]:.4f} / {ln[1]:.4f} | {ln[0] / FROZ[0]:.3f}x / {ln[1] / FROZ[1]:.3f}x')
        say(f'        nearest-other-original chord: median {np.median(e[:, 0]):.4f} / {np.median(e[:, 1]):.4f}; 99th percentile {np.percentile(e[:, 0], 99):.4f} / {np.percentile(e[:, 1], 99):.4f}; '
            f'chord 0 (exact duplicate) {int((e[:, 0] < 1e-4).sum())} / {int((e[:, 1] < 1e-4).sum())}')
        LINES[lab_.split(',')[0][:3] + (' nodup' if 'out,' in lab_ or 'duplicates out' in lab_ else ' any')] = ln
    # (b) out-of-sample repaints
    eB, _ = rep_e(['P2', 'pos'])
    m_, th, ln = line_of(eB)
    say(f'   (b) the 1,512 repaints not used (P2 1,000 + pos 512), masked as for the line used: {m_[0]:.4f} / {m_[1]:.4f} | {th:.4f} | {ln[0]:.4f} / {ln[1]:.4f} | '
        f'{ln[0] / FROZ[0]:.3f}x / {ln[1] / FROZ[1]:.3f}x')
    LINES['(b) 1,512 held-out repaints'] = ln
    # (c) second draw / halves / date parts
    cand = sorted(set(glob.glob(f'{O}/push_today*_emb.npz')) - {f'{O}/push_today_emb.npz'})
    say(f'   (c) second draw of today\'s reference: other push_today*_emb.npz files: {cand if cand else "NONE (no second draw was painted)"}')
    rng = np.random.default_rng(SEED)
    hl = []
    for _ in range(20):
        p = rng.permutation(len(eT))[:len(eT) // 2]
        hl.append(line_of(eT[p])[2])
    hl = np.array(hl)
    say(f'   (c) 20 random halves of T (1,024 each): line D {hl[:, 0].min():.4f} to {hl[:, 0].max():.4f}; line S {hl[:, 1].min():.4f} to {hl[:, 1].max():.4f} '
        f'({hl[:, 0].min() / FROZ[0]:.3f}x to {hl[:, 0].max() / FROZ[0]:.3f}x; {hl[:, 1].min() / FROZ[1]:.3f}x to {hl[:, 1].max() / FROZ[1]:.3f}x)')
    k_lo, k_hi = int(np.argmin(hl[:, 0] + hl[:, 1])), int(np.argmax(hl[:, 0] + hl[:, 1]))
    LINES['(c) half, shortest'], LINES['(c) half, longest'] = hl[k_lo], hl[k_hi]
    yr = C['year'][srcT]
    for lab_, msk in (('before 1900', yr < 1900), ('1900-1969', (yr >= 1900) & (yr < 1970)), ('1970 on', yr >= 1970)):
        m_, th, ln = line_of(eT[msk])
        say(f'   (c) T\'s part {lab_} ({int(msk.sum())}): {m_[0]:.4f} / {m_[1]:.4f} | {th:.4f} | {ln[0]:.4f} / {ln[1]:.4f} | {ln[0] / FROZ[0]:.3f}x / {ln[1] / FROZ[1]:.3f}x')
        LINES[f'(c) T part {lab_}'] = ln
    for L_ in (95, 98, 99.5):
        LINES[f'cut {L_}% on T'] = line_of(eT, L_)[2]
    for f_ in (0.90, 0.95, 1.05, 1.10):
        LINES[f'used x {f_:.2f}'] = FROZ * f_
    # (ii) labels under each line
    say()
    say('== (ii) LABELS of the far and faithful pictures under each line: OFF (beyond in both) / SINCE (later within, none before), per loop as count (share of FF);'
        ' search OFF by start; Fra Angelico (s7) share of the search OFF; median nearest-original chord of the search OFF, D / S')
    for name, L in LINES.items():
        lb = labels(L)
        parts = []
        for arm, tg, w in ARMS:
            ff = sum(lb[arm]['ff'])
            parts.append(f'{arm} OFF {sum(lb[arm]["off"])} ({100 * sum(lb[arm]["off"]) / ff:.1f}%) SINCE {sum(lb[arm]["since"])} ({100 * sum(lb[arm]["since"]) / ff:.1f}%)')
        offV = np.concatenate([R[f'qgc_ra_s{n}']['e'][R[f'qgc_ra_s{n}']['FF'] & np.all(R[f'qgc_ra_s{n}']['e'] > L, axis=1)] for n in STARTS])
        tot = sum(lb['V']['off'])
        say(f'   {name:42s} line {L[0]:.4f} / {L[1]:.4f}: ' + ' | '.join(parts))
        say(f'   {"":42s} search OFF by start {lb["V"]["off"]}; s7 share {100 * lb["V"]["off"][1] / tot if tot else float("nan"):.0f}%; '
            f'median chord of OFF {np.median(offV[:, 0]) if len(offV) else float("nan"):.3f} / {np.median(offV[:, 1]) if len(offV) else float("nan"):.3f}; '
            f'C1 OFF by start {lb["C1"]["off"]}; C2 OFF by start {lb["C2"]["off"]}')
    say()
    say('== (ii) RATIO r = min over observers of (nearest-original chord / used line), far and faithful pictures (OFF iff r > 1)')
    for arm, tg, w in ARMS:
        r = np.concatenate([(R[f'qgc_{tg}_s{n}']['e'][R[f'qgc_{tg}_s{n}']['FF']] / FROZ).min(1) for n in STARTS])
        sv[f'{arm}__ratio'] = r
        say(f'   {arm} {w:18s} n {len(r)}: median {np.median(r):.3f}; within 5% of 1: {int((np.abs(r - 1) <= 0.05).sum())} ({100 * np.mean(np.abs(r - 1) <= 0.05):.1f}%); '
            f'within 10%: {int((np.abs(r - 1) <= 0.10).sum())} ({100 * np.mean(np.abs(r - 1) <= 0.10):.1f}%); over 1 (OFF) {int((r > 1).sum())}')
    with open(prefix + '.npz', 'xb') as fh:
        np.savez(fh, **sv)
    say(f'saved {prefix}.npz; done {time.time() - t0:.0f} s')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
