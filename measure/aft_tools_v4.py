"""Checks around the exact embedding files and the evaluator's safety pass.
Usage: python aft_tools_v4.py list RUNFILE OUTLIST N | plist NPZ OUTLIST | katlist OUTLIST | katcmp KATNPZ [ENCS] | valid EXACT RUNFILE [ENCS] | cmp NEW STORED | cmpc NEW STORED | safety LIST OUTJSON N | safetycheck STOREDJSON OUTJSON | cardcmp NEW STORED RUNFILE KATNPZ | merge OUTJSON IN1 IN2 ..."""
import json
import os
import sys

import numpy as np

O = os.environ['DUROOM_OUT'] + '/sphere'
CODE = os.path.dirname(os.path.abspath(__file__))
ENC = ('clip-vit-l-14', 'dinov3-l', 'siglip-so400m', 'aimv2-huge')
TOL = 2.5e-4


def unit(X):
    X = np.asarray(X, np.float64)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def newfile(p):
    return open(p, 'x')   # refuses an existing file


def chords(a, b):
    return np.linalg.norm(unit(a) - unit(b), axis=-1)


def main(cmd, *a):
    if cmd == 'list':
        run, out, n = a
        p = np.load(run, allow_pickle=True)['paths']
        assert p.shape[0] == int(n), f'run file has {p.shape[0]} rows, expected {n}'
        newfile(out).write('\n'.join(map(str, p)) + '\n')
        print(f'list written: {out}, rows {p.shape[0]}')
    elif cmd == 'plist':
        z, out = a
        p = np.load(z, allow_pickle=True)['paths']
        newfile(out).write('\n'.join(map(str, p)) + '\n')
        print(f'list written: {out}, rows {p.shape[0]}')
    elif cmd == 'katlist':
        out, = a
        s = np.load(f'{O}/push/shard1.npz', allow_pickle=True)
        ids = [int(i) for i in s['ids'][:8]]
        P = np.load(f'{O}/push_emb.npz', allow_pickle=True)['paths']
        byid = {int(os.path.basename(str(p))[5:-4]): str(p) for p in P}
        newfile(out).write('\n'.join(byid[i] for i in ids) + '\n')
        print(f'known-answer list written: {out}, 8 pictures of P (ids {ids})')
    elif cmd == 'katcmp':
        k, encs = a[0], (a[1].split(',') if len(a) > 1 else ENC)
        K, P = np.load(k, allow_pickle=True), np.load(f'{O}/push_emb.npz', allow_pickle=True)
        row = {str(p): i for i, p in enumerate(P['paths'])}
        idx = [row[str(p)] for p in K['paths']]
        assert len(idx) == 8
        bad = 0
        for e in encs:
            c = chords(K[f'emb_{e}'], P[f'emb_{e}'][idx])
            bad += int((c > TOL).sum())
            print(f'known-answer {e}: max chord to the stored reference reading {c.max():.3e} over 8 ({int((c > TOL).sum())} above {TOL})')
        print('known-answer', 'PASS' if bad == 0 else 'FAIL')
        sys.exit(0 if bad == 0 else 4)
    elif cmd == 'valid':
        ex, run, encs = a[0], a[1], (a[2].split(',') if len(a) > 2 else ENC)
        X, R = np.load(ex, allow_pickle=True), np.load(run, allow_pickle=True)['paths']
        assert np.array_equal(X['paths'].astype(str), R.astype(str)), 'exact file paths differ from the run file paths'
        for e in encs:
            v = X[f'emb_{e}']
            assert v.shape[0] == R.shape[0] and np.isfinite(v).all() and np.abs(np.linalg.norm(v.astype(np.float64), axis=1) - 1).max() < 1e-3, e
        print(f'exact file valid: rows {R.shape[0]}, encoders {list(encs)} finite and unit, paths equal the run file in order')
    elif cmd == 'cmp':
        new, old = a
        N, S = np.load(new, allow_pickle=True), np.load(old, allow_pickle=True)
        assert np.array_equal(N['paths'].astype(str), S['paths'].astype(str)), 'paths differ'
        print(f'rows {N["paths"].shape[0]}; keys new {sorted(N.files)} stored {sorted(S.files)}')
        bad = 0
        for e in ENC:
            c = chords(N[f'emb_{e}'], S[f'emb_{e}'])
            bad += int((c > TOL).sum())
            print(f'{e}: max chord {c.max():.3e}, median {np.median(c):.3e}, p99 {np.percentile(c, 99):.3e}, rows above {TOL}: {int((c > TOL).sum())} of {len(c)}, '
                  f'bit-identical rows {int((N[f"emb_{e}"] == S[f"emb_{e}"]).all(1).sum())}')
        print('new against stored:', 'PASS' if bad == 0 else 'FAIL')
    elif cmd == 'safety':
        lst, out, n = a
        sys.path.insert(0, CODE)
        import controls_evaluate as ev
        paths = [line.strip() for line in open(lst) if line.strip()]
        assert len(paths) == int(n), f'{len(paths)} pictures in the list, expected {n}'
        assert not os.path.exists(out), f'{out} exists'
        cache = {}
        fl = ev.safety(np.array(paths), cache, out, True)
        assert len(fl) == int(n) and len(json.load(open(out))) == int(n)
        print(f'safety pass done: {len(fl)} pictures, cache {out} holds {int(n)} keys (path|size|mtime_ns, as the evaluator keys them)')
    elif cmd == 'safetycheck':
        stored, out = a
        sys.path.insert(0, CODE)
        import controls_evaluate as ev
        S = json.load(open(stored))
        key = lambda p: f'{p}|{os.stat(p).st_size}|{os.stat(p).st_mtime_ns}'  # noqa: E731
        paths = [k.split('|')[0] for k in S]
        ok = [p for p, k in zip(paths, S) if os.path.exists(p) and key(p) == k]
        assert not os.path.exists(out)
        cache = {}
        fl = ev.safety(np.array(ok), cache, out, True)
        st = np.array([S[key(p)] for p in ok], bool)
        print(f'stored cache {stored}: {len(S)} keys, {len(ok)} files unchanged and recomputed here')
        print(f'stored flagged {int(st.sum())}, recomputed flagged {int(fl.sum())}, disagreements {int((st != fl).sum())}')
        print('safety recomputed against stored:', 'PASS' if (st == fl).all() else 'FAIL')
    elif cmd == 'cmpc':
        new, old = a
        N, S = np.load(new, allow_pickle=True), np.load(old, allow_pickle=True)
        assert np.array_equal(N['paths'].astype(str), S['paths'].astype(str)), 'paths differ'
        com = [e for e in ENC if f'emb_{e}' in N.files and f'emb_{e}' in S.files]
        print(f'rows {N["paths"].shape[0]}; encoders in both {com}; new only {[e for e in ENC if f"emb_{e}" in N.files and e not in com]}')
        bad = 0
        for e in com:
            c = chords(N[f'emb_{e}'], S[f'emb_{e}'])
            bad += int((c > TOL).sum())
            print(f'{e}: max chord {c.max():.3e}, median {np.median(c):.3e}, rows above {TOL}: {int((c > TOL).sum())} of {len(c)}')
        print('new against stored:', 'PASS' if bad == 0 else 'FAIL')
    elif cmd == 'cardcmp':
        new, old, run, kat = a
        assert '/qgt_' not in run and '/qgt_' not in old, 'cardcmp is for stored cells only, not qgt cells'
        N, S, R = np.load(new, allow_pickle=True), np.load(old, allow_pickle=True), np.load(run, allow_pickle=True)
        assert np.array_equal(N['paths'].astype(str), S['paths'].astype(str)) and np.array_equal(N['paths'].astype(str), R['paths'].astype(str)), 'paths differ'
        st = lambda c: f'median {np.median(c):.3e}, max {c.max():.3e}, rows above {TOL}: {int((c > TOL).sum())} of {len(c)}'  # noqa: E731
        for e in ENC:
            print(f'(a) new vs stored exact file, {e}: {st(chords(N[f"emb_{e}"], S[f"emb_{e}"]))}')
        print(f'(b) new CLIP vs run file render_clip (loop reader): {st(chords(N["emb_clip-vit-l-14"], R["render_clip"]))}')
        print(f'(b) stored exact CLIP vs run file render_clip (for scale): {st(chords(S["emb_clip-vit-l-14"], R["render_clip"]))}')
        K, P = np.load(kat, allow_pickle=True), np.load(f'{O}/push_emb.npz', allow_pickle=True)
        row = {str(p): i for i, p in enumerate(P['paths'])}
        idx = [row[str(p)] for p in K['paths']]
        assert len(idx) == 8
        for e in ENC:
            print(f'(c) known-answer vs push_emb.npz, {e}: {st(chords(K[f"emb_{e}"], P[f"emb_{e}"][idx]))}')
    elif cmd == 'merge':
        out, *ins = a
        assert not os.path.exists(out), f'{out} exists'
        m = {}
        for f in ins:
            d = json.load(open(f))
            assert not set(d) & set(m), f'{f} shares keys with an earlier file'
            m.update(d)
        json.dump(m, newfile(out))
        print(f'merged {len(ins)} caches into {out}: {len(m)} keys')
    else:
        raise SystemExit(f'unknown command {cmd}')


if __name__ == '__main__':
    main(*sys.argv[1:])
