"""Leftover basis and line of a reference: refits P (checked against pre_v1_basis.npz), then fits T (push_today_emb.npz) and counts rows over the line.
Usage: python -B tr_fitbasis_v1.py OUTPREFIX -> OUTPREFIX_P_check.npz, OUTPREFIX_T_basis.npz, OUTPREFIX.txt (refused if any exists)"""
import hashlib
import importlib.util
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
B = os.environ['DUROOM_OUT']
C, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
LO = f'{B}/tests12/leftover'
R2 = f'{B}/tests12/t2/renders_v1'
LF_FILE = f'{C}/leftover_fn.py'
PBASIS = f'{LO}/out/pre_v1_basis.npz'
DINO, SIG, CLIP = 'dinov3-l', 'siglip-so400m', 'clip-vit-l-14'
NOV, ALL = (DINO, SIG), (DINO, SIG, CLIP)
RN = {DINO: 'DINOv3', SIG: 'SigLIP', CLIP: 'CLIP (logged only)'}
K, FOLDS, SEED = 32, 5, 0
md5 = lambda f: hashlib.md5(open(f, 'rb').read()).hexdigest()
FH = [None]


def say(s=''):
    print(s, flush=True)
    FH[0].write(s + '\n')
    FH[0].flush()


def fit_ref(lf, refpath):
    Rz = np.load(refpath, allow_pickle=True)
    art = np.array([str(a) for a in Rz['artist']])
    out = dict(K=np.array(K), folds=np.array(FOLDS), seed=np.array(SEED), p_md5=np.array(md5(refpath)), ref_file=np.array(refpath),
               observers=np.array(ALL), nov_observers=np.array(NOV))
    for o in ALL:
        E = np.asarray(Rz[f'emb_{o}'], dtype=np.float64)
        mean, V = lf.fit(E, K)
        assert V.shape == (K, E.shape[1]) and np.allclose(V @ V.T, np.eye(K), atol=1e-10)
        Lcf, fold = lf.crossfit(E, art, K, FOLDS, SEED)
        assert np.isfinite(Lcf).all() and set(fold.tolist()) == set(range(FOLDS))
        for f in range(FOLDS):
            assert not set(art[fold == f]) & set(art[fold != f]), f
        med, mad, q99 = lf.null(Lcf)
        out.update({f'mean_{o}': mean, f'V_{o}': V, f'null_median_{o}': np.array(med), f'null_mad_{o}': np.array(mad), f'null_q99_{o}': np.array(q99),
                    f'insample_q99_{o}': np.array(float(np.percentile(lf.share(E, mean, V), 99))), f'crossfit_share_{o}': Lcf, f'fold_{o}': fold})
    return out, art


def renders(name, sub, tagkey):
    em = np.load(f'{R2}/{name}/emb.npz', allow_pickle=True)
    kat = [ln.strip() for ln in open(f'{R2}/{name}/inputs/expected_kat.txt') if ln.strip()]
    assert [str(p) for p in em['paths'][:len(kat)]] == kat
    man = np.load(f'{R2}/{name}/{sub}/manifest.npz', allow_pickle=True)
    k = len(kat)
    assert [str(p) for p in em['paths'][k:]] == [str(p) for p in man['paths']] and len(em['paths']) == k + len(man['paths'])
    return {o: em[f'emb_{o}'][k:] for o in ALL}, np.array([str(t) for t in man[tagkey]])


def main(prefix):
    for ext in ('_P_check.npz', '_T_basis.npz', '.txt'):
        assert not os.path.exists(prefix + ext), f'{prefix + ext} exists: refusing to overwrite'
    FH[0] = open(prefix + '.txt', 'x')
    spec = importlib.util.spec_from_file_location('leftover_fn', LF_FILE)
    lf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lf)
    say(f'tr_fitbasis_v1.py; recipe: K {K}, artist-out cross-fit {FOLDS} folds seed {SEED}, line = null 99th, per reader')
    SB = np.load(PBASIS, allow_pickle=True)
    bp, _ = fit_ref(lf, f'{O}/push_emb.npz')
    worst = {}
    for o in ALL:
        for k in ('mean', 'V', 'crossfit_share', 'null_median', 'null_mad', 'null_q99', 'fold'):
            worst[f'{k}_{o}'] = float(np.abs(np.asarray(bp[f'{k}_{o}'], float) - np.asarray(SB[f'{k}_{o}'], float)).max())
    say('refit of P against the saved basis pre_v1_basis.npz, largest |difference| per array: ' + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()))
    with open(prefix + '_P_check.npz', 'xb') as fh:
        np.savez(fh, **bp)
    if max(worst.values()) > 1e-12:
        say('STOP: the refit of P does not reproduce the saved basis; T is not fitted.')
        sys.exit(3)
    say('   the refit reproduces P\'s saved basis and line.')
    say()
    tpath = f'{O}/push_today_emb.npz'
    bt, artT = fit_ref(lf, tpath)
    with open(prefix + '_T_basis.npz', 'xb') as fh:
        np.savez(fh, **bt)
    say(f'T = {tpath}: 2,048 rows expected, {len(artT)} rows, {len(np.unique(artT))} artists')
    assert len(artT) == 2048
    say(f'   T basis written: {prefix}_T_basis.npz')
    line = {o: float(bt[f'null_q99_{o}']) for o in ALL}
    for o in ALL:
        say(f'   {RN[o]:18s} null median {float(bt[f"null_median_{o}"]):.6f} MAD {float(bt[f"null_mad_{o}"]):.6f} line (99th) {line[o]:.6f} | in-sample 99th (not used) {float(bt[f"insample_q99_{o}"]):.6f} '
            f'| fold sizes {np.bincount(bt[f"fold_{o}"]).tolist()} | T rows over the line (cross-fit shares) {int((bt[f"crossfit_share_{o}"] > line[o]).sum())} of 2048 '
            f'| beside it, P\'s line {float(SB[f"null_q99_{o}"]):.6f}')
    both = (bt[f'crossfit_share_{DINO}'] > line[DINO]) & (bt[f'crossfit_share_{SIG}'] > line[SIG])
    say(f'   T rows over the line in both DINOv3 and SigLIP: {int(both.sum())} of 2048')
    say()
    say('FOR SCALE against T\'s basis and line (decoder renders; known-answer rows dropped):')
    fitted = {o: (bt[f'mean_{o}'], bt[f'V_{o}']) for o in ALL}
    for name, sub, tk, kinds in (('noise160_v1', 'noise', 'tag', 'abcde'), ('prior256_v1', 'prior', 'kind', ('cone', 'sphere'))):
        emb, kind = renders(name, sub, tk)
        L = {o: lf.share(emb[o], *fitted[o]) for o in ALL}
        say(f'   {name} n {len(kind)}')
        for kd in list(kinds) + ['all']:
            s = np.ones(len(kind), bool) if kd == 'all' else kind == kd
            ov = {o: L[o][s] > line[o] for o in ALL}
            say(f'      {kd:6s} n {int(s.sum()):3d} | over DINOv3 {int(ov[DINO].sum()):3d} | over SigLIP {int(ov[SIG].sum()):3d} | over BOTH {int((ov[DINO] & ov[SIG]).sum()):3d} | over CLIP (logged only) {int(ov[CLIP].sum()):3d}')
    FH[0].close()


if __name__ == '__main__':
    main(sys.argv[1])
