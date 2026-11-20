"""Lean (controls_evaluate.qfun) of the free runs' pictures: free search main_s1-3, the same loop told random scores rf_s1-3, random directions at its angles b0_s1-3; exact CLIP readings eval/<tag>_emb.npz, painted rows only.
Usage: python -B lk_lean_v1.py OUTPREFIX -> OUTPREFIX.txt (refused if it exists); prints per run and pooled the reading lean median [p10, p90] and the request lean median."""
import importlib.util, os, sys
import numpy as np
B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
EV_FILE = f'{C_}/controls_evaluate.py'
FH = [None]
def say(s=''):
    print(s, flush=True); FH[0].write(str(s) + '\n'); FH[0].flush()
def main(prefix):
    assert not os.path.exists(prefix + '.txt')
    FH[0] = open(prefix + '.txt', 'x', encoding='utf-8')
    spec = importlib.util.spec_from_file_location('ce', EV_FILE); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    q = ev.qfun(O)
    st = lambda a: f'median {np.median(a):+.4f} [p10 {np.percentile(a, 10):+.4f}, p90 {np.percentile(a, 90):+.4f}] n {len(a)}'  # noqa: E731
    for lab, pre, fn in (('free search, told the real score', 'main', 'all_candidates.npz'), ('free search, told random scores', 'rf', 'all_candidates.npz'),
                         ('random directions at its angles', 'b0', 'all_draws.npz')):
        R, Q = [], []
        for n in (1, 2, 3):
            tag = f'{pre}_s{n}'
            z = np.load(f'{O}/runs/{tag}_committee_euclidean_min_seedrow/{fn}', allow_pickle=True)
            x = np.load(f'{O}/eval/{tag}_emb.npz', allow_pickle=True)
            assert len(z['paths']) == len(x['paths']) and all(str(a) == str(b) for a, b in zip(z['paths'], x['paths'])), tag
            m = np.asarray(z['iteration']) >= 0
            r, rq = q(x['emb_clip-vit-l-14'])[m], q(z['style_embs'])[m]
            R.append(r); Q.append(rq)
            say(f'   {lab:34s} {tag}: reading {st(r)} | request median {np.median(rq):+.4f}')
        say(f'   {lab:34s} POOLED: reading {st(np.concatenate(R))} | request median {np.median(np.concatenate(Q)):+.4f}')
    FH[0].close()
if __name__ == '__main__':
    main(sys.argv[1])
