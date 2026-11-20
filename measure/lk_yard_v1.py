"""Exact counts behind the region-yardstick: the share of the 2,048 repainted reference works whose CLIP reading falls in its original's region and class (cosine to the mean-removed centres of the region fit).
Usage: python -B lk_yard_v1.py OUTPREFIX -> OUTPREFIX.txt (refused if it exists); stops unless the region rule reproduces the fit's own assignment of all its works."""
import importlib.util, json, os, sys
import numpy as np
B = os.environ['DUROOM_OUT']
C_, O = os.path.dirname(os.path.abspath(__file__)), f'{B}/sphere'
T1 = f'{B}/tests12/t1/run_rob_clip'
def unit(X):
    X = np.asarray(X, np.float64); return X / np.linalg.norm(X, axis=-1, keepdims=True)
def main(prefix):
    assert not os.path.exists(prefix + '.txt'); fh = open(prefix + '.txt', 'x')
    say = lambda s: (print(s), fh.write(s + '\n'))  # noqa: E731
    af, pw = np.load(f'{T1}/clip-vit-l-14_H2_allfit_K32.npz'), np.load(f'{T1}/clip-vit-l-14_H1_perwork.npz', allow_pickle=True)
    vz = np.load(os.environ['DUROOM_EMB'] + '/clip-vit-l-14.npz', allow_pickle=True)
    idk = [k for k in ('ids', 'id', 'work_ids') if k in vz.files][0]
    row = {int(i): j for j, i in enumerate(np.asarray(vz[idk]).astype(np.int64))}
    V = unit(vz['embeddings'])
    Qids, Qreg = np.asarray(af['Q_ids']).astype(np.int64), np.asarray(af['Q_region'])
    mean, C = np.asarray(af['mean'], np.float64), unit(af['centres'])
    region = lambda X: np.argmax(unit(unit(X) - mean) @ C.T, axis=1)  # noqa: E731
    ok = np.array_equal(region(V[[row[int(i)] for i in Qids]]), Qreg)
    say(f'GUARD rule reproduces Q_region on {len(Qids)} works: {ok}')
    if not ok:
        fh.close(); sys.exit(3)
    g = json.load(open(f'{B}/tests12/geom/geom_clip-vit-l-14_v1.json'))['mapA']['regions']
    cls = np.digitize(np.array([r['pre1900_share_of_A_plus_later'] for r in g]), [0.01, 0.10, 0.50])
    spec = importlib.util.spec_from_file_location('ce', f'{C_}/controls_evaluate.py'); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    P, T, Pz = ev.references(O)
    pid = np.asarray(P['id']).astype(np.int64); qrow = {int(i): j for j, i in enumerate(Qids)}
    have = np.array([int(i) in qrow for i in pid]); oreg = np.array([Qreg[qrow[int(i)]] if int(i) in qrow else -1 for i in pid])
    preg = region(Pz['emb_clip-vit-l-14'])
    sr, sc = int(np.sum(preg[have] == oreg[have])), int(np.sum(cls[preg[have]] == cls[oreg[have]]))
    say(f'yardstick: originals in the fit set {int(have.sum())} of {len(pid)}; same region {sr} ({sr / have.sum():.4f}); same class {sc} ({sc / have.sum():.4f})')
    fh.close()
if __name__ == '__main__':
    main(sys.argv[1])
