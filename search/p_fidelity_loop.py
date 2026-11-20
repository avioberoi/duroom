"""Encodes every P render with the loop's CLIP reader and saves/prints its chord to the requested vector (the q99 is --fidelity-max).
Usage: python p_fidelity_loop.py PUSH_EMB_NPZ OUT_NPZ   (GPU, render env, SPHERE_CLIP=quickgelu)"""
import os
import sys

import numpy as np

CODE = os.environ.get('SPHERE_CODE', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import utils  # noqa: E402

P = np.load(sys.argv[1], allow_pickle=True)
enc = utils.load_encoder('clip-vit-l-14')[0]
V = np.asarray(P['vec'], np.float64)
V /= np.linalg.norm(V, axis=1, keepdims=True)
R = np.stack([enc(str(p)) for p in P['paths']]).astype(np.float64)
R /= np.linalg.norm(R, axis=1, keepdims=True)
fid = np.sqrt(np.clip(2 - 2 * (R * V).sum(1), 0, None))
np.savez(sys.argv[2], paths=P['paths'], fidelity_loop=fid.astype(np.float32))
print('P fidelity on the loop encoder: p50 %.3f p90 %.3f p99 %.3f (exact, clip package: p50 %.3f p99 %.3f)' % (
    *np.percentile(fid, (50, 90, 99)), *np.percentile(P['fidelity_clip'], (50, 99))))
