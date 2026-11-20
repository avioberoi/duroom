"""Saves the SigLIP gate's negative-prompt embedding (unit mean of "noise", "static", "random pixels", "distorted") as $DUROOM_OUT/inputs/neg_emb_siglip.npy.
Usage: python neg_emb.py   (GPU, network access: fetches the SigLIP tokenizer once)"""
import os
import sys

import numpy as np

CODE = os.environ.get('SPHERE_CODE', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import utils  # noqa: E402

neg, enc, thr = utils.load_coherence_gate(utils.load_config(os.path.join(CODE, 'config.yaml')), compute_neg=True)
out = os.environ['DUROOM_OUT'] + '/inputs/neg_emb_siglip.npy'
np.save(out, np.asarray(neg, dtype='float32'))
print('saved', out, neg.shape, 'norm %.4f' % float(np.linalg.norm(neg)), 'threshold', thr)
