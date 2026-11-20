"""Writes 160 noise PNGs (512x512, seed 20261114, 32 of each kind a-e: uniform, gaussian, solid colour, low-frequency, 8x8 colour blocks).
Usage: python noise_make.py OUT_DIR LIST_FILE"""
import os
import sys

import numpy as np
from PIL import Image

out, lst = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20261114)
N, S = 32, 512
u8 = lambda x: np.clip(np.rint(x), 0, 255).astype(np.uint8)  # noqa: E731
paths = []
for kind in 'abcde':
    for i in range(N):
        if kind == 'a':
            a = rng.integers(0, 256, (S, S, 3), dtype=np.uint8)
        elif kind == 'b':
            a = u8(rng.normal(128, 50, (S, S, 3)))
        elif kind == 'c':
            a = np.broadcast_to(rng.integers(0, 256, (1, 1, 3), dtype=np.uint8), (S, S, 3)).copy()
        elif kind == 'd':
            small = u8(rng.normal(128, 50, (16, 16, 3)))
            a = np.asarray(Image.fromarray(small).resize((S, S), Image.BICUBIC))
        else:
            blocks = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
            a = np.repeat(np.repeat(blocks, 64, axis=0), 64, axis=1)
        assert a.shape == (S, S, 3) and a.dtype == np.uint8
        p = f'{out}/{kind}_{i:02d}.png'
        Image.fromarray(a, 'RGB').save(p)
        paths.append(p)
open(lst, 'w').write('\n'.join(paths) + '\n')
print(len(paths), 'images written to', out)
