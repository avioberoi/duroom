"""Cut points of a tall gallery PDF: thin even-coloured rules between rows at 144 dpi, at most 820 pt apart; rules_y and MAXH are used by print_split_v3.
Usage: imported by print_split_v3.py"""
import os
from pathlib import Path

import fitz
import numpy as np

HERE = Path(os.environ['DUROOM_OUT']) / 'figures'
MAXH, DPI = 820.0, 144


def rules_y(path):
    pm = fitz.open(path)[0].get_pixmap(dpi=DPI)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, :3].astype(int)
    bg = np.array([241, 235, 224])
    ink = (np.abs(a - bg).max(2) > 12) & (np.abs(a - 255).max(2) > 12)
    full = ink.mean(1) >= 0.6
    ys, y = [], 0
    while y < pm.h:
        if not full[y]:
            y += 1
            continue
        y1 = y
        while y1 < pm.h and full[y1]:
            y1 += 1
        run = (y, y1)                                  # rows [y, y1) are full
        thin = (y1 - y) <= 3
        clear = (y - 3 >= 0 and not full[max(0, y - 3):y].any()) and (y1 + 3 <= pm.h and not full[y1:y1 + 3].any())
        if thin and clear:
            px = a[run[0]:run[1]][ink[run[0]:run[1]]]
            if len(px) and px.std(0).max() <= 18:
                ys.append((y + y1) / 2 * 72.0 / DPI)
        y = y1
    return ys
