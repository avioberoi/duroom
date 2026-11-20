"""Far and faithful pictures by way of asking as vertical bars, with the free search set apart, the loops, the lines aimed at real later works and the real works repainted for scale.
Usage: python ways_of_asking_vert_v4.py"""
import os

import ways_of_asking_titles_v1  # noqa: F401  (import sets the title on ways_of_asking_vert_v2)
import ways_of_asking_vert_v2 as vert
import ways_of_asking_v4 as W
from _style import *

FREE = "free search"
assert FREE not in W.ROWS
W.ROWS[FREE] = {"way": FREE, "kind": "free", "n": "5760", "far": "4094", "far_and_faithful": "0"}

ROWS = vert.LADDER[:6] + [None, ("free search,\n3 other\nstarts", FREE, "loop")] + vert.LADDER[6:]
SHORT = vert.SHORT[:6] + [None, "free\nsearch"] + vert.SHORT[6:]
assert len(ROWS) == len(SHORT) == 12 and ROWS[7][1] == FREE and ROWS[9][1] == W.REAL[0]


def desktop():
    fig, ax, xs, bracket, below = vert.chart(ROWS, (8.8, 5.3), [0.06, 0.29, 0.93, 0.56], 7.6, 10, 7.6, 0.62)   # 12 slots: smaller labels so they never touch
    bracket(3, 5, "DuROOM search and its 2 controls, 9,600 pictures each")
    bracket(7, 7, "own runs: no restart,\nno faithful check")
    bracket(9, 11, "real works repainted, for scale")
    bracket(1, 1, "hatched: uses later art")
    return fig


def mobile():
    rows = [None if r is None else (sh, r[1], r[2]) for r, sh in zip(ROWS, SHORT)]
    fig, ax, xs, bracket, below = vert.chart(rows, (3.9, 6.1), [0.09, 0.33, 0.89, 0.50], 6.0, 8.4, 5.3, 0.66,   # 12 slots: smaller counts so they never touch
                                             title_fs=11.5, title_lines=2, two_line_count=True)
    bracket(3, 5, "search and 2 controls")
    bracket(7, 7, "own\nruns")
    bracket(9, 11, "real works, for scale")
    return fig


if __name__ == "__main__":
    for name, fn in (("ways_of_asking_vert_v4", desktop), ("ways_of_asking_vert_v4-mobile", mobile)):
        assert os.environ.get("VERT4_ITER") or not (HERE / f"{name}.svg").exists(), name
        save(fn(), name)
