"""Short ladder of five upright bars in the order no search, single walk, DuROOM search, random-score loop, no-step loop.
Usage: python ways_of_asking_ladder_v3.py"""
import os

import ways_of_asking_titles_v1 as T   # import sets the titles on ways_of_asking_ladder_v1 and vert_v2
import ways_of_asking_ladder_v1 as ladder
import ways_of_asking_vert_v2 as V
from _style import *

ORDER = ["all six recipes", "single walk with selection", "search", "random-score loop", "no-step loop"]
byway = {w: (n, w, s) for n, w, s in ladder.ROWS}
ladder.ROWS = [byway[w] for w in ORDER]
assert [r[0] for r in ladder.ROWS] == ["no search, 6 kinds", "single walk", "DuROOM search", "random-score loop", "no-step loop"]


def desktop():
    return ladder.desktop()


def mobile():
    """phone layout: 3-line title, figure 0.2 in taller"""
    keep = ladder.TITLE
    ladder.TITLE = "With the 6 kinds of request and no search,\n0.5% of the pictures are far and faithful;\nDuROOM search paints 46%"
    try:
        H = 3.4 + 0.2
        return ladder.draw((3.9, H), [0.11, 0.25 * 3.4 / H, 0.87, 0.46 * 3.4 / H], 7.2, 8.4, 6.4, 0.5, 11.0, 3, True,
                           ["no search", "single\nwalk", "DuROOM\nsearch", "random-\nscore loop", "no-step\nloop"])
    finally:
        ladder.TITLE = keep


if __name__ == "__main__":
    for name, fn in (("ways_of_asking_ladder_v3", desktop), ("ways_of_asking_ladder_v3-mobile", mobile)):
        assert os.environ.get("LADDER3_ITER") or not (HERE / f"{name}.svg").exists(), name
        save(fn(), name)
