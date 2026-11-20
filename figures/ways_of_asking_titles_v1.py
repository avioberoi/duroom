"""Writes ways_of_asking_vert_v2c and ways_of_asking_ladder_v2: the vertical ladder and the short ladder with the title that names the 6 kinds of request.
Usage: python ways_of_asking_titles_v1.py"""
import os

import ways_of_asking_vert_v2 as vert
import ways_of_asking_ladder_v1 as ladder
from _style import *

vert.TITLE = "46% of DuROOM search's pictures are far and faithful; with the 6 kinds of request and no search, 0.5%"
ladder.TITLE = "With the 6 kinds of request and no search, 0.5% of the pictures are far and faithful; DuROOM search paints 46%"

def ladder_mobile():
    """The ladder's phone layout with the title on 3 lines and the figure 0.2 in taller, so the plot keeps its size."""
    keep = ladder.TITLE
    ladder.TITLE = "With the 6 kinds of request and no search,\n0.5% of the pictures are far and faithful;\nDuROOM search paints 46%"
    try:
        H = 3.4 + 0.2
        return ladder.draw((3.9, H), [0.11, 0.25 * 3.4 / H, 0.87, 0.46 * 3.4 / H], 7.2, 8.4, 6.4, 0.5, 11.0, 3, True,
                           ["no search", "single\nwalk", "random-\nscore loop", "no-step\nloop", "DuROOM\nsearch"])
    finally:
        ladder.TITLE = keep


if __name__ == "__main__":
    for name, fn in (("ways_of_asking_vert_v2c", vert.ladder_desktop), ("ways_of_asking_vert_v2c-mobile", vert.ladder_mobile),
                     ("ways_of_asking_ladder_v2", ladder.desktop), ("ways_of_asking_ladder_v2-mobile", ladder_mobile)):
        assert os.environ.get("TITLES_ITER") or not (HERE / f"{name}.svg").exists(), name
        save(fn(), name)
