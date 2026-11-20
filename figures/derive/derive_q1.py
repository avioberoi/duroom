"""Gap per group at k = 32 and kept-against-k curves from common/clip-vit-l-14/heldout.json (cap 20, default basis size), to figures/data/q1.json.
Usage: python derive_q1.py"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(os.environ["DUROOM_OUT"])
HERE = OUT / "figures" / "data"
HERE.mkdir(parents=True, exist_ok=True)
SRC = OUT / "common" / "clip-vit-l-14" / "heldout.json"
d = json.loads(SRC.read_text(encoding="utf-8"))
K32 = d["ks"].index(32)
cells = defaultdict(list)
for r in d["rows"]:
    if r["cap"] == 20 and r["fit"] not in (1000, 5000, 10000):
        cells[r["group"]].append(r)
assert all(len(v) == 5 for v in cells.values()), {g: len(v) for g, v in cells.items() if len(v) != 5}


def name(g):
    if g.startswith("cluster:"):
        c = d["clusters"][g.split(":")[1]]
        return ", ".join(s for s, _ in c["styles"][:2])
    return {"artist": "half the artists", "after_1900": "after 1900", "before_1900": "before 1900",
            "after_1900_no_shared_artists": "after 1900, no shared artists"}.get(g, g.split(":", 1)[-1])


def kind(g):
    return "time" if "1900" in g else g.split(":")[0]


groups = []
for g, rs in cells.items():
    gap = [r["unseen"][K32] - r["random"][K32] for r in rs]
    null = [r["random2"][K32] - r["random"][K32] for r in rs]
    groups.append({"group": g, "label": name(g), "kind": kind(g), "n_test": rs[0]["n_test"],
                   "gap_draws": [100 * float(x) for x in gap],
                   "gap": 100 * float(np.mean(gap)), "sd": 100 * float(np.std(gap, ddof=1)), "null_sd": 100 * float(np.std(null, ddof=1))})
groups.sort(key=lambda x: x["gap"])

curves = {}
for g in ("artist", "after_1900", "cluster:15"):
    rs = cells[g]
    curves[g] = {"label": name(g), **{key: [float(np.mean([r[key][i] for r in rs])) for i in range(len(d["ks"]))]
                                      for key in ("unseen", "random", "null")}}

out = {"ks": d["ks"], "reference_k": d["ks"][K32],
       "cap": 20, "draws": len(next(iter(cells.values()))), "groups": groups, "curves": curves}
(HERE / "q1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for x in groups[:8] + groups[-3:]:
    print(f'{x["gap"]:6.1f}  {x["null_sd"]:.2f}  {x["kind"]:8s} {x["label"]}')
print(len(groups), "groups")
