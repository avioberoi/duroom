"""Counts of dated works by decade and the before/after 1900 totals, from counts/counts.json (data/counts.py), to figures/data/record_plot.json.
Usage: python derive_record.py"""
import json
import os
from pathlib import Path

OUT = Path(os.environ["DUROOM_OUT"])
HERE = OUT / "figures" / "data"
HERE.mkdir(parents=True, exist_ok=True)
SOURCE = OUT / "counts" / "counts.json"
SPLIT_YEAR = 1900
SNAPSHOT_YEAR = 2026  # the corpus snapshot's last year
d = json.loads(SOURCE.read_text(encoding="utf-8"))
decades = sorted((int(year), count) for year, count in d["by_decade"].items())
assert sum(count for _, count in decades) == d["named_dated"]
out = {
    "split_year": SPLIT_YEAR,
    "decades": [year for year, _ in decades], "counts": [count for _, count in decades],
    "total": d["named_dated"],
    "before": sum(count for year, count in decades if year < SPLIT_YEAR),
    "after": sum(count for year, count in decades if year >= SPLIT_YEAR),
    "last_year": SNAPSHOT_YEAR, "last_decade_years": SNAPSHOT_YEAR-decades[-1][0]+1,
}
(HERE / "record_plot.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("record_plot:", out["total"], "dated works;", out["before"], "/", out["after"])
