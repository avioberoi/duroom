"""Shared look for every figure: palette, Palatino, save() that writes the SVG and PDF of a figure into $DUROOM_OUT/figures.
Usage: from _style import *; draw with INK/MUTE/ACCENT/PALE/DEEP; save(fig, "name")"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

if "DUROOM_OUT" not in os.environ:
    raise SystemExit("DUROOM_OUT is not set")
HERE = Path(os.environ["DUROOM_OUT"]) / "figures"   # data/, assets/, finds/ and every output figure live here
HERE.mkdir(parents=True, exist_ok=True)

PAPER, INK, MUTE, ACCENT = "#F1EBE0", "#231F1A", "#7A7062", "#963A27"
PALE = "#D2C8B6"
DEEP = "#E6DDCB"
HAIR = "#CFC5B3"

_names = {f.name for f in font_manager.fontManager.ttflist}
SERIF = "Palatino Linotype" if "Palatino Linotype" in _names else "serif"

plt.rcParams.update({
    "font.family": SERIF,
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "lines.linewidth": 0.8,
    "axes.linewidth": 0.6,
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "savefig.transparent": True,
    "axes.facecolor": "none",
    "savefig.pad_inches": 0.05,
})


def save(fig, name):
    """Write figures/<name>.svg and figures/<name>.pdf."""
    svg = HERE / f"{name}.svg"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    text = svg.read_text(encoding="utf-8")
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    svg.write_text(text, encoding="utf-8")
    fig.savefig(HERE / f"{name}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.svg and {name}.pdf")


def load_data(name):
    """Return figures/data/<name>.json as a dict."""
    import json
    p = HERE / "data" / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}: run the derive step that writes it")
    return json.loads(p.read_text(encoding="utf-8"))
