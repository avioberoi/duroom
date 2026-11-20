"""Gap per held-out group at 32 directions: all group means with direct labels.
Usage: python q1_gap_all.py"""
from collections import Counter
import numpy as np
from _panels import *

d = load_data("q1")
rows = [("time","Time"),("cluster","The encoder's own clusters"),("style","Styles"),("genre","Genres"),("artist","Artists")]
labels = {"artist":"Half the artists", "style:Ukiyo-e":"Ukiyo-e",
          "genre:religious painting":"Religious painting",
          "cluster:15":"East Asian prints and ink",
          "after_1900":"After 1900", "before_1900":"Before 1900",
          "after_1900_no_shared_artists":"After 1900, no shared artists"}
# Other labelled clusters are selected by their measured gap, not by guessed IDs.
for g in d["groups"]:
    if g["kind"] == "cluster" and g["gap"] < -4 and g["group"] not in labels:
        labels[g["group"]] = ("Renaissance religious painting" if "Renaissance" in g["label"] else "18th and 19th c. portraits")
for mobile in (False, True):
    fig, axes = plt.subplots(len(rows), 1, figsize=(3.9, 9.0) if mobile else (8.8, 5.5), sharex=True)
    for ax, (kind, name) in zip(axes, rows):
        groups = sorted((g for g in d["groups"] if g["kind"] == kind), key=lambda g:g["gap"])
        ax.axvline(0, color=HAIR, linewidth=0.8, zorder=0)
        ax.set_ylim(-0.7, 1.65 if mobile else 1.05)
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        panel_label(ax, f"{name} ({len(groups)} " + ("group)" if len(groups)==1 else "groups)"), mobile)
        buckets = Counter()
        highlights = [g for g in groups if g["group"] in labels]
        for g in groups:
            hot = g["group"] in labels
            bucket = round(g["gap"] / (0.32 if mobile else 0.14))
            rank = buckets[bucket]; buckets[bucket] += 1
            y = (0 if rank == 0 else ((rank+1)//2) * (1 if rank%2 else -1) * 0.085)
            color = ACCENT if hot and kind != "artist" else INK if hot else MUTE
            ax.scatter([g["gap"]],[y],s=25 if hot else 14,color=color,zorder=4,edgecolors=PAPER,linewidths=0.5)
            if hot:
                index = highlights.index(g)
                # Separate label lanes prevent neighboring time/cluster outliers colliding.
                if mobile:
                    label_y = 0.45 + index*0.46
                    label_x = -12.5
                    ha = "left"
                else:
                    label_y = 0.55 if index%2 == 0 else -0.48
                    label_x = g["gap"]
                    ha = "right" if g["gap"] > -1 else "left"
                text = labels[g["group"]] + "  " + points(g["gap"])
                if not mobile and g["group"] == "after_1900_no_shared_artists":
                    text = "After 1900, no shared artists\n" + points(g["gap"])
                    label_x = -12.5
                if not mobile and g["group"] == "before_1900":
                    label_x = -7.4
                ax.annotate(text, xy=(g["gap"],y), xytext=(label_x,label_y), fontsize=11 if mobile else 12,
                            color=INK,ha=ha,va="center",arrowprops={"arrowstyle":"-","color":HAIR,"lw":0.7})
        ax.set_xlim(-12.8, 1.7)
    ax = axes[-1]
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(HAIR)
    ax.tick_params(axis="x",bottom=True,labelbottom=True,length=3,labelsize=11)
    ax.set_xticks(np.arange(-12,1,4 if mobile else 2))
    ax.set_xlabel("Gap in percentage points\nBasis without the group − random basis",fontsize=11,labelpad=9)
    fig.subplots_adjust(left=0.035, right=0.98, bottom=0.085 if mobile else 0.12,
                        top=0.95, hspace=0.55 if mobile else 0.65)
    finish(fig, "q1_gap_all", mobile)
