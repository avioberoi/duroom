"""Kept share against the number of directions for three groups, with matched scales.
Usage: python q1_kept_k.py"""
import numpy as np
from _panels import *

d = load_data("q1")
ks = np.array(d["ks"])
reference = d["reference_k"]
index = list(ks).index(reference)
panels = [("artist","Half the artists"),("after_1900","After 1900"),("cluster:15","East Asian prints and ink")]
for mobile in (False, True):
    fig, axes = plt.subplots(3 if mobile else 1, 1 if mobile else 3,
                             figsize=(3.9, 10.6) if mobile else (8.8, 3.7))
    for ax,(key,label) in zip(axes.flat,panels):
        c = d["curves"][key]
        unseen, random, null = [np.array(c[x])*100 for x in ("unseen","random","null")]
        ax.plot(ks,null,color=MUTE,lw=1.0,ls=(0,(4,3)))
        ax.plot(ks,random,color=MUTE,lw=1.5,marker="o",markersize=3,markerfacecolor=PAPER,markeredgewidth=0.7)
        ax.plot(ks,unseen,color=ACCENT,lw=1.6)
        ax.plot([reference,reference],[unseen[index],random[index]],color=INK,lw=1.2)
        ax.scatter([reference]*2,[unseen[index],random[index]],s=17,color=[ACCENT,MUTE],zorder=5)
        ax.set_xscale("log",base=2);ax.set_xlim(ks[0],ks[-1]);ax.set_ylim(0,105)
        ticks = ks[[0,3,5,7,9]]
        ax.set_xticks(ticks);ax.set_xticklabels([str(k) for k in ticks])
        ax.set_yticks([0,50,100]);clean_axes(ax,mobile)
        panel_label(ax,label,mobile)
        ax.set_xlabel("Directions in the basis",fontsize=11,labelpad=8)
        if mobile or key=="artist": ax.set_ylabel("Kept (%)",fontsize=11)
        # Label each curve where it can be read without covering the measurements.
        if mobile:
            ax.annotate("Random basis",xy=(ks[2],random[2]),xytext=(ks[0]*1.35,70),
                        fontsize=11,color=MUTE,arrowprops={"arrowstyle":"-","color":MUTE,"lw":0.6})
            ax.annotate("Basis without\nthe group",xy=(ks[6],unseen[6]),xytext=(ks[3],88),
                        fontsize=11,color=ACCENT,arrowprops={"arrowstyle":"-","color":ACCENT,"lw":0.6})
            ax.annotate("Shuffled",xy=(ks[6],null[6]),xytext=(ks[6],3),
                        fontsize=11,color=MUTE,arrowprops={"arrowstyle":"-","color":MUTE,"lw":0.6})
        ax.text(0, -0.35, f"Gap at {reference} directions: {points(unseen[index]-random[index])} points",
                transform=ax.transAxes,fontsize=11,color=INK,ha="left",va="top")
    if not mobile:
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([],[],color=ACCENT,label='Basis without the group'),
                            Line2D([],[],color=MUTE,marker='o',mfc=PAPER,label='Random basis'),
                            Line2D([],[],color=MUTE,ls='--',label='Shuffled')],
                   loc='upper center',bbox_to_anchor=(.52,1.04),ncol=3,frameon=False,fontsize=11)
    fig.subplots_adjust(left=0.15 if mobile else 0.065,right=0.98,top=0.91 if mobile else 0.83,
                        bottom=0.11 if mobile else 0.28,wspace=0.28,hspace=0.8)
    finish(fig,"q1_kept_k",mobile)
