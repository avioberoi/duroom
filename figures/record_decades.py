"""Dated works by decade, from record_plot.json.
Usage: python record_decades.py"""
import numpy as np
from matplotlib.ticker import MaxNLocator, FuncFormatter
from _panels import *

d = load_data("record_plot")
dec, count = np.array(d["decades"]), np.array(d["counts"])
split = d["split_year"]
for mobile in (False, True):
    fig, ax = plt.subplots(figsize=(3.9, 3.5) if mobile else (8.8, 3.5))
    fig.subplots_adjust(left=0.15 if mobile else 0.09, right=0.985, bottom=0.19, top=0.62)
    fig.text(0.15 if mobile else 0.09, 0.96, f'Before {split}', fontsize=11, va="top", color=MUTE)
    fig.text(0.15 if mobile else 0.09, 0.86, f'{d["before"]:,} works', fontsize=14, va="top", color=INK)
    fig.text(0.61, 0.96, f'{split} onward', fontsize=11, va="top", color=MUTE)
    fig.text(0.61, 0.86, f'{d["after"]:,} works', fontsize=14, va="top", color=INK)
    step = float(np.median(np.diff(dec)))
    ax.bar(dec + step / 2, count, width=step * 0.85, color=np.where(dec < split, PALE, INK), linewidth=0)
    ax.bar(dec[-1]+step/2,count[-1],width=step*.85,color=PAPER,edgecolor=ACCENT,hatch='///',linewidth=.7)
    fig.text(.985,.74,f'{dec[-1]}–{d["last_year"]}\n{count[-1]:,} works · {d["last_decade_years"]} years',ha='right',va='top',fontsize=9 if mobile else 10,color=ACCENT)
    ax.axvline(split, color=ACCENT, linewidth=0.9)
    ax.set_xlim(dec.min(), dec.max() + step)
    ax.set_ylim(0, count.max() * 1.10)
    ticks = dec[(dec - dec.min()) % (400 if mobile else 200) == 0]
    ax.set_xticks(np.append(ticks, dec[-1]) if mobile else ticks)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
    clean_axes(ax,mobile)
    ax.set_xlabel("Year", fontsize=11)
    fig.text(.15 if mobile else .09,.74,"Works per decade",fontsize=10,color=MUTE,va='top')
    finish(fig, "record_decades", mobile)
