"""Retrodiction figure: share of works above the 99th- and 95th-percentile bars by decade; also the data/axes helpers of search_response_v2 and marks_v1. Reads data/search_retro.json.
Usage: python search_figures.py  (writes search_retro)"""
from _style import *
import numpy as np

def data(name): return load_data('search_'+name)
def axes(ax,xlabel='',ylabel=''):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(HAIR)
    ax.tick_params(length=3,color=HAIR,labelsize=8)
    ax.set_xlabel(xlabel,labelpad=9);ax.set_ylabel(ylabel,labelpad=9)
    ax.grid(axis='y',color=HAIR,lw=.4,alpha=.5)
    ax.set_axisbelow(True)
def title(ax,s): ax.set_title(s,loc='left',fontsize=11,pad=13)
def emit(fn,name):
    for mobile in (False,True): fn(mobile,name+('-mobile' if mobile else ''))

def retro(mobile,name):
    d=next(c for c in data('retro')['cuts'] if c['cutoff']==1900); rows=[r for r in d['decades'] if r['decade']<2020]
    fig,ax=plt.subplots(figsize=(3.9,4.3) if mobile else (7.5,3.6))
    x=[r['decade'] for r in rows]
    for key,label,c,ls in [('q99','99th-percentile bar',ACCENT,'-'),('q95','95th-percentile bar',MUTE,'--')]:
        y=[100*r[key] for r in rows];ax.plot(x,y,ls=ls,color=c,lw=1.4,marker='o',ms=3,label=label)
    axes(ax,'Decade','Works above the bar (%)');ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100]);ax.legend(frameon=False,fontsize=9.5 if mobile else 8,loc='upper left')
    ax.axvspan(1955,1965,color=DEEP,zorder=-1);title(ax,'The passing share rises'+chr(10)+'in the 1960s' if mobile else 'The passing share rises in the 1960s')
    if mobile:   # phone twin: labels at least 10.5 px when drawn 322 px wide
        ax.tick_params(labelsize=9.5);ax.xaxis.label.set_size(9.5);ax.yaxis.label.set_size(9.5);ax.title.set_size(11.5)
        ax.set_xticks([1900,1940,1980,2010],['1900','1940','1980','2010'])
    fig.tight_layout();save(fig,name)

if __name__=='__main__':
    emit(retro,'search_retro')
