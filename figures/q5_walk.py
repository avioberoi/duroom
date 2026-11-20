"""Straight route against the decade path: works within reach, and their ratio, for three artist caps.
Usage: python q5_walk.py"""
from _panels import *
d=load_data('q5_walk');main=next(s for s in d['settings'] if s['cap']==d['default_cap'])
for mobile in (False,True):
    fig,(ax,bx)=plt.subplots(2,1,figsize=(3.9,7.1) if mobile else (8.8,6.2),sharex=True,gridspec_kw={'height_ratios':[1.35,1]})
    for route,color,label in [('path',INK,'Decade path'),('straight',ACCENT,'Straight route')]:
        rows=main[route];x=np.array([r['progress'] for r in rows])*100;y=np.array([r['count'] for r in rows]);sd=np.array([r['sd'] for r in rows])
        ax.fill_between(x,y-sd,y+sd,color=color,alpha=.10,lw=0)
        ax.plot(x,y,color=color,lw=1.3,marker='o',ms=3,label=label)
    ax.set_ylim(bottom=0);ax.set_ylabel('Works within reach',fontsize=11);ax.legend(frameon=False,loc='upper left',fontsize=10)
    ax.set_title(f'Counts · cap {d["default_cap"]} per artist',loc='left',fontsize=12,pad=12)
    bx.axhline(1,color=HAIR,lw=1)
    for s,color,ls in zip(d['settings'],[MUTE,ACCENT,INK],['--','-',(0,(2,2))]):
        bx.plot([100*r['progress'] for r in s['ratios']],[r['ratio'] for r in s['ratios']],color=color,ls=ls,lw=1.2,label=f'Cap {s["cap"]}' if s['cap'] else 'No cap')
    bx.set_ylim(0,1.2);bx.set_yticks([0,.5,1]);bx.set_ylabel('Straight / path count',fontsize=11)
    bx.legend(frameon=False,loc='lower right',fontsize=10,ncol=1 if mobile else 3)
    bx.set_xlabel('Progress from pre-1900 to 1960–1979 (%)',fontsize=10 if mobile else 11)
    bx.set_xlim(0,100);bx.set_xticks([0,25,50,75,100])
    for axis in (ax,bx):clean_axes(axis,mobile)
    fig.subplots_adjust(left=.19 if mobile else .12,right=.98,top=.91,bottom=.10,hspace=.34)
    finish(fig,'q5_walk',mobile)
