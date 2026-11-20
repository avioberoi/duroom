"""Decade means of the later works in the first two new directions.
Usage: python q1b_path.py"""
from matplotlib.patches import Circle
from _panels import *

d = load_data('q1b')
scale = d['past_axis1_rms']
rows = d['plane']
past = np.array([[r['axis1'],r['axis2']] for r in rows if r['decade']<d['cutoff']])/scale
later = [r for r in rows if r['decade']>=d['cutoff']]
xy = np.array([[r['axis1'],r['axis2']] for r in later])/scale
labels = d['axis_labels']
for mobile in (False,True):
    fig, ax = plt.subplots(figsize=(4.2,4.9) if mobile else (8.8,5.9))
    ax.set_aspect('equal')
    ax.add_patch(Circle((0,0),1,fill=False,edgecolor=MUTE,lw=0.9))
    ax.axhline(0,color=HAIR,lw=0.6,zorder=0)
    ax.axvline(0,color=HAIR,lw=0.6,zorder=0)
    ax.scatter(past[:,0],past[:,1],s=13,color=MUTE,alpha=0.65,linewidth=0,zorder=2)
    ax.plot(xy[:,0],xy[:,1],color=INK,lw=1.0,zorder=3)
    ax.scatter(xy[:,0],xy[:,1],s=23,facecolor=PAPER,edgecolor=INK,lw=0.9,zorder=4)
    ax.scatter(*xy[0],s=30,color=ACCENT,zorder=5)
    # Label selected observations, never alter coordinates to tidy the path.
    offsets = [(10,-4) if mobile else (8,-12),(-35,10),(4,10),(9,7),(8,-14)]
    for year,offset in zip(d['label_decades'],offsets):
        index = next(i for i,r in enumerate(later) if r['decade']==year)
        ax.annotate(str(year),xy[index],xytext=offset,textcoords='offset points',
                    fontsize=10 if mobile else 11,color=ACCENT if year==d['cutoff'] else INK,
                    arrowprops={'arrowstyle':'-','color':HAIR,'lw':0.6})
    past_label = f'Before\n{d["cutoff"]}' if mobile else f'Before {d["cutoff"]}'
    ax.annotate(past_label,xy=(0,0),xytext=(-1.7,-1.45) if mobile else (-1.7,2.2),fontsize=10,color=MUTE,
                arrowprops={'arrowstyle':'-','color':MUTE,'lw':0.7},ha='left')
    ax.annotate('One axis-1 past spread',xy=(-0.7,-0.7),xytext=(1.4,-1.45),fontsize=10,color=MUTE,
                arrowprops={'arrowstyle':'-','color':MUTE,'lw':0.7},ha='left')
    ax.set_xlim(-1.9,float(xy[:,0].max())+1.1)
    ax.set_ylim(min(float(past[:,1].min())-0.3,-1.9),float(xy[:,1].max())+0.8)
    clean_axes(ax,mobile)
    ax.set_xlabel('Axis 1' if mobile else f'Axis 1\n{labels["axis1_positive"]} (+)\n{labels["axis1_negative"]} (−)',fontsize=11,labelpad=10)
    ax.set_ylabel('Axis 2' if mobile else f'Axis 2\n{labels["axis2_positive"]} (+)\n{labels["axis2_negative"]} (−)',fontsize=11,labelpad=10)
    ax.text(0,1.06,f'Units: past axis-1 RMS ({scale:.3f})',transform=ax.transAxes,fontsize=10,color=MUTE)
    fig.subplots_adjust(left=0.14 if mobile else 0.16,right=0.98,top=0.91,bottom=0.38 if mobile else 0.24)
    if mobile:
        meanings = (f'Axis 1: {labels["axis1_positive"]} (+)\n'
                    f'{labels["axis1_negative"]} (−)\n'
                    f'Axis 2: {labels["axis2_positive"]} (+)\n'
                    f'{labels["axis2_negative"]} (−)')
        fig.text(0.06, 0.04, meanings, fontsize=10, color=MUTE, linespacing=1.5)
    finish(fig,'q1b_path',mobile)
